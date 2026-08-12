"""Attention-weight explainer for sequence models.

Extracts transformer attention weights from a
:class:`~visxai.models.pytorch_wrapper.PyTorchSequenceWrapper`-wrapped model
and maps them to atoms via
:attr:`~visxai.core.data_types.MoleculeRepresentation.token_to_atom_map`, and
to bonds via
:attr:`~visxai.core.data_types.MoleculeRepresentation.token_to_bond_map`
where available.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from visxai.core.base_explainer import BaseExplainer
from visxai.core.base_model import BaseModelWrapper
from visxai.core.data_types import Explanation, MoleculeRepresentation
from visxai.models.pytorch_wrapper import PyTorchSequenceWrapper
from visxai.utils.mapping import (
    aggregate_token_scores_to_atoms,
    aggregate_token_scores_to_bonds,
    build_token_provenance,
)


def _last_layer_query_attention(
    attentions: tuple[torch.Tensor, ...],
    query_token_index: int,
) -> np.ndarray:
    """Attention from ``query_token_index`` to every token, last layer, head-averaged.

    Parameters
    ----------
    attentions : tuple[torch.Tensor, ...]
        Per-layer attention tensors, each of shape
        ``(1, num_heads, seq_len, seq_len)``.
    query_token_index : int
        Token position treated as the summarizing query (e.g. ``0`` for a
        ``[CLS]``-style token).

    Returns
    -------
    numpy.ndarray
        1-D array of shape ``(seq_len,)`` with one score per token position.
    """
    last_layer = attentions[-1]  # (1, num_heads, seq_len, seq_len)
    avg_heads = last_layer.mean(dim=1)  # (1, seq_len, seq_len)
    query_row = avg_heads[0, query_token_index, :]  # (seq_len,)
    return query_row.detach().cpu().numpy()


def _attention_rollout(attentions: tuple[torch.Tensor, ...]) -> np.ndarray:
    """Full attention-rollout across all layers (Abnar & Zuidema, 2020).

    Averages heads per layer, adds an identity term to account for the
    residual connection, row-normalizes, and multiplies the resulting
    matrices across layers. Relevance is read off relative to token
    position ``0`` (e.g. a ``[CLS]``-style summarizing token).

    Parameters
    ----------
    attentions : tuple[torch.Tensor, ...]
        Per-layer attention tensors, each of shape
        ``(1, num_heads, seq_len, seq_len)``.

    Returns
    -------
    numpy.ndarray
        1-D array of shape ``(seq_len,)`` with one score per token position.
    """
    seq_len = attentions[0].shape[-1]
    device = attentions[0].device
    identity = torch.eye(seq_len, device=device)

    rollout = identity
    for layer_attn in attentions:
        avg_heads = layer_attn.mean(dim=1)[0]  # (seq_len, seq_len)
        avg_heads = avg_heads + identity
        avg_heads = avg_heads / avg_heads.sum(dim=-1, keepdim=True)
        rollout = avg_heads @ rollout

    token_scores = rollout[0, :]
    return token_scores.detach().cpu().numpy()


class AttentionExplainer(BaseExplainer):
    """Compute per-atom and per-bond XAI scores from transformer attention weights.

    Extracted token-level attention scores are summed onto every atom each
    token maps to, via
    :attr:`~visxai.core.data_types.MoleculeRepresentation.token_to_atom_map`.
    Atoms covered by no token receive a score of ``0.0``.

    When ``mol_rep.token_to_bond_map`` is available, token scores are also
    summed onto bonds. Most bonds have no explicit character in the SMILES
    string (a plain single or aromatic bond is implicit) — only bonds with
    an explicit symbol (``=``, ``#``, a ring closure, etc.) can receive a
    score, and this covers roughly 10-25% of bonds on real drug molecules.
    Rather than discard that partial signal, bonds that *do* have a
    contributing token are populated in ``bond_scores``; bonds with no
    contributing token are **omitted entirely** rather than filled with
    ``0.0`` — "no character to measure" is kept visually and semantically
    distinct from "measured, and it came out to zero."

    Raw attention weights are non-negative (softmax output), unlike SHAP or
    Integrated Gradients. :class:`~visxai.visualizers.rdkit_2d.RDKitSVGVisualizer`
    already handles an all-non-negative score range correctly (it renders a
    white-to-green gradient when the minimum score is ``0``), so no
    visualizer changes are needed to display these explanations.

    Parameters
    ----------
    strategy : {"last_layer_mean_heads", "rollout"}, optional
        Aggregation strategy. ``"last_layer_mean_heads"`` (default) uses
        the last layer's attention, averaged across heads, from
        ``query_token_index`` to every other token — the simplest
        defensible default, treating that token as a summarizing query.
        ``"rollout"`` computes full attention-rollout across all layers
        (Abnar & Zuidema), relative to token position ``0``.
    query_token_index : int, optional
        Token position treated as the summarizing query. Only used by the
        ``"last_layer_mean_heads"`` strategy. Defaults to ``0`` (e.g. a
        ``[CLS]``-style token).

    Attributes
    ----------
    _strategy : str
        Stored aggregation strategy.
    _query_token_index : int
        Stored query token position.

    Examples
    --------
    >>> explainer = AttentionExplainer()
    >>> explanation = explainer.explain(pytorch_sequence_wrapper, mol_rep)
    >>> explanation.atom_scores  # {atom_idx: attention_score, ...}
    """

    def __init__(
        self,
        strategy: Literal["last_layer_mean_heads", "rollout"] = "last_layer_mean_heads",
        query_token_index: int = 0,
    ) -> None:
        self._strategy: Literal["last_layer_mean_heads", "rollout"] = strategy
        self._query_token_index: int = query_token_index

    def explain(
        self,
        model: BaseModelWrapper,
        mol_rep: MoleculeRepresentation,
    ) -> Explanation:
        """Compute atom-level attention attributions for a single molecule.

        Parameters
        ----------
        model : BaseModelWrapper
            Must be a
            :class:`~visxai.models.pytorch_wrapper.PyTorchSequenceWrapper`
            wrapping a model whose ``forward`` accepts
            ``output_attentions=True`` and returns an object exposing
            ``.attentions``: a tuple of per-layer tensors of shape
            ``(1, num_heads, seq_len, seq_len)``.
        mol_rep : MoleculeRepresentation
            Featurised molecule. ``token_ids``, ``attention_mask``, and
            ``token_to_atom_map`` must all be populated (use
            :func:`visxai.features.sequences.generate_sequence_representation`).
            ``token_to_bond_map`` is consulted if present (also populated
            by the same function) to additionally compute ``bond_scores``.

        Returns
        -------
        Explanation
            Dataclass with:

            - ``atom_scores`` — every RDKit atom index mapped to its
              cumulative attention contribution. Atoms not covered by any
              token receive a score of ``0.0``.
            - ``bond_scores`` — RDKit bond index mapped to cumulative
              attention contribution, for the sparse subset of bonds with
              an explicit SMILES character (see
              :func:`visxai.features.sequences.compute_bond_char_spans`).
              Empty if ``mol_rep.token_to_bond_map`` is ``None``. Bonds
              with no contributing token are omitted, not zero-filled.
            - ``metadata`` — contains ``"strategy"`` (the aggregation
              strategy used) and ``"predicted_value"`` (the model's raw
              prediction).

        Raises
        ------
        TypeError
            If ``model`` is not a
            :class:`~visxai.models.pytorch_wrapper.PyTorchSequenceWrapper`.
        ValueError
            If the wrapped model does not return attention weights.
        """
        if not isinstance(model, PyTorchSequenceWrapper):
            raise TypeError(
                f"AttentionExplainer requires a PyTorchSequenceWrapper, "
                f"got {type(model).__name__}"
            )

        input_ids = torch.tensor([mol_rep.token_ids], dtype=torch.long)
        attention_mask = torch.tensor([mol_rep.attention_mask], dtype=torch.long)

        model.model.eval()
        with torch.no_grad():
            out = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )

        attentions = getattr(out, "attentions", None)
        if not attentions:
            raise ValueError(
                "The wrapped model did not return attention weights; "
                "its forward() must accept output_attentions=True and "
                "return an object exposing a non-empty .attentions tuple."
            )

        if self._strategy == "last_layer_mean_heads":
            token_scores = _last_layer_query_attention(
                attentions, self._query_token_index
            )
        else:
            token_scores = _attention_rollout(attentions)

        atom_scores = aggregate_token_scores_to_atoms(
            token_scores, mol_rep.token_to_atom_map, mol_rep.mol.GetNumAtoms()
        )
        bond_scores = (
            aggregate_token_scores_to_bonds(token_scores, mol_rep.token_to_bond_map)
            if mol_rep.token_to_bond_map is not None
            else {}
        )

        return Explanation(
            atom_scores=atom_scores,
            bond_scores=bond_scores,
            metadata={
                "strategy": self._strategy,
                "predicted_value": float(model.predict(mol_rep)[0]),
                "attribution": "sequence/duplicate",
            },
            atom_provenance=build_token_provenance(
                token_scores, mol_rep.token_to_atom_map
            ),
            bond_provenance=(
                build_token_provenance(token_scores, mol_rep.token_to_bond_map)
                if mol_rep.token_to_bond_map is not None
                else {}
            ),
        )
