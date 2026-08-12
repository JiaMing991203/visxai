"""Gradient-based explainers for PyTorch sequence (SMILES/CNN) models.

Provides two explainers for :class:`~visxai.models.pytorch_wrapper.PyTorchSequenceWrapper`-
wrapped models, both producing per-token scores that are then aggregated
onto atoms and bonds via :func:`visxai.utils.mapping.aggregate_token_scores_to_atoms`/
:func:`~visxai.utils.mapping.aggregate_token_scores_to_bonds` — the same
aggregation :class:`~visxai.explainers.attention.AttentionExplainer` uses,
since that logic is agnostic to *how* a per-token score was produced (raw
attention weight, summed Integrated Gradients attribution, or Grad-CAM
channel-weighted activation).

- :class:`IntegratedGradientsExplainer` — ``input_ids`` are discrete token
  indices, so plain ``captum.attr.IntegratedGradients`` (which interpolates
  a continuous tensor between baseline and input) can't attribute them
  directly. ``captum.attr.LayerIntegratedGradients`` is Captum's own
  documented pattern for exactly this: it hooks a caller-specified
  embedding layer, interpolates *that layer's continuous output* between
  the real and baseline token ids' embeddings, and attributes back to
  that output. ``embedding_layer`` is a required, explicit constructor
  argument — the same "don't guess a model-specific detail" principle
  applied to :class:`GradCAMExplainer`'s ``target_layer`` below and the
  graph path's ``atom_featurizer``/``target_layer`` (see the README's
  architecture principle) — there is no reliable, convention-free way to
  identify "the embedding layer" generically across arbitrary model
  architectures.
- :class:`GradCAMExplainer` — the sequence adaptation of Grad-CAM for a
  1-D convolutional layer. Hand-rolled with plain PyTorch forward/backward
  hooks, mirroring :mod:`visxai.explainers.gradient_based`'s graph-path
  Grad-CAM (and for the same reason: Captum's default ``LayerGradCam``
  treats a 2-D-or-higher activation's leading (batch) dimension as
  independent examples with no shared cross-position weight, which is a
  real deviation from Grad-CAM's defining class-discriminative global
  weighting). A ``Conv1d`` activation is ``(batch, channels, seq_len)`` —
  one more non-channel dimension than the graph path's ``(n_nodes,
  hidden_dim)`` — so the canonical "average the gradient over every
  non-channel dimension" step averages over *both* batch and seq_len here.

Unlike the graph path, bond-level attribution is **not opt-in** for either
explainer here: both explainers already produce one score per token
position as an intermediate result, and bond aggregation is just a second
call to the same per-token array through
:func:`~visxai.utils.mapping.aggregate_token_scores_to_bonds` — there is no
separate edge-indexed activation/feature tensor to opt into the way the
graph path's ``uses_edge_attr``/``edge_target_layer`` require. Bond
coverage is inherently **partial** here regardless (most SMILES bonds have
no explicit character to attribute a token to — see
:mod:`visxai.explainers.attention`'s module docstring), the same
"omitted, not zero-filled" convention ``AttentionExplainer`` already uses.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from captum.attr import LayerIntegratedGradients

from visxai.core.base_explainer import BaseExplainer
from visxai.core.base_model import BaseModelWrapper
from visxai.core.data_types import Explanation, MoleculeRepresentation
from visxai.models.pytorch_wrapper import PyTorchSequenceWrapper
from visxai.utils.mapping import (
    aggregate_token_scores_to_atoms,
    aggregate_token_scores_to_bonds,
    build_token_provenance,
)


def _raw_model_output(
    model: PyTorchSequenceWrapper,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Call the wrapped model directly and extract its raw prediction tensor.

    Mirrors :meth:`~visxai.models.pytorch_wrapper.PyTorchSequenceWrapper.predict`'s
    output-attribute extraction, but without ``torch.no_grad()`` — both
    explainers in this module need gradients to flow through the call,
    the same reason :class:`~visxai.explainers.attention.AttentionExplainer`
    calls ``model.model(...)`` directly instead of ``model.predict(...)``.
    """
    out = model.model(input_ids=input_ids, attention_mask=attention_mask)
    return getattr(out, model._output_attr, out)


def _to_token_arrays(mol_rep: MoleculeRepresentation) -> tuple[torch.Tensor, torch.Tensor]:
    """Build batch-size-1 ``input_ids``/``attention_mask`` tensors from a molecule."""
    input_ids = torch.tensor([mol_rep.token_ids], dtype=torch.long)
    attention_mask = torch.tensor([mol_rep.attention_mask], dtype=torch.long)
    return input_ids, attention_mask


def _integrated_gradients_token_scores(
    model: PyTorchSequenceWrapper,
    embedding_layer: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    baseline_token_id: int,
    n_steps: int,
) -> np.ndarray:
    """Run ``LayerIntegratedGradients`` and return one score per token position.

    Factored out of :meth:`IntegratedGradientsExplainer.explain` so the raw
    per-token array — the granularity IG's completeness axiom actually
    holds at here (see :class:`IntegratedGradientsExplainer`'s docstring)
    — is independently testable, the same way
    :func:`visxai.explainers.attention._last_layer_query_attention`/
    :func:`~visxai.explainers.attention._attention_rollout` are testable
    module-level functions rather than inlined into ``explain()``.

    Returns
    -------
    numpy.ndarray
        1-D array of shape ``(seq_len,)``, one score per token position
        (attributions summed across the embedding dimension).
    """
    baseline_ids = torch.full_like(input_ids, baseline_token_id)

    def forward_fn(ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return _raw_model_output(model, ids, mask)

    lig = LayerIntegratedGradients(forward_fn, embedding_layer)
    attributions = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        n_steps=n_steps,
    )
    return attributions.sum(dim=-1)[0].detach().cpu().numpy()


def _gradcam_token_scores(
    model: PyTorchSequenceWrapper,
    target_layer: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> np.ndarray:
    """Run the hand-rolled Grad-CAM hook/backward pass and return one score per token position.

    Factored out of :meth:`GradCAMExplainer.explain` for the same
    independent-testability reason as
    :func:`_integrated_gradients_token_scores`.

    Returns
    -------
    numpy.ndarray
        1-D array of shape ``(seq_len,)``, one non-negative score per token
        position.

    Raises
    ------
    ValueError
        If no gradient reaches ``target_layer``'s output, or its
        activation isn't a ``(1, channels, seq_len)`` tensor.
    """
    seq_len = input_ids.shape[1]
    captured: dict[str, torch.Tensor] = {}

    def forward_hook(
        _module: torch.nn.Module,
        _input: tuple,
        output: torch.Tensor,
    ) -> None:
        output.retain_grad()
        captured["activation"] = output

    handle = target_layer.register_forward_hook(forward_hook)
    try:
        model.model.eval()
        out = _raw_model_output(model, input_ids, attention_mask)
        model.model.zero_grad()
        out.sum().backward()
    finally:
        handle.remove()

    activation = captured.get("activation")
    if activation is None or activation.grad is None:
        raise ValueError(
            "No gradient reached target_layer's output; it may not "
            "have been invoked during the forward pass."
        )
    if activation.dim() != 3 or activation.shape[0] != 1 or activation.shape[2] != seq_len:
        raise ValueError(
            f"target_layer's activation must have shape "
            f"(1, channels, seq_len) with seq_len={seq_len}, got "
            f"{tuple(activation.shape)}; GradCAMExplainer requires a "
            f"Conv1d-style layer whose output is (batch, channels, "
            f"seq_len)."
        )

    alpha = activation.grad.mean(dim=(0, 2))  # (channels,), shared across positions
    cam = torch.relu((activation * alpha.view(1, -1, 1)).sum(dim=1))  # (1, seq_len)
    return cam[0].detach().cpu().numpy()


class IntegratedGradientsExplainer(BaseExplainer):
    """Compute per-atom and per-bond XAI scores using Integrated Gradients on token embeddings.

    Attributes ``input_ids`` via ``captum.attr.LayerIntegratedGradients``
    targeting a caller-specified embedding layer: Captum interpolates the
    embedding layer's *output* between the real tokens' embeddings and an
    all-``baseline_token_id`` baseline's embeddings, then attributes back
    to that continuous output. Per-token scores (summed across the
    embedding dimension) are then aggregated onto atoms (dense, every atom
    gets an entry) and bonds (sparse, only bonds with a token-covered
    character get an entry) via
    :func:`visxai.utils.mapping.aggregate_token_scores_to_atoms`/
    :func:`~visxai.utils.mapping.aggregate_token_scores_to_bonds` — the
    same aggregation :class:`~visxai.explainers.attention.AttentionExplainer`
    uses.

    **``sum(atom_scores) + sum(bond_scores)`` does not satisfy IG's
    completeness axiom against the model's raw prediction difference** —
    unlike the graph path's ``IntegratedGradientsExplainer``, where node
    ``i`` = atom ``i`` exactly (every node is an atom, nothing is
    dropped). Here, completeness genuinely holds at the **token**
    granularity (verified via Captum's own ``return_convergence_delta``
    against ``predicted_value - model(baseline_input_ids)``, see
    the test suite), but aggregating onto atoms
    necessarily drops whatever attribution mass fell on tokens with no
    atom mapping at all — special tokens like ``[CLS]``/``[SEP]`` (see
    :func:`visxai.features.sequences.default_smiles_tokenizer`), which
    always exist and always have an empty ``token_to_atom_map`` entry.
    This is the same "no entry = no information available" convention
    :class:`~visxai.explainers.attention.AttentionExplainer` already uses
    for bonds, just also true for atoms here since — unlike attention
    weights, which were never claimed to sum to anything in particular —
    IG's raw per-token array *does* satisfy a real conservation law that
    the atom/bond aggregation step then partially discards.

    Parameters
    ----------
    embedding_layer : torch.nn.Module
        The wrapped model's token-embedding layer (a submodule of
        ``model.model``, typically a ``torch.nn.Embedding``). Required,
        with no auto-detection — there is no reliable, convention-free way
        to identify "the embedding layer" generically across arbitrary
        model architectures.
    baseline_token_id : int, optional
        Token id used to build the all-``baseline_token_id`` baseline
        ``input_ids`` tensor Integrated Gradients interpolates from.
        Defaults to ``0`` (this package's reference tokenizer's
        ``[PAD]`` id — see
        :func:`visxai.features.sequences.default_smiles_tokenizer`). A
        model trained with a real tokenizer whose pad id differs should
        pass that id explicitly.
    n_steps : int, optional
        Number of interpolation steps between the baseline and the input.
        Defaults to ``50``, Captum's own common default.

    Attributes
    ----------
    _embedding_layer : torch.nn.Module
        Stored embedding layer.
    _baseline_token_id : int
        Stored baseline token id.
    _n_steps : int
        Stored interpolation step count.

    Examples
    --------
    >>> explainer = IntegratedGradientsExplainer(embedding_layer=cnn_model.embedding)
    >>> explanation = explainer.explain(pytorch_sequence_wrapper, mol_rep)
    >>> explanation.atom_scores  # {atom_idx: ig_score, ...}
    """

    def __init__(
        self,
        embedding_layer: torch.nn.Module,
        baseline_token_id: int = 0,
        n_steps: int = 50,
    ) -> None:
        self._embedding_layer: torch.nn.Module = embedding_layer
        self._baseline_token_id: int = baseline_token_id
        self._n_steps: int = n_steps

    def explain(
        self,
        model: BaseModelWrapper,
        mol_rep: MoleculeRepresentation,
    ) -> Explanation:
        """Compute atom-level and bond-level IG attributions for a molecule.

        Parameters
        ----------
        model : BaseModelWrapper
            Must be a
            :class:`~visxai.models.pytorch_wrapper.PyTorchSequenceWrapper`
            wrapping a model whose ``embedding_layer`` (passed to this
            explainer's constructor) is one of its own submodules.
        mol_rep : MoleculeRepresentation
            Featurised molecule. ``token_ids``, ``attention_mask``, and
            ``token_to_atom_map`` must all be populated (use
            :func:`visxai.features.sequences.generate_sequence_representation`).
            ``token_to_bond_map`` is consulted if present to additionally
            compute ``bond_scores``.

        Returns
        -------
        Explanation
            Dataclass with:

            - ``atom_scores`` — every RDKit atom index mapped to its
              Integrated Gradients contribution. Atoms not covered by any
              token receive a score of ``0.0``.
            - ``bond_scores`` — RDKit bond index mapped to cumulative
              Integrated Gradients contribution, for the sparse subset of
              bonds with an explicit SMILES character. Empty if
              ``mol_rep.token_to_bond_map`` is ``None``.
            - ``metadata`` — contains ``"n_steps"`` (the interpolation step
              count used) and ``"predicted_value"`` (the model's raw
              prediction).

        Raises
        ------
        TypeError
            If ``model`` is not a
            :class:`~visxai.models.pytorch_wrapper.PyTorchSequenceWrapper`.
        ValueError
            If ``embedding_layer`` is not a submodule of ``model.model``.
        """
        if not isinstance(model, PyTorchSequenceWrapper):
            raise TypeError(
                f"IntegratedGradientsExplainer requires a PyTorchSequenceWrapper, "
                f"got {type(model).__name__}"
            )
        if self._embedding_layer not in model.model.modules():
            raise ValueError(
                "embedding_layer must be a submodule of the wrapped model "
                "(model.model); it was not found among model.model.modules()."
            )

        input_ids, attention_mask = _to_token_arrays(mol_rep)
        model.model.eval()

        token_scores = _integrated_gradients_token_scores(
            model,
            self._embedding_layer,
            input_ids,
            attention_mask,
            self._baseline_token_id,
            self._n_steps,
        )

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
                "n_steps": self._n_steps,
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


class GradCAMExplainer(BaseExplainer):
    """Compute per-atom and per-bond XAI scores for CNNs using the sequence adaptation of Grad-CAM.

    Follows the same canonical Grad-CAM formulation as
    :class:`visxai.explainers.gradient_based.GradCAMExplainer` (the graph
    path): a single per-channel importance weight is computed by averaging
    that channel's gradient across every non-channel dimension of the
    target layer's activation, then applied uniformly across the
    activation to get one non-negative score per position:

    .. code-block:: text

        alpha_c = mean over batch and seq_len of grad(output, activation)[:, c, :]
        score_t = ReLU( sum_c alpha_c * activation[0, c, t] )

    A ``torch.nn.Conv1d`` layer's activation is ``(batch, channels,
    seq_len)`` — one more non-channel dimension (``seq_len``) than the
    graph path's ``(n_nodes, hidden_dim)``, so ``alpha`` averages over both
    batch and seq_len instead of just the leading dimension. This gives
    one score per **token position**, structurally identical to
    :class:`~visxai.explainers.attention.AttentionExplainer`'s
    ``token_scores`` — so the same
    :func:`visxai.utils.mapping.aggregate_token_scores_to_atoms`/
    :func:`~visxai.utils.mapping.aggregate_token_scores_to_bonds` helpers
    apply unchanged.

    Hand-rolled with plain PyTorch forward/backward hooks rather than
    wrapping ``captum.attr.LayerGradCam``, for the same reason as the graph
    path: Captum's default treats every non-leading dimension independently
    with no shared cross-position weight, a real deviation from Grad-CAM's
    defining class-discriminative global weighting.

    Scores are **non-negative** (the ``ReLU`` is part of the algorithm's
    definition) — like :class:`~visxai.explainers.attention.AttentionExplainer`'s
    raw attention scores, and unlike :class:`IntegratedGradientsExplainer`'s
    signed scores. :class:`~visxai.visualizers.rdkit_2d.RDKitSVGVisualizer`
    already renders an all-non-negative score range correctly, so no
    visualizer changes are needed.

    Parameters
    ----------
    target_layer : torch.nn.Module
        The model's ``Conv1d`` layer (a submodule of ``model.model``) whose
        activation Grad-CAM is computed over. Required, with no
        auto-detection. Must produce a 3-D tensor of shape ``(1, channels,
        seq_len)``.

    Attributes
    ----------
    _target_layer : torch.nn.Module
        Stored target layer.

    Examples
    --------
    >>> explainer = GradCAMExplainer(target_layer=cnn_model.conv2)
    >>> explanation = explainer.explain(pytorch_sequence_wrapper, mol_rep)
    >>> explanation.atom_scores  # {atom_idx: cam_score >= 0.0, ...}
    """

    def __init__(self, target_layer: torch.nn.Module) -> None:
        self._target_layer: torch.nn.Module = target_layer

    def explain(
        self,
        model: BaseModelWrapper,
        mol_rep: MoleculeRepresentation,
    ) -> Explanation:
        """Compute atom-level and bond-level Grad-CAM attributions for a molecule.

        Parameters
        ----------
        model : BaseModelWrapper
            Must be a
            :class:`~visxai.models.pytorch_wrapper.PyTorchSequenceWrapper`
            wrapping a model whose ``target_layer`` (passed to this
            explainer's constructor) is one of its own submodules.
        mol_rep : MoleculeRepresentation
            Featurised molecule. ``token_ids``, ``attention_mask``, and
            ``token_to_atom_map`` must all be populated. ``token_to_bond_map``
            is consulted if present to additionally compute ``bond_scores``.

        Returns
        -------
        Explanation
            Dataclass with:

            - ``atom_scores`` — every RDKit atom index mapped to its
              (non-negative) Grad-CAM score.
            - ``bond_scores`` — RDKit bond index mapped to its
              (non-negative) Grad-CAM score, for the sparse subset of bonds
              with an explicit SMILES character. Empty if
              ``mol_rep.token_to_bond_map`` is ``None``.
            - ``metadata`` — contains ``"predicted_value"`` (the model's
              raw prediction).

        Raises
        ------
        TypeError
            If ``model`` is not a
            :class:`~visxai.models.pytorch_wrapper.PyTorchSequenceWrapper`.
        ValueError
            If ``target_layer`` is not a submodule of ``model.model``; if
            no gradient reaches ``target_layer``'s output (e.g. it was
            never invoked during the forward pass); or if its activation
            isn't a 3-D ``(1, channels, seq_len)`` tensor.
        """
        if not isinstance(model, PyTorchSequenceWrapper):
            raise TypeError(
                f"GradCAMExplainer requires a PyTorchSequenceWrapper, "
                f"got {type(model).__name__}"
            )
        if self._target_layer not in model.model.modules():
            raise ValueError(
                "target_layer must be a submodule of the wrapped model "
                "(model.model); it was not found among model.model.modules()."
            )

        input_ids, attention_mask = _to_token_arrays(mol_rep)
        token_scores = _gradcam_token_scores(
            model, self._target_layer, input_ids, attention_mask
        )

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
