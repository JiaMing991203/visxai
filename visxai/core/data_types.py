"""Shared dataclasses for molecule representations and XAI explanations.

This module defines the two central data containers that flow through every
VisXAI pipeline:

- :class:`MoleculeRepresentation` — stores all featurised views of a single
  molecule (SMILES, RDKit Mol, fingerprint array, and bit-to-atom metadata).
- :class:`Explanation` — stores the XAI output as per-atom and per-bond
  scores, plus free-form metadata.

A third, smaller type — :class:`ScoreContribution` — itemises *how* an
individual score was assembled, so a visualizer can explain a number rather
than only display it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, TypedDict

import numpy as np
from rdkit.Chem import Mol


@dataclass
class MoleculeRepresentation:
    """All featurised views of a single molecule.

    This dataclass is the single source of truth passed between feature
    extractors, model wrappers, and explainers.  The ``bit_info`` field is
    critical for the fingerprint-to-atom mapping step performed by
    :mod:`visxai.utils.mapping`.

    Parameters
    ----------
    smiles : str
        Canonical SMILES string for the molecule.
    mol : rdkit.Chem.Mol
        RDKit ``Mol`` object, sanitized and ready for atom/bond queries.
    fingerprint_array : numpy.ndarray
        Dense 1-D binary (or count) fingerprint vector of shape ``(n_bits,)``.
    bit_info : dict[int, list[tuple[int, ...]]]
        Mapping from an *active* bit index to the list of atom-index tuples
        that triggered that bit.  Each tuple contains the atom indices
        involved in the corresponding circular environment.

        For a Morgan fingerprint generated with
        ``rdMolDescriptors.GetMorganFingerprintAsBitVect(..., bitInfo=info)``,
        ``info`` has type ``dict[int, tuple[tuple[int, int], ...]]`` where
        each inner tuple is ``(atom_idx, radius)``.  Before storing here,
        callers should normalise the value to ``list[tuple[int, ...]]`` so
        that MACCS and other fingerprint schemes can reuse the same field
        with their own atom-index conventions.
    pyg_data : torch_geometric.data.Data, optional
        PyTorch Geometric ``Data`` object populated by
        :mod:`visxai.features.graphs`.  ``None`` when the graph
        representation is not needed.
    token_ids : list[int], optional
        Token IDs produced by whichever tokenizer generated this
        representation (VisXAI's built-in reference tokenizer or a
        user-supplied one, e.g. a real HuggingFace tokenizer), aligned with
        ``token_to_atom_map``.  ``None`` when the sequence representation
        is not needed.
    token_to_atom_map : dict[int, list[int]], optional
        Mapping from token position index to the list of RDKit atom indices
        that the token covers.  Populated by
        :mod:`visxai.features.sequences` and consumed by
        :mod:`visxai.explainers.attention` and
        :mod:`visxai.utils.mapping`.  ``None`` when the sequence
        representation is not needed.
    token_to_bond_map : dict[int, list[int]], optional
        Mapping from token position index to the list of RDKit *bond*
        indices that the token covers, parallel to ``token_to_atom_map``.
        Most bonds have no explicit character in a SMILES string (a plain
        single or aromatic bond is implicit), so this map is populated
        only for the sparse subset of bonds with an explicit symbol (e.g.
        ``=``, ``#``, or a ring-closure digit) — see
        :func:`visxai.features.sequences.compute_bond_char_spans`.  ``None``
        when the sequence representation is not needed.
    bit_bond_info : dict[int, list[tuple[int, ...]]], optional
        Mapping from an *active* bit index to the list of bond-index tuples
        that triggered that bit, parallel to ``bit_info`` (which stores the
        corresponding atom-index tuples).  Populated by
        :mod:`visxai.features.fingerprints` for both Morgan and MACCS
        fingerprints and consumed by
        :class:`visxai.explainers.tree_shap.TreeSHAPExplainer` to produce
        bond-level scores.  A bit whose triggering environment(s) contain
        no bonds at all (e.g. a radius-0 Morgan environment, which is a
        single atom) is simply absent from the per-bit tuple list rather
        than represented with an empty tuple.  ``None`` when the
        representation does not carry bond-level bit info (e.g. any
        representation other than a fingerprint one).
    attention_mask : list[int], optional
        Per-token mask (``1`` = real token, ``0`` = padding) aligned with
        ``token_ids``, required by batched transformer-style model
        wrappers (e.g. :mod:`visxai.models.pytorch_wrapper`).  ``None``
        when the sequence representation is not needed.

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> import numpy as np
    >>> mol = MolFromSmiles("CCO")
    >>> rep = MoleculeRepresentation(
    ...     smiles="CCO",
    ...     mol=mol,
    ...     fingerprint_array=np.zeros(2048, dtype=np.uint8),
    ...     bit_info={},
    ... )
    """

    smiles: str
    mol: Mol
    fingerprint_array: np.ndarray
    bit_info: dict[int, list[tuple[int, ...]]]

    # Optional representations — populated only when the corresponding
    # feature extractor has been called.
    pyg_data: Optional[object] = field(default=None)  # torch_geometric.data.Data
    token_ids: Optional[list[int]] = field(default=None)
    token_to_atom_map: Optional[dict[int, list[int]]] = field(default=None)
    token_to_bond_map: Optional[dict[int, list[int]]] = field(default=None)
    attention_mask: Optional[list[int]] = field(default=None)
    bit_bond_info: Optional[dict[int, list[tuple[int, ...]]]] = field(default=None)


class ScoreContribution(TypedDict):
    """One source's itemised contribution to a single atom's or bond's score.

    An atom/bond score in :class:`Explanation` is a *sum of shares* drawn from
    several sources — fingerprint bits in the tree path, token positions in the
    sequence path.  The accumulation loop that produces the sum discards how it
    was reached, so a score of ``-0.42`` is indistinguishable from one bit
    landing alone, one bit worth ``-2.52`` divided six ways, or four weak bits
    summing.  This TypedDict records the itemisation before it collapses.

    A ``TypedDict`` rather than a dataclass on purpose: it is strictly typed
    *and* JSON-native at runtime, so it crosses into the
    :class:`~visxai.visualizers.hover_widget.MoleculeHoverWidget` payload with
    no conversion layer.

    Parameters
    ----------
    source_kind : {"bit", "token"}
        What kind of source produced this contribution.  ``"bit"`` for a
        fingerprint bit (tree path), ``"token"`` for a token position
        (sequence path).
    source_index : int
        Index of the source — the fingerprint bit index, or the token
        position within the tokenised sequence.
    source_score : float
        The source's *own* score before any sharing: the bit's raw SHAP
        value, or the token's raw attribution.
    shared_among : int
        How many elements ``source_score`` was divided across to produce
        ``contribution``.  **This is the field that distinguishes splitting
        from duplication**: ``1`` means the source handed over its full score
        undivided (duplication), while ``> 1`` means the score was split that
        many ways.  Recording the divisor rather than a mode name keeps the
        distinction meaningful across all three of VisXAI's sharing
        behaviours, which do not share a common vocabulary — see
        :func:`~visxai.utils.mapping.distribute_bit_score` (always splits) and
        :func:`~visxai.utils.mapping.aggregate_token_scores_to_atoms` (always
        duplicates).
    contribution : float
        What actually landed on this element.  Equal to
        ``source_score / shared_among`` for the tree path; equal to
        ``source_score`` when ``shared_among`` is ``1``.

    Examples
    --------
    >>> c: ScoreContribution = {
    ...     "source_kind": "bit",
    ...     "source_index": 314,
    ...     "source_score": -2.52,
    ...     "shared_among": 6,
    ...     "contribution": -0.42,
    ... }
    >>> c["shared_among"] > 1  # split, not duplicated
    True
    """

    source_kind: Literal["bit", "token"]
    source_index: int
    source_score: float
    shared_among: int
    contribution: float


@dataclass
class Explanation:
    """XAI output for a single molecule.

    Stores per-atom and per-bond importance scores produced by any
    :class:`~visxai.core.base_explainer.BaseExplainer` implementation.
    All visualizers in :mod:`visxai.visualizers` consume this dataclass.

    Parameters
    ----------
    atom_scores : dict[int, float]
        Mapping from RDKit atom index to a scalar importance score.  All
        atoms in the molecule should be present as keys; atoms with no
        contribution may be stored with a score of ``0.0``.
    bond_scores : dict[int, float], optional
        Mapping from RDKit bond index to a scalar importance score.
        Defaults to an empty dict when the explainer does not produce
        bond-level attributions.
    metadata : dict, optional
        Free-form dictionary for auxiliary information such as the model's
        predicted value, class probabilities, or confidence scores.
        Defaults to an empty dict.  Explainers that share a score across
        several elements also record an ``"attribution"`` descriptor here
        (e.g. ``"tree/duplicate"``, ``"sequence/duplicate"``,
        ``"graph/direct"``) naming the convention that produced the scores.
    atom_provenance : dict[int, list[ScoreContribution]], optional
        Per-atom itemisation of where each score came from.  Defaults to an
        empty dict, which means "not recorded" rather than "no contributions"
        — the same "no entry = no information available" convention
        ``bond_scores`` already uses for unmeasurable bonds.  Explainers whose
        mapping is 1:1 (the graph path, where node *i* is atom *i*) leave this
        empty on purpose: a single-entry list per atom carries no information
        the score itself does not already give.
    bond_provenance : dict[int, list[ScoreContribution]], optional
        Per-bond equivalent of ``atom_provenance``.  Defaults to an empty dict.

    Examples
    --------
    >>> exp = Explanation(
    ...     atom_scores={0: 0.8, 1: 0.3, 2: -0.1},
    ...     bond_scores={0: 0.5},
    ...     metadata={"predicted_value": 3.14, "model": "RF"},
    ... )
    >>> exp.atom_scores[0]
    0.8
    """

    atom_scores: dict[int, float]
    bond_scores: dict[int, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    atom_provenance: dict[int, list[ScoreContribution]] = field(default_factory=dict)
    bond_provenance: dict[int, list[ScoreContribution]] = field(default_factory=dict)
