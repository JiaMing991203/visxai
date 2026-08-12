"""XAI translation layer: maps feature-level scores back to molecular atoms.

This module is the core of the VisXAI "mapping problem".  Explainers operate
on feature representations (fingerprint bits, graph nodes, SMILES tokens), not
on atoms directly.  The utilities here bridge that gap so that every
:class:`~visxai.core.data_types.Explanation` can be expressed in terms of
RDKit atom indices regardless of the underlying model type.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from visxai.core.data_types import ScoreContribution


def distribute_bit_score(
    bit_atoms_list: list[tuple[int, ...]],
    score: float,
) -> dict[int, float]:
    """Distribute a single bit's XAI score equally across its contributing atoms.

    A fingerprint bit can be triggered by multiple atom environments (e.g., the
    same circular substructure centred on different atoms, or multiple
    substructure matches for a MACCS key).  ``bit_atoms_list`` contains one
    tuple of atom indices per triggering environment, as stored in
    :attr:`~visxai.core.data_types.MoleculeRepresentation.bit_info`.

    The function collects the *unique* atom indices across all tuples, then
    divides ``score`` equally among them.  Equal distribution is the
    maximally unbiased choice when the explainer provides no finer-grained
    information about which environment is more responsible for the bit's
    activation.

    Parameters
    ----------
    bit_atoms_list : list[tuple[int, ...]]
        List of atom-index tuples that triggered the bit.  Each tuple
        corresponds to one environment match.  May contain overlapping atom
        indices across tuples; duplicates are collapsed before distribution.
        Must not be empty.
    score : float
        The XAI attribution score (e.g., SHAP value) assigned to this bit by
        the explainer.  May be negative (indicating a feature that decreases
        the predicted value).

    Returns
    -------
    dict[int, float]
        Mapping from unique atom index to its share of ``score``.  Every atom
        index that appears in any tuple of ``bit_atoms_list`` receives an equal
        portion: ``score / n_unique_atoms``.

    Raises
    ------
    ValueError
        If ``bit_atoms_list`` is empty, since there are no atoms to distribute
        the score to.

    Examples
    --------
    Score distributed across three atoms from two overlapping environments:

    >>> distribute_bit_score([(0, 1), (1, 2)], score=0.6)
    {0: 0.2, 1: 0.2, 2: 0.2}

    Single-atom environment (radius-0 Morgan bit):

    >>> distribute_bit_score([(3,)], score=0.9)
    {3: 0.9}

    Negative SHAP value:

    >>> distribute_bit_score([(0, 1, 2)], score=-0.3)
    {0: -0.1, 1: -0.1, 2: -0.1}
    """
    if not bit_atoms_list:
        raise ValueError(
            "bit_atoms_list must not be empty: there are no atoms to distribute "
            "the score to.  Check that bit_info was correctly populated for this bit."
        )

    unique_atoms: set[int] = set()
    for atom_tuple in bit_atoms_list:
        unique_atoms.update(atom_tuple)

    share: float = score / len(unique_atoms)
    return {atom_idx: share for atom_idx in unique_atoms}


def align_tokens_to_atoms(
    target_spans: dict[int, tuple[int, int]],
    token_offsets: list[tuple[int, int]],
) -> dict[int, list[int]]:
    """Map each token position to the target indices whose span it overlaps.

    This is the tokenizer-agnostic half of the sequence "mapping problem":
    regardless of which tokenizer produced a token stream (VisXAI's own
    built-in regex tokenizer, or a real Hugging Face tokenizer's own
    ``return_offsets_mapping=True`` output), token-to-target alignment can
    always be derived from character-offset overlap between the tokenizer's
    reported spans and each target's known character span in the original
    SMILES string. This keeps the alignment itself model-agnostic even
    though the token *content* is not.

    ``target_spans`` is deliberately generic: it is used both for atoms
    (via :func:`visxai.features.sequences.compute_atom_char_spans`, which
    covers every atom index densely) and for bonds (via
    :func:`visxai.features.sequences.compute_bond_char_spans`, which only
    covers the sparse subset of bonds that have an explicit character in
    the SMILES string, e.g. ``=``, ``#``, or a ring-closure digit). A
    ``dict`` keyed by target index (rather than a plain list) is what makes
    the sparse bond case work without any sentinel-value tricks.

    Parameters
    ----------
    target_spans : dict[int, tuple[int, int]]
        Mapping from target index (an atom or bond index) to its
        ``(start_char, end_char)`` character span in the original SMILES
        string. Need not be dense or cover every index — indices absent
        from this dict simply cannot be matched to any token.
    token_offsets : list[tuple[int, int]]
        ``(start_char, end_char)`` character span of each token, in token
        position order, as reported by the tokenizer that produced the
        token stream. A zero-length span (e.g. ``(0, 0)``, the common
        convention for special tokens like ``[CLS]``/``[SEP]``/``[PAD]``)
        never overlaps any target and yields an empty list.

    Returns
    -------
    dict[int, list[int]]
        Mapping from token position to the list of target indices whose
        span overlaps that token's span. A token with no overlapping
        target (e.g. an implicit bond, or a special token) maps to an
        empty list — the same "unattributable → empty, consumer skips it"
        convention already used for MACCS ``'?'`` bits. A single token may
        overlap multiple targets and a single target may be covered by
        multiple tokens; both are valid and handled the same way
        ``bit_info`` already handles overlapping index sets.

    Examples
    --------
    One token per atom, no overlap (typical for an atom-wise tokenizer):

    >>> align_tokens_to_atoms({0: (0, 1), 1: (1, 2)}, [(0, 1), (1, 2)])
    {0: [0], 1: [1]}

    A special token with a zero-length span maps to no targets:

    >>> align_tokens_to_atoms({0: (0, 1)}, [(0, 0), (0, 1)])
    {0: [], 1: [0]}

    A single token spanning two atoms (e.g. a merged sub-token):

    >>> align_tokens_to_atoms({0: (0, 1), 1: (1, 2)}, [(0, 2)])
    {0: [0, 1]}

    Sparse bond spans: bond index 2 has no explicit character, so it never
    appears in the output even though token position 1 has no overlap:

    >>> align_tokens_to_atoms({0: (1, 2)}, [(0, 1), (1, 2)])
    {0: [], 1: [0]}
    """
    token_to_target_map: dict[int, list[int]] = {}
    for token_pos, (t_start, t_end) in enumerate(token_offsets):
        overlapping_targets: list[int] = []
        if t_start < t_end:
            for target_idx, (s_start, s_end) in target_spans.items():
                if t_start < s_end and s_start < t_end:
                    overlapping_targets.append(target_idx)
        token_to_target_map[token_pos] = sorted(overlapping_targets)
    return token_to_target_map


def aggregate_token_scores_to_atoms(
    token_scores: np.ndarray,
    token_to_atom_map: dict[int, list[int]],
    n_atoms: int,
) -> dict[int, float]:
    """Sum per-token XAI scores onto the atoms each token maps to.

    Promoted from :mod:`visxai.explainers.attention` (originally private,
    ``_aggregate_token_scores_to_atoms``) once a second sequence-path
    explainer module (:mod:`visxai.explainers.gradient_based_sequence`)
    needed the identical aggregation logic — the logic itself is agnostic
    to *how* a per-token score was produced (raw attention weight, summed
    Integrated Gradients attribution, or Grad-CAM channel-weighted
    activation), so it belongs here alongside :func:`align_tokens_to_atoms`
    rather than duplicated or imported cross-explainer-module.

    Dense: every atom index in ``range(n_atoms)`` is present in the
    result, defaulting to ``0.0`` if no token maps to it — mirrors
    :func:`distribute_bit_score`'s "every measurable index gets an entry"
    convention.

    Parameters
    ----------
    token_scores : numpy.ndarray
        1-D array of shape ``(seq_len,)`` with one score per token
        position.
    token_to_atom_map : dict[int, list[int]]
        Token position to atom-index list, as produced by
        :func:`align_tokens_to_atoms` (dense — every atom index appears in
        some token's mapped list).
    n_atoms : int
        Total number of atoms in the molecule, so every atom index gets an
        entry even when uncovered by any token.

    Returns
    -------
    dict[int, float]
        Every atom index in ``range(n_atoms)`` mapped to its cumulative
        token-score contribution (``0.0`` if uncovered).
    """
    atom_scores: dict[int, float] = defaultdict(float)
    for token_pos, atoms in token_to_atom_map.items():
        if not atoms:
            continue
        score = float(token_scores[token_pos])
        for atom_idx in atoms:
            atom_scores[atom_idx] += score

    for i in range(n_atoms):
        if i not in atom_scores:
            atom_scores[i] = 0.0
    return dict(atom_scores)


def aggregate_token_scores_to_bonds(
    token_scores: np.ndarray,
    token_to_bond_map: dict[int, list[int]],
) -> dict[int, float]:
    """Sum per-token XAI scores onto the bonds each token maps to.

    Promoted from :mod:`visxai.explainers.attention` (originally private,
    ``_aggregate_token_scores_to_bonds``) — see
    :func:`aggregate_token_scores_to_atoms`'s docstring for why.

    Unlike :func:`aggregate_token_scores_to_atoms`, bonds with no
    contributing token are **omitted** rather than filled with ``0.0``.
    Most bonds have no explicit character in the SMILES string at all (see
    :func:`visxai.features.sequences.compute_bond_char_spans`), so "no
    entry" means "no information available" — distinct from a token-backed
    measurement that happened to come out at zero. This keeps
    :class:`~visxai.visualizers.rdkit_2d.RDKitSVGVisualizer` from painting
    the (large) majority of unmeasured bonds as if they were confirmed to
    have zero contribution.

    Parameters
    ----------
    token_scores : numpy.ndarray
        1-D array of shape ``(seq_len,)`` with one score per token
        position.
    token_to_bond_map : dict[int, list[int]]
        Token position to bond-index list, as produced by
        :func:`align_tokens_to_atoms` (sparse — only bond indices with an
        explicit SMILES character appear as keys anywhere in this
        structure at all).

    Returns
    -------
    dict[int, float]
        Bond index mapped to its cumulative token-score contribution, for
        only the bonds actually covered by some token.
    """
    bond_scores: dict[int, float] = defaultdict(float)
    for token_pos, bonds in token_to_bond_map.items():
        if not bonds:
            continue
        score = float(token_scores[token_pos])
        for bond_idx in bonds:
            bond_scores[bond_idx] += score
    return dict(bond_scores)


def build_token_provenance(
    token_scores: np.ndarray,
    token_to_target_map: dict[int, list[int]],
) -> dict[int, list[ScoreContribution]]:
    """Itemise which tokens contributed to each atom or bond, and how.

    The companion to :func:`aggregate_token_scores_to_atoms` /
    :func:`aggregate_token_scores_to_bonds`, which sum their contributions
    away. Kept as a separate function rather than an extra return value so
    those two keep their established signatures and stay usable unchanged by
    callers that do not want provenance.

    Like :func:`align_tokens_to_atoms` and :func:`distribute_bit_score`, this
    is deliberately index-type-agnostic: pass ``token_to_atom_map`` or
    ``token_to_bond_map`` and the logic is identical, because the sequence
    path treats atoms and bonds the same way.

    **Every contribution has ``shared_among=1``**, which is the point of
    recording it. The sequence path *duplicates*: a token overlapping three
    atoms gives its full score to each, undivided, so the total attributed
    mass exceeds the token scores' own sum. This is the opposite of the
    fingerprint path, where :func:`distribute_bit_score` always divides — and
    the two behaviours are otherwise indistinguishable from the final scores
    alone.

    Parameters
    ----------
    token_scores : numpy.ndarray
        1-D array of shape ``(seq_len,)`` with one score per token position.
    token_to_target_map : dict[int, list[int]]
        Token position to target-index list, as produced by
        :func:`align_tokens_to_atoms`. Tokens mapping to an empty list (special
        tokens such as ``[CLS]``/``[SEP]``, or implicit bonds) contribute
        nothing and appear nowhere in the result.

    Returns
    -------
    dict[int, list[ScoreContribution]]
        Target index mapped to the list of contributions it received. Only
        targets actually covered by some token appear as keys, matching
        :func:`aggregate_token_scores_to_bonds`'s "no entry means no
        information available" convention rather than zero-filling.

    Examples
    --------
    >>> import numpy as np
    >>> prov = build_token_provenance(np.array([0.5, 0.2]), {0: [0, 1], 1: [1]})
    >>> [c["contribution"] for c in prov[1]]  # atom 1 got both tokens in full
    [0.5, 0.2]
    >>> prov[0][0]["shared_among"]  # never divided
    1
    """
    provenance: dict[int, list[ScoreContribution]] = {}
    for token_pos, targets in token_to_target_map.items():
        if not targets:
            continue
        score = float(token_scores[token_pos])
        for target_idx in targets:
            provenance.setdefault(target_idx, []).append(
                ScoreContribution(
                    source_kind="token",
                    source_index=int(token_pos),
                    source_score=score,
                    shared_among=1,
                    contribution=score,
                )
            )
    return provenance
