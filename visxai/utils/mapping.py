"""XAI translation layer: maps feature-level scores back to molecular atoms.

This module is the core of the VisXAI "mapping problem".  Explainers operate
on feature representations (fingerprint bits, graph nodes, SMILES tokens), not
on atoms directly.  The utilities here bridge that gap so that every
:class:`~visxai.core.data_types.Explanation` can be expressed in terms of
RDKit atom indices regardless of the underlying model type.
"""

from __future__ import annotations


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
