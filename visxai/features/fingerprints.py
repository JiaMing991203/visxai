"""Fingerprint feature extraction with atom-index metadata capture.

This module provides two public functions that convert a SMILES string into a
:class:`~visxai.core.data_types.MoleculeRepresentation` populated with both
a dense fingerprint array and a ``bit_info`` dictionary.  The ``bit_info``
mapping is the critical artefact consumed by :mod:`visxai.utils.mapping` to
translate bit-level SHAP values back to individual atoms.

Supported fingerprint schemes
------------------------------
- **Morgan (ECFP-style)** — circular fingerprints computed with
  :func:`generate_morgan_representation`.
- **MACCS keys** — 166 predefined structural keys computed with
  :func:`generate_maccs_representation`.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdMolDescriptors

from visxai.core.data_types import MoleculeRepresentation
from visxai.utils.chem_utils import parse_smiles


# ---------------------------------------------------------------------------
# Morgan fingerprints
# ---------------------------------------------------------------------------


def generate_morgan_representation(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> MoleculeRepresentation:
    """Compute a Morgan (ECFP-style) fingerprint and capture atom environments.

    For each active bit, RDKit provides the ``(center_atom, radius)`` pairs
    responsible for that bit via ``bitInfo``.  This function expands each pair
    into the full set of atom indices in that circular environment using
    :func:`rdkit.Chem.FindAtomEnvironmentOfRadiusN`, then stores those tuples
    in the generalised ``bit_info`` field of
    :class:`~visxai.core.data_types.MoleculeRepresentation`.
    ``FindAtomEnvironmentOfRadiusN`` itself returns the *bond* indices within
    that same environment; this function also stores those, in the parallel
    ``bit_bond_info`` field, rather than discarding them.

    Parameters
    ----------
    smiles : str
        Input SMILES string.  Must be parseable by RDKit.
    radius : int, optional
        Maximum circular radius for the Morgan algorithm (analogous to ECFP
        ``diameter / 2``).  Defaults to ``2`` (ECFP4).
    n_bits : int, optional
        Length of the folded bit-vector.  Defaults to ``2048``.

    Returns
    -------
    MoleculeRepresentation
        Fully populated container with:

        - ``smiles`` — the original input string.
        - ``mol`` — the sanitized RDKit ``Mol`` object.
        - ``fingerprint_array`` — binary ``np.ndarray`` of shape ``(n_bits,)``.
        - ``bit_info`` — ``dict[int, list[tuple[int, ...]]]`` mapping each
          active bit to the sorted atom-index tuples of every circular
          environment that triggered it.
        - ``bit_bond_info`` — ``dict[int, list[tuple[int, ...]]]`` mapping
          each active bit to the sorted bond-index tuples of every circular
          environment that triggered it.  A radius-0 environment (a single
          atom) contributes no tuple, since it contains no bonds; a bit
          whose environments are all radius-0 maps to ``[]``.

    Raises
    ------
    ValueError
        If RDKit cannot parse ``smiles``.

    Examples
    --------
    >>> rep = generate_morgan_representation("CCO", radius=2, n_bits=2048)
    >>> rep.fingerprint_array.shape
    (2048,)
    >>> all(isinstance(k, int) for k in rep.bit_info)
    True
    """
    mol: Chem.Mol = parse_smiles(smiles)

    # Collect RDKit's raw bitInfo: bit -> ((center_atom, radius), ...)
    rdkit_bit_info: dict[int, tuple[tuple[int, int], ...]] = {}
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=n_bits,
        bitInfo=rdkit_bit_info,
    )

    fp_array: np.ndarray = np.zeros(n_bits, dtype=np.uint8)
    for bit in fp.GetOnBits():
        fp_array[bit] = 1

    # Expand each (center_atom, rad) pair into a sorted tuple of all atom
    # indices within that circular bond environment, plus the sorted tuple
    # of bond indices within that same environment.
    bit_info: dict[int, list[tuple[int, ...]]] = {}
    bond_info: dict[int, list[tuple[int, ...]]] = {}
    for bit_idx, env_list in rdkit_bit_info.items():
        atom_tuples: list[tuple[int, ...]] = []
        bond_tuples: list[tuple[int, ...]] = []
        for center_atom, rad in env_list:
            if rad == 0:
                # Radius-0 environment is just the center atom itself;
                # FindAtomEnvironmentOfRadiusN returns an empty bond set, so
                # this environment contributes no bond tuple.
                atom_tuples.append((center_atom,))
            else:
                bond_env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, center_atom)
                atom_set: set[int] = {center_atom}
                for bond_idx in bond_env:
                    bond = mol.GetBondWithIdx(bond_idx)
                    atom_set.add(bond.GetBeginAtomIdx())
                    atom_set.add(bond.GetEndAtomIdx())
                atom_tuples.append(tuple(sorted(atom_set)))
                bond_tuples.append(tuple(sorted(bond_env)))
        bit_info[bit_idx] = atom_tuples
        bond_info[bit_idx] = bond_tuples

    return MoleculeRepresentation(
        smiles=smiles,
        mol=mol,
        fingerprint_array=fp_array,
        bit_info=bit_info,
        bit_bond_info=bond_info,
    )


# ---------------------------------------------------------------------------
# MACCS keys
# ---------------------------------------------------------------------------


def generate_maccs_representation(smiles: str) -> MoleculeRepresentation:
    """Compute 166 MACCS structural keys and capture matching atom indices.

    MACCS keys are binary substructure flags defined by a fixed set of SMARTS
    patterns.  For each active bit, this function runs
    :meth:`rdkit.Chem.Mol.GetSubstructMatches` with the corresponding SMARTS
    pattern and stores every match tuple (each tuple being a set of atom
    indices) in the ``bit_info`` field.

    Bits whose SMARTS pattern is ``'?'`` (pattern cannot be expressed as
    SMARTS, e.g. isotope queries) are recorded with an empty list so the key
    is still present in ``bit_info`` but carries no atom-index data.

    For each match, the corresponding bond indices are also derived: each
    bond in the SMARTS query pattern (``query.GetBonds()``) connects two
    *query*-atom indices, which are mapped through the match tuple to real
    atom indices and resolved to an actual bond via
    ``mol.GetBondBetweenAtoms``.  The results are stored in the parallel
    ``bit_bond_info`` field.  A match with no bonds in its query pattern
    (e.g. a single-atom SMARTS) contributes no tuple.

    Parameters
    ----------
    smiles : str
        Input SMILES string.  Must be parseable by RDKit.

    Returns
    -------
    MoleculeRepresentation
        Fully populated container with:

        - ``smiles`` — the original input string.
        - ``mol`` — the sanitized RDKit ``Mol`` object.
        - ``fingerprint_array`` — binary ``np.ndarray`` of shape ``(167,)``
          (MACCS key vectors are 167 bits; index 0 is unused by convention).
        - ``bit_info`` — ``dict[int, list[tuple[int, ...]]]`` mapping each
          active bit to the list of atom-index match tuples.
        - ``bit_bond_info`` — ``dict[int, list[tuple[int, ...]]]`` mapping
          each active bit to the list of bond-index tuples derived from the
          same matches.

    Raises
    ------
    ValueError
        If RDKit cannot parse ``smiles``.

    Examples
    --------
    >>> rep = generate_maccs_representation("CCO")
    >>> rep.fingerprint_array.shape
    (167,)
    >>> isinstance(rep.bit_info, dict)
    True
    """
    mol: Chem.Mol = parse_smiles(smiles)

    fp = MACCSkeys.GenMACCSKeys(mol)
    fp_array: np.ndarray = np.zeros(167, dtype=np.uint8)
    for bit in fp.GetOnBits():
        fp_array[bit] = 1

    bit_info: dict[int, list[tuple[int, ...]]] = {}
    bond_info: dict[int, list[tuple[int, ...]]] = {}
    for bit_idx in fp.GetOnBits():
        if bit_idx not in MACCSkeys.smartsPatts:
            bit_info[bit_idx] = []
            bond_info[bit_idx] = []
            continue

        smarts_str, _min_count = MACCSkeys.smartsPatts[bit_idx]

        # Some MACCS keys cannot be expressed as a SMARTS pattern.
        if smarts_str == "?":
            bit_info[bit_idx] = []
            bond_info[bit_idx] = []
            continue

        query: Chem.Mol | None = Chem.MolFromSmarts(smarts_str)
        if query is None:
            bit_info[bit_idx] = []
            bond_info[bit_idx] = []
            continue

        matches = mol.GetSubstructMatches(query)
        bit_info[bit_idx] = [tuple(match) for match in matches]

        query_bond_pairs = [
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            for bond in query.GetBonds()
        ]
        bond_tuples: list[tuple[int, ...]] = []
        for match in matches:
            bond_set: set[int] = set()
            for query_begin, query_end in query_bond_pairs:
                real_bond = mol.GetBondBetweenAtoms(
                    match[query_begin], match[query_end]
                )
                if real_bond is not None:
                    bond_set.add(real_bond.GetIdx())
            if bond_set:
                bond_tuples.append(tuple(sorted(bond_set)))
        bond_info[bit_idx] = bond_tuples

    return MoleculeRepresentation(
        smiles=smiles,
        mol=mol,
        fingerprint_array=fp_array,
        bit_info=bit_info,
        bit_bond_info=bond_info,
    )
