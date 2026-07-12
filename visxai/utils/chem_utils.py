"""RDKit I/O helpers shared across feature-extraction modules.

Currently provides a single shared SMILES-parsing helper that replaces the
``Chem.MolFromSmiles`` + raise-on-failure pattern duplicated across
:mod:`visxai.features.fingerprints` and :mod:`visxai.features.sequences`.
"""

from __future__ import annotations

from rdkit import Chem


def parse_smiles(smiles: str) -> Chem.Mol:
    """Parse a SMILES string into a sanitized RDKit ``Mol``.

    Parameters
    ----------
    smiles : str
        Input SMILES string.

    Returns
    -------
    rdkit.Chem.Mol
        The sanitized RDKit ``Mol`` object.

    Raises
    ------
    ValueError
        If RDKit cannot parse ``smiles``.

    Examples
    --------
    >>> mol = parse_smiles("CCO")
    >>> mol.GetNumAtoms()
    3
    """
    mol: Chem.Mol | None = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    return mol
