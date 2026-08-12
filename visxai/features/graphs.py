"""PyTorch Geometric graph construction for VisXAI graph models.

This module provides :func:`generate_graph_representation`, which converts a
SMILES string into a :class:`~visxai.core.data_types.MoleculeRepresentation`
populated with a :class:`torch_geometric.data.Data` object in ``pyg_data``.

Design
------
VisXAI wraps an already-trained model; it must not assume the node/edge
feature scheme that model was trained with matches any particular encoding
(see the architecture principle in the README). Unlike
:mod:`visxai.features.sequences`'s tokenizer (which has a reasonable
reference default), there is no dominant real-world convention for GNN
node/edge features analogous to Morgan/MACCS fingerprints for tree models —
so ``atom_featurizer`` is a **required parameter with no shipped default**,
forcing the caller to consciously supply one matching their target model's
actual training-time scheme.

The one thing this module *does* own unconditionally is index alignment:
node ``i`` in the returned graph is always atom ``i`` in ``mol_rep.mol``,
by construction (row ``i`` of the node feature matrix comes from
``mol.GetAtoms()[i]``, iterated in RDKit's own atom order). This is the
"universal bookkeeping" half of the architecture principle, and is what lets
:class:`~visxai.explainers.gradient_based.IntegratedGradientsExplainer`
assign a node-level attribution directly to the corresponding atom with no
redistribution step.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

from visxai.core.data_types import MoleculeRepresentation
from visxai.utils.chem_utils import parse_smiles

AtomFeaturizer = Callable[[Chem.Atom], np.ndarray]
"""A callable mapping a single RDKit ``Atom`` to a 1-D feature vector."""

BondFeaturizer = Callable[[Chem.Bond], np.ndarray]
"""A callable mapping a single RDKit ``Bond`` to a 1-D feature vector."""


def _build_node_features(mol: Chem.Mol, atom_featurizer: AtomFeaturizer) -> np.ndarray:
    """Stack ``atom_featurizer(atom)`` for every atom, in RDKit atom-index order.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Parsed molecule.
    atom_featurizer : AtomFeaturizer
        Callable producing one feature vector per atom.

    Returns
    -------
    numpy.ndarray
        2-D array of shape ``(n_atoms, n_atom_features)``, ``float32``. Row
        ``i`` is ``atom_featurizer(mol.GetAtoms()[i])`` — this ordering is
        the node-index-equals-atom-index guarantee the rest of the pipeline
        relies on.
    """
    rows = [atom_featurizer(atom) for atom in mol.GetAtoms()]
    return np.stack(rows, axis=0).astype(np.float32)


def _build_edge_index(mol: Chem.Mol) -> np.ndarray:
    """Build a bidirectional ``edge_index`` in PyTorch Geometric's convention.

    Each RDKit bond ``(a, b)`` contributes both directed edges ``(a, b)``
    and ``(b, a)`` — PyG's standard convention for an undirected graph, not
    a model-specific choice.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Parsed molecule.

    Returns
    -------
    numpy.ndarray
        2-D array of shape ``(2, 2 * n_bonds)``, ``int64``, in the
        ``edge_index`` layout PyG expects (row 0 = source atom indices, row
        1 = target atom indices).
    """
    sources: list[int] = []
    targets: list[int] = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        sources.extend((a, b))
        targets.extend((b, a))
    return np.array([sources, targets], dtype=np.int64)


def _build_edge_attr(mol: Chem.Mol, bond_featurizer: BondFeaturizer) -> np.ndarray:
    """Build a bidirectional ``edge_attr`` matching :func:`_build_edge_index`.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Parsed molecule.
    bond_featurizer : BondFeaturizer
        Callable producing one feature vector per bond. The same feature
        vector is used for both directed edges of a bond.

    Returns
    -------
    numpy.ndarray
        2-D array of shape ``(2 * n_bonds, n_bond_features)``, ``float32``,
        row-aligned with :func:`_build_edge_index`'s output.
    """
    rows: list[np.ndarray] = []
    for bond in mol.GetBonds():
        feat = np.asarray(bond_featurizer(bond))
        rows.append(feat)
        rows.append(feat)
    return np.stack(rows, axis=0).astype(np.float32)


def generate_graph_representation(
    smiles: str,
    atom_featurizer: AtomFeaturizer,
    bond_featurizer: Optional[BondFeaturizer] = None,
) -> MoleculeRepresentation:
    """Build a PyTorch Geometric graph representation for a SMILES string.

    Parameters
    ----------
    smiles : str
        Input SMILES string. Must be parseable by RDKit.
    atom_featurizer : AtomFeaturizer
        Callable mapping an RDKit ``Atom`` to a 1-D feature vector. Required
        with no default — must match the target model's actual
        training-time node-feature scheme (see the architecture principle
        in the README); VisXAI cannot verify this and does not assume
        one specific scheme is universally correct.
    bond_featurizer : BondFeaturizer, optional
        Callable mapping an RDKit ``Bond`` to a 1-D feature vector. If
        omitted, the returned graph has no ``edge_attr`` (topology-only
        edges via ``edge_index``).

    Returns
    -------
    MoleculeRepresentation
        Fully populated container with:

        - ``smiles`` / ``mol`` — the original string and sanitized RDKit
          ``Mol``.
        - ``fingerprint_array`` / ``bit_info`` — irrelevant to the graph
          path; set to an empty array and an empty dict respectively.
        - ``pyg_data`` — a :class:`torch_geometric.data.Data` object with
          ``x`` (node features, node ``i`` = atom ``i``), ``edge_index``
          (bidirectional), and ``edge_attr`` if ``bond_featurizer`` was
          given. No self-loops and no ``pos`` (3D coordinates) are added —
          self-loops are left to whichever GNN layer the target model
          uses, and 3D coordinates are out of scope for this 2D-topology
          representation.

    Raises
    ------
    ValueError
        If RDKit cannot parse ``smiles``.

    Examples
    --------
    >>> def atom_featurizer(atom):
    ...     return [atom.GetAtomicNum(), atom.GetDegree()]
    >>> rep = generate_graph_representation("CCO", atom_featurizer)
    >>> rep.pyg_data.x.shape
    torch.Size([3, 2])
    >>> rep.pyg_data.edge_index.shape
    torch.Size([2, 4])
    """
    mol = parse_smiles(smiles)

    node_features = _build_node_features(mol, atom_featurizer)
    edge_index = _build_edge_index(mol)

    data_kwargs: dict[str, torch.Tensor] = {
        "x": torch.tensor(node_features, dtype=torch.float32),
        "edge_index": torch.tensor(edge_index, dtype=torch.long),
    }
    if bond_featurizer is not None:
        edge_attr = _build_edge_attr(mol, bond_featurizer)
        data_kwargs["edge_attr"] = torch.tensor(edge_attr, dtype=torch.float32)

    return MoleculeRepresentation(
        smiles=smiles,
        mol=mol,
        fingerprint_array=np.zeros(0, dtype=np.uint8),
        bit_info={},
        pyg_data=Data(**data_kwargs),
    )
