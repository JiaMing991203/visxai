"""TreeSHAP explainer for scikit-learn / XGBoost models on fingerprint features.

Uses ``shap.TreeExplainer`` to compute exact SHAP values for tree-based models,
then maps the resulting bit-level scores back to RDKit atom indices — and,
when available, bond indices — via
:func:`~visxai.utils.mapping.distribute_bit_score` and the ``bit_info`` /
``bit_bond_info`` stored in
:class:`~visxai.core.data_types.MoleculeRepresentation`.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal, Optional

import numpy as np
import shap

from visxai.core.base_explainer import BaseExplainer
from visxai.core.base_model import BaseModelWrapper
from visxai.core.data_types import Explanation, MoleculeRepresentation
from visxai.models.sklearn_wrapper import SklearnModelWrapper
from visxai.utils.mapping import distribute_bit_score


def _unique_index_count(tuples_list: Optional[list[tuple[int, ...]]]) -> int:
    """Count the unique indices across a (possibly ``None``/empty) list of tuples."""
    if not tuples_list:
        return 0
    unique: set[int] = set()
    for t in tuples_list:
        unique.update(t)
    return len(unique)


class TreeSHAPExplainer(BaseExplainer):
    """Compute per-atom (and optionally per-bond) XAI scores using TreeSHAP.

    For each active fingerprint bit the corresponding SHAP value is
    distributed among the atom indices recorded in
    :attr:`~visxai.core.data_types.MoleculeRepresentation.bit_info` via
    :func:`~visxai.utils.mapping.distribute_bit_score`. Contributions from
    different bits that touch the same atom are summed, giving a final
    per-atom importance score.

    When ``mol_rep.bit_bond_info`` is populated (both
    :func:`~visxai.features.fingerprints.generate_morgan_representation` and
    :func:`~visxai.features.fingerprints.generate_maccs_representation` do
    this), the same SHAP value is also distributed among the bond indices
    recorded there, giving a per-bond importance score. Since a fingerprint
    bit represents a whole substructure — atoms *and* the bonds connecting
    them — this makes the resulting explanation more directly comparable to
    the graph path's atom+bond attributions
    (:class:`~visxai.explainers.gradient_based.IntegratedGradientsExplainer`).
    The exact relationship between a bit's SHAP value and its bond scores is
    controlled by ``bond_score_mode`` (see below).

    Inactive bits (fingerprint value == 0) are skipped: their SHAP values
    represent the contribution of *absence*, which has no meaningful
    atom/bond attribution.

    Parameters
    ----------
    model_output : str, optional
        Passed directly to ``shap.TreeExplainer(model, model_output=...)``.
        Use ``"raw"`` (default) for raw model output, or ``"probability"``
        for probability outputs in classifiers that support it.
    bond_score_mode : {"duplicate", "split"}, optional
        How a bit's SHAP score is shared between its atoms and its bonds.
        Defaults to ``"duplicate"``.

        - ``"duplicate"`` — the bit's full SHAP score is distributed twice,
          independently: once across its atoms (identical to what
          ``atom_scores`` has always contained) and once across its bonds.
          A bond within a small environment often ends up with a score
          equal to its endpoint atoms'. ``sum(atom_scores) +
          sum(bond_scores)`` is therefore *not* a completeness-style total —
          it double-counts each bit's contribution from two angles.
        - ``"split"`` — the bit's SHAP score is divided across the combined
          set of its unique atoms and unique bonds, with every element
          (atom or bond) receiving an equal per-element share
          (``bit_score / (n_unique_atoms + n_unique_bonds)``), so that
          ``sum(atom_scores) + sum(bond_scores)`` over a bit's contribution
          equals the bit's own SHAP score — mirroring the graph path's
          Integrated-Gradients-style completeness axiom. This changes
          ``atom_scores`` values relative to ``"duplicate"`` mode (and
          relative to what this explainer produced before bond-level
          attribution existed).

        When ``mol_rep.bit_bond_info`` is ``None`` (no bond-level bit info
        available), both modes behave identically: ``bond_scores`` is an
        empty dict and every bit's full SHAP score goes to its atoms, exactly
        as before bond-level attribution was added.

    Attributes
    ----------
    _model_output : str
        Stored ``model_output`` setting used when building the
        ``shap.TreeExplainer``.
    _bond_score_mode : {"duplicate", "split"}
        Stored ``bond_score_mode`` setting.

    Examples
    --------
    >>> explainer = TreeSHAPExplainer()
    >>> explanation = explainer.explain(sklearn_wrapper, mol_rep)
    >>> explanation.atom_scores  # {atom_idx: cumulative_shap_score, ...}
    """

    def __init__(
        self,
        model_output: str = "raw",
        bond_score_mode: Literal["duplicate", "split"] = "duplicate",
    ) -> None:
        if bond_score_mode not in ("duplicate", "split"):
            raise ValueError(
                f"bond_score_mode must be 'duplicate' or 'split', got {bond_score_mode!r}"
            )
        self._model_output: str = model_output
        self._bond_score_mode: Literal["duplicate", "split"] = bond_score_mode

    def explain(
        self,
        model: BaseModelWrapper,
        mol_rep: MoleculeRepresentation,
    ) -> Explanation:
        """Compute atom-level SHAP attributions for a single molecule.

        Parameters
        ----------
        model : BaseModelWrapper
            Must be a :class:`~visxai.models.sklearn_wrapper.SklearnModelWrapper`
            wrapping a tree-based estimator compatible with
            ``shap.TreeExplainer``.
        mol_rep : MoleculeRepresentation
            Featurised molecule.  Both ``fingerprint_array`` and ``bit_info``
            must be populated (use one of the generators in
            :mod:`visxai.features.fingerprints`).

        Returns
        -------
        Explanation
            Dataclass with:

            - ``atom_scores`` — every RDKit atom index present in the
              molecule mapped to its cumulative SHAP contribution.  Atoms
              not covered by any active bit receive a score of ``0.0``.
            - ``bond_scores`` — every RDKit bond index present in the
              molecule mapped to its cumulative SHAP contribution, when
              ``mol_rep.bit_bond_info`` is populated (see
              ``bond_score_mode``).  Bonds not covered by any active bit
              receive a score of ``0.0``.  Empty dict when
              ``mol_rep.bit_bond_info`` is ``None``.
            - ``metadata`` — contains ``"base_value"`` (the SHAP expected
              value) and ``"predicted_value"`` (the model's raw prediction).

        Raises
        ------
        TypeError
            If ``model`` is not a
            :class:`~visxai.models.sklearn_wrapper.SklearnModelWrapper`.
        ValueError
            If ``mol_rep.fingerprint_array`` is not 1-D.
        """
        if not isinstance(model, SklearnModelWrapper):
            raise TypeError(
                f"TreeSHAPExplainer requires a SklearnModelWrapper, "
                f"got {type(model).__name__}"
            )

        fp: np.ndarray = mol_rep.fingerprint_array
        if fp.ndim != 1:
            raise ValueError(
                f"fingerprint_array must be 1-D, got shape {fp.shape}"
            )

        # Build TreeExplainer once per call (stateless, no caching needed).
        tree_explainer = shap.TreeExplainer(
            model.model,
            model_output=self._model_output,
        )

        # shap_values shape for a single sample, depending on shap version and
        # estimator type:
        #   binary classifier → list of two (1, n_bits) arrays (older shap), or
        #                        a single (1, n_bits, n_classes) ndarray (newer shap)
        #   regressor         → (1, n_bits)
        X: np.ndarray = fp.reshape(1, -1)
        raw_shap = tree_explainer.shap_values(X)

        # For binary classifiers we use the positive-class (index 1) values by
        # convention. Older shap versions return a list [class0_array,
        # class1_array]; newer versions instead return a single ndarray with an
        # extra trailing class axis, (1, n_bits, n_classes) — ravelling that
        # directly (without first slicing out the class axis) interleaves
        # every bit's class-0/class-1 values and silently corrupts the
        # per-bit lookup below, so the two ndarray cases must be told apart.
        if isinstance(raw_shap, list):
            shap_array: np.ndarray = np.asarray(raw_shap[1]).ravel()
        else:
            raw_shap_arr = np.asarray(raw_shap)
            if raw_shap_arr.ndim == 3:
                shap_array = raw_shap_arr[..., 1].ravel()
            else:
                shap_array = raw_shap_arr.ravel()

        # base_value has the same list structure for classifiers.
        base_value_raw = tree_explainer.expected_value
        if isinstance(base_value_raw, (list, np.ndarray)):
            base_value: float = float(np.asarray(base_value_raw).ravel()[-1])
        else:
            base_value = float(base_value_raw)

        # Accumulate atom/bond scores: sum contributions from every active bit.
        n_atoms: int = mol_rep.mol.GetNumAtoms()
        atom_scores: dict[int, float] = defaultdict(float)
        bond_scores: dict[int, float] = defaultdict(float)
        has_bond_info: bool = mol_rep.bit_bond_info is not None

        active_bits = np.flatnonzero(fp)
        for bit_idx in active_bits:
            bit_score: float = float(shap_array[bit_idx])
            bit_atoms_list: Optional[list[tuple[int, ...]]] = mol_rep.bit_info.get(bit_idx)
            bit_bonds_list: Optional[list[tuple[int, ...]]] = (
                mol_rep.bit_bond_info.get(bit_idx) if has_bond_info else None
            )

            if self._bond_score_mode == "split" and has_bond_info:
                n_atoms_unique = _unique_index_count(bit_atoms_list)
                n_bonds_unique = _unique_index_count(bit_bonds_list)
                n_total = n_atoms_unique + n_bonds_unique
                if n_total == 0:
                    # Active bit with no atom or bond mapping (e.g., a MACCS
                    # '?' key). The score cannot be attributed — skip.
                    continue

                if n_atoms_unique:
                    per_atom = distribute_bit_score(
                        bit_atoms_list, bit_score * n_atoms_unique / n_total
                    )
                    for atom_idx, contribution in per_atom.items():
                        atom_scores[atom_idx] += contribution
                if n_bonds_unique:
                    per_bond = distribute_bit_score(
                        bit_bonds_list, bit_score * n_bonds_unique / n_total
                    )
                    for bond_idx, contribution in per_bond.items():
                        bond_scores[bond_idx] += contribution
            else:
                # "duplicate" mode, or "split" mode with no bond info
                # available: the bit's full score goes to its atoms, and
                # (independently, if present) its full score also goes to
                # its bonds.
                if bit_atoms_list:
                    per_atom = distribute_bit_score(bit_atoms_list, bit_score)
                    for atom_idx, contribution in per_atom.items():
                        atom_scores[atom_idx] += contribution
                elif not bit_bonds_list:
                    # Active bit with no atom or bond mapping (e.g., a MACCS
                    # '?' key). The score cannot be attributed — skip.
                    continue

                if bit_bonds_list:
                    per_bond = distribute_bit_score(bit_bonds_list, bit_score)
                    for bond_idx, contribution in per_bond.items():
                        bond_scores[bond_idx] += contribution

        # Ensure every atom in the molecule has an entry (default 0.0).
        for i in range(n_atoms):
            if i not in atom_scores:
                atom_scores[i] = 0.0

        # Ensure every bond has an entry (default 0.0), but only when
        # bond-level bit info was actually available for this representation.
        if has_bond_info:
            n_bonds: int = mol_rep.mol.GetNumBonds()
            for i in range(n_bonds):
                if i not in bond_scores:
                    bond_scores[i] = 0.0
            bond_scores_out: dict[int, float] = dict(bond_scores)
        else:
            bond_scores_out = {}

        return Explanation(
            atom_scores=dict(atom_scores),
            bond_scores=bond_scores_out,
            metadata={
                "base_value": base_value,
                "predicted_value": float(model.predict(mol_rep)[0]),
            },
        )
