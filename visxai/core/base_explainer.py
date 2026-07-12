"""Abstract base class for all VisXAI XAI explainers.

Every algorithm implementation (Integrated Gradients, TreeSHAP, attention
extraction, etc.) must subclass :class:`BaseExplainer` and implement
:meth:`explain`.  The returned :class:`~visxai.core.data_types.Explanation`
dataclass is the single output contract consumed by all visualizers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from visxai.core.base_model import BaseModelWrapper
from visxai.core.data_types import Explanation, MoleculeRepresentation


class BaseExplainer(ABC):
    """Uniform interface for XAI attribution algorithms.

    Subclasses implement :meth:`explain`, which interrogates a
    :class:`~visxai.core.base_model.BaseModelWrapper` and a
    :class:`~visxai.core.data_types.MoleculeRepresentation` to produce an
    :class:`~visxai.core.data_types.Explanation` with per-atom (and
    optionally per-bond) importance scores.

    Notes
    -----
    Explainers that require extra configuration (e.g. number of integration
    steps, baseline choice) should accept those parameters in ``__init__``
    and store them as instance attributes.  The :meth:`explain` signature
    must remain stable so that the rest of the pipeline can call any
    explainer interchangeably.
    """

    @abstractmethod
    def explain(
        self,
        model: BaseModelWrapper,
        mol_rep: MoleculeRepresentation,
    ) -> Explanation:
        """Generate atom- and bond-level attribution scores for a molecule.

        Parameters
        ----------
        model : BaseModelWrapper
            A wrapped model exposing the :meth:`~BaseModelWrapper.predict`
            interface.  The explainer may call ``model.predict`` internally
            or access ``model.model`` for gradient hooks, depending on the
            algorithm.
        mol_rep : MoleculeRepresentation
            Featurised molecule container.  The explainer selects the
            appropriate feature view (``fingerprint_array``, ``pyg_data``,
            ``token_ids``, etc.) based on the algorithm it implements.

        Returns
        -------
        Explanation
            Dataclass carrying ``atom_scores``, ``bond_scores``, and
            ``metadata``.  All atom indices present in ``mol_rep.mol``
            must appear as keys in ``atom_scores``; atoms with no
            measurable contribution should be stored with a score of
            ``0.0``.

        Raises
        ------
        NotImplementedError
            If the subclass has not implemented this method.
        """
        raise NotImplementedError
