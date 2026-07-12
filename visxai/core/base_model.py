"""Abstract base class for all VisXAI model wrappers.

Every framework-specific wrapper (PyTorch, Scikit-Learn, HuggingFace) must
subclass :class:`BaseModelWrapper` and implement :meth:`predict`.  This
guarantees a uniform calling convention for all explainers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from visxai.core.data_types import MoleculeRepresentation


class BaseModelWrapper(ABC):
    """Uniform interface for model inference across all ML frameworks.

    Subclasses wrap a framework-native model object and expose a single
    :meth:`predict` method that accepts a :class:`MoleculeRepresentation`
    and returns a raw score array.  This decoupling lets every explainer
    remain framework-agnostic.

    Notes
    -----
    Subclasses may store the underlying model as an instance attribute (e.g.
    ``self.model``) and perform any required preprocessing inside
    :meth:`predict`.
    """

    @abstractmethod
    def predict(self, mol_rep: MoleculeRepresentation) -> np.ndarray:
        """Run inference and return raw model output scores.

        Parameters
        ----------
        mol_rep : MoleculeRepresentation
            Featurised molecule container produced by one of the extractors in
            :mod:`visxai.features`.

        Returns
        -------
        numpy.ndarray
            1-D array of shape ``(n_tasks,)`` containing raw model outputs
            (logits, probabilities, or regression values) for every prediction
            head.  Binary classifiers with a single output should return shape
            ``(1,)``.

        Raises
        ------
        NotImplementedError
            If the subclass has not implemented this method.
        """
        raise NotImplementedError
