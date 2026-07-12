"""Scikit-learn / XGBoost model wrapper for VisXAI.

Wraps any scikit-learn-compatible estimator (RandomForest, GradientBoosting,
XGBoost, etc.) and exposes it through the uniform
:class:`~visxai.core.base_model.BaseModelWrapper` interface.  The wrapper
extracts the dense fingerprint vector from a
:class:`~visxai.core.data_types.MoleculeRepresentation` and passes it to the
underlying model, returning a 1-D ``np.ndarray`` of raw predictions.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from visxai.core.base_model import BaseModelWrapper
from visxai.core.data_types import MoleculeRepresentation


@runtime_checkable
class _SklearnEstimator(Protocol):
    """Structural protocol matching any scikit-learn-compatible estimator."""

    def predict(self, X: np.ndarray) -> np.ndarray: ...  # noqa: N803


class SklearnModelWrapper(BaseModelWrapper):
    """Wrap a scikit-learn-compatible estimator for use in VisXAI pipelines.

    The wrapper extracts ``mol_rep.fingerprint_array``, reshapes it to a
    ``(1, n_bits)`` matrix, calls the underlying estimator's ``predict``
    method, and returns the result as a flat 1-D ``np.ndarray``.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Any object that exposes a ``predict(X: np.ndarray) -> np.ndarray``
        method.  Typical examples: ``sklearn.ensemble.RandomForestClassifier``,
        ``sklearn.ensemble.GradientBoostingRegressor``,
        ``xgboost.XGBClassifier``.

    Attributes
    ----------
    model : sklearn-compatible estimator
        The wrapped estimator, stored for direct access (e.g., by
        :class:`~visxai.explainers.tree_shap.TreeSHAPExplainer`).

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> clf = RandomForestClassifier().fit(X_train, y_train)
    >>> wrapper = SklearnModelWrapper(clf)
    >>> predictions = wrapper.predict(mol_rep)
    """

    def __init__(self, model: _SklearnEstimator) -> None:
        self.model: _SklearnEstimator = model

    def predict(self, mol_rep: MoleculeRepresentation) -> np.ndarray:
        """Run inference on the molecule's fingerprint array.

        Parameters
        ----------
        mol_rep : MoleculeRepresentation
            Featurised molecule container.  The ``fingerprint_array`` field
            must be populated (shape ``(n_bits,)``).

        Returns
        -------
        numpy.ndarray
            1-D array of shape ``(n_outputs,)`` containing the model's raw
            prediction(s) for the input molecule.

        Raises
        ------
        ValueError
            If ``mol_rep.fingerprint_array`` is not a 1-D array.
        """
        fp = mol_rep.fingerprint_array
        if fp.ndim != 1:
            raise ValueError(
                f"fingerprint_array must be 1-D, got shape {fp.shape}"
            )
        X: np.ndarray = fp.reshape(1, -1)
        raw = self.model.predict(X)
        return np.asarray(raw).ravel()
