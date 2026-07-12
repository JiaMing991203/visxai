"""Abstract base class for all VisXAI visualizers.

Every rendering backend (RDKit 2D SVG, py3Dmol 3D WebGL, etc.) must
subclass :class:`BaseVisualizer` and implement :meth:`visualize`.  The
method contract guarantees that callers always receive a renderable SVG
string, keeping the rest of the pipeline decoupled from any particular
drawing library.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from visxai.core.data_types import Explanation, MoleculeRepresentation


class BaseVisualizer(ABC):
    """Uniform interface for mapping XAI scores to molecular visualizations.

    Subclasses translate the continuous scores in an
    :class:`~visxai.core.data_types.Explanation` into a color gradient
    applied to atoms and bonds, then serialize the result as an SVG string.

    Notes
    -----
    All 2-D implementations **must** use RDKit's native SVG canvas
    (``rdkit.Chem.Draw.rdMolDraw2D``) for rendering.  ``matplotlib`` is
    explicitly forbidden for molecule drawing.

    Subclasses may expose constructor parameters to configure canvas
    dimensions, color maps, or score normalization strategy.
    """

    @abstractmethod
    def visualize(
        self,
        mol_rep: MoleculeRepresentation,
        explanation: Explanation,
    ) -> str:
        """Render an explanation as an SVG string.

        Parameters
        ----------
        mol_rep : MoleculeRepresentation
            Featurised molecule container.  Implementations typically use
            ``mol_rep.mol`` (the RDKit ``Mol`` object) and ``mol_rep.smiles``
            for layout and atom/bond indexing.
        explanation : Explanation
            XAI output holding ``atom_scores`` and ``bond_scores`` to be
            mapped onto the molecular structure as a color gradient.

        Returns
        -------
        str
            A complete, self-contained SVG string that can be written to a
            file or rendered inline in a Jupyter notebook via
            ``IPython.display.SVG``.

        Raises
        ------
        NotImplementedError
            If the subclass has not implemented this method.
        """
        raise NotImplementedError
