"""PyTorch model wrappers for VisXAI.

Wraps PyTorch models exposing a specific calling convention through the
uniform :class:`~visxai.core.base_model.BaseModelWrapper` interface.

Provides :class:`PyTorchSequenceWrapper` for sequence models (hand-rolled
1D CNNs, Transformers, or real HuggingFace models) that accept
``input_ids``/``attention_mask``, and :class:`PyTorchGNNWrapper` for
PyTorch Geometric GNNs that accept ``(x, edge_index, batch)`` or, when
``uses_edge_attr=True``, ``(x, edge_index, edge_attr, batch)``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Batch

from visxai.core.base_model import BaseModelWrapper
from visxai.core.data_types import MoleculeRepresentation


class PyTorchSequenceWrapper(BaseModelWrapper):
    """Wrap a PyTorch sequence model expecting ``input_ids``/``attention_mask``.

    Covers both real HuggingFace ``transformers`` models (which return a
    ``ModelOutput`` object exposing ``.logits``) and hand-rolled models
    (e.g. a 1D CNN or a custom Transformer) built to accept the same
    calling convention: ``forward(input_ids, attention_mask) -> output``,
    where ``output`` either has an attribute named by ``output_attr`` or is
    itself the raw prediction tensor.

    This is a safe convention to assume regardless of the model's internal
    architecture or training-time feature scheme — it describes only how
    the model is *called*, not what its inputs mean, consistent with
    VisXAI's principle of not assuming a specific feature scheme (see
    ``CLAUDE.md``).

    Parameters
    ----------
    model : torch.nn.Module
        A PyTorch model exposing ``forward(input_ids, attention_mask)``.
    output_attr : str, optional
        Attribute name to extract from the model's output object (e.g.
        ``"logits"`` for a real HuggingFace ``ModelOutput``). Defaults to
        ``"logits"``. If the output object has no such attribute (e.g. a
        hand-rolled model that returns a raw tensor directly), the output
        itself is used unchanged.

    Attributes
    ----------
    model : torch.nn.Module
        The wrapped model, stored for direct access (e.g. by
        :class:`~visxai.explainers.attention.AttentionExplainer`, which
        needs to call it with ``output_attentions=True``).

    Examples
    --------
    >>> wrapper = PyTorchSequenceWrapper(my_transformer_model)
    >>> predictions = wrapper.predict(mol_rep)
    """

    def __init__(self, model: torch.nn.Module, output_attr: str = "logits") -> None:
        self.model: torch.nn.Module = model
        self._output_attr: str = output_attr

    def predict(self, mol_rep: MoleculeRepresentation) -> np.ndarray:
        """Run inference on the molecule's token sequence.

        Parameters
        ----------
        mol_rep : MoleculeRepresentation
            Featurised molecule container. Both ``token_ids`` and
            ``attention_mask`` must be populated (use
            :func:`visxai.features.sequences.generate_sequence_representation`).

        Returns
        -------
        numpy.ndarray
            1-D array of shape ``(n_outputs,)`` containing the model's raw
            prediction(s) for the input molecule. The molecule is always
            run as a batch of size 1.

        Raises
        ------
        ValueError
            If ``mol_rep.token_ids`` or ``mol_rep.attention_mask`` is
            ``None``.
        """
        if mol_rep.token_ids is None or mol_rep.attention_mask is None:
            raise ValueError(
                "PyTorchSequenceWrapper requires both token_ids and "
                "attention_mask to be populated on mol_rep; use "
                "generate_sequence_representation() to build it."
            )

        input_ids = torch.tensor([mol_rep.token_ids], dtype=torch.long)
        attention_mask = torch.tensor([mol_rep.attention_mask], dtype=torch.long)

        self.model.eval()
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = getattr(out, self._output_attr, out)
        return logits.detach().cpu().numpy().ravel()


class PyTorchGNNWrapper(BaseModelWrapper):
    """Wrap a PyTorch Geometric GNN expecting ``(x, edge_index, [edge_attr,] batch)``.

    ``forward(x, edge_index, batch)`` is PyTorch Geometric's own
    near-universal calling convention — a safe convention to assume
    regardless of the model's internal architecture or training-time node/
    edge feature scheme, analogous to :class:`PyTorchSequenceWrapper`
    assuming ``forward(input_ids, attention_mask)`` or
    :class:`~visxai.models.sklearn_wrapper.SklearnModelWrapper` assuming
    ``.predict(X)``. The risk in this package is always in feature
    *content* (see the architecture principle in ``CLAUDE.md``), never in
    a framework's own established calling convention.

    Models that also consume bond features (``edge_attr``) do **not** all
    agree on where that argument goes in their ``forward`` signature —
    unlike ``(x, edge_index, batch)``, there is no single dominant PyG
    convention here. ``uses_edge_attr`` is therefore an **explicit,
    required-to-opt-in flag** rather than something silently inferred from
    whether ``mol_rep.pyg_data.edge_attr`` happens to be populated — the
    same "don't silently assume a specific scheme" principle already
    applied to :func:`visxai.features.graphs.generate_graph_representation`'s
    featurizers.

    Parameters
    ----------
    model : torch.nn.Module
        A PyTorch Geometric model exposing ``forward(x, edge_index, batch)
        -> output`` (``uses_edge_attr=False``, the default) or
        ``forward(x, edge_index, edge_attr, batch) -> output``
        (``uses_edge_attr=True``), where ``output`` is the raw prediction
        tensor.
    uses_edge_attr : bool, optional
        Whether ``model.forward`` accepts ``edge_attr`` as its third
        positional argument (between ``edge_index`` and ``batch``).
        Defaults to ``False``.

    Attributes
    ----------
    model : torch.nn.Module
        The wrapped model, stored for direct access (e.g. by
        :class:`~visxai.explainers.gradient_based.IntegratedGradientsExplainer`,
        which needs gradients with respect to node/edge features and so
        cannot go through :meth:`predict`'s ``no_grad()`` block).
    uses_edge_attr : bool
        Stored flag, also consulted by
        :class:`~visxai.explainers.gradient_based.IntegratedGradientsExplainer`
        to decide whether to additionally attribute ``edge_attr`` and
        populate bond-level scores.

    Examples
    --------
    >>> wrapper = PyTorchGNNWrapper(my_gnn_model)
    >>> predictions = wrapper.predict(mol_rep)
    >>> edge_aware_wrapper = PyTorchGNNWrapper(my_edge_aware_gnn, uses_edge_attr=True)
    >>> predictions = edge_aware_wrapper.predict(mol_rep)
    """

    def __init__(self, model: torch.nn.Module, uses_edge_attr: bool = False) -> None:
        self.model: torch.nn.Module = model
        self.uses_edge_attr: bool = uses_edge_attr

    def predict(self, mol_rep: MoleculeRepresentation) -> np.ndarray:
        """Run inference on the molecule's graph representation.

        Parameters
        ----------
        mol_rep : MoleculeRepresentation
            Featurised molecule container. ``pyg_data`` must be populated
            (use
            :func:`visxai.features.graphs.generate_graph_representation`).
            If ``self.uses_edge_attr`` is ``True``, ``pyg_data.edge_attr``
            must also be populated (pass a ``bond_featurizer`` when
            building the representation).

        Returns
        -------
        numpy.ndarray
            1-D array of shape ``(n_outputs,)`` containing the model's raw
            prediction(s) for the input molecule. The molecule is always
            run as a batch of size 1 via
            :meth:`torch_geometric.data.Batch.from_data_list`, since PyG
            models expect batched input even for a single graph.

        Raises
        ------
        ValueError
            If ``mol_rep.pyg_data`` is ``None``, or if
            ``self.uses_edge_attr`` is ``True`` but
            ``mol_rep.pyg_data.edge_attr`` is ``None``.
        """
        if mol_rep.pyg_data is None:
            raise ValueError(
                "PyTorchGNNWrapper requires pyg_data to be populated on "
                "mol_rep; use generate_graph_representation() to build it."
            )

        batch = Batch.from_data_list([mol_rep.pyg_data])

        self.model.eval()
        with torch.no_grad():
            if self.uses_edge_attr:
                if batch.edge_attr is None:
                    raise ValueError(
                        "PyTorchGNNWrapper(uses_edge_attr=True) requires "
                        "mol_rep.pyg_data.edge_attr to be populated; pass a "
                        "bond_featurizer to generate_graph_representation() "
                        "to build it."
                    )
                out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            else:
                out = self.model(batch.x, batch.edge_index, batch.batch)

        return out.detach().cpu().numpy().ravel()
