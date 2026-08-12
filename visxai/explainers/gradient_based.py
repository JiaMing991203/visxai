"""Gradient-based explainers for PyTorch Geometric graph models.

Provides two explainers for :class:`~visxai.models.pytorch_wrapper.PyTorchGNNWrapper`-
wrapped GNNs, both producing per-atom scores via node ``i`` = atom ``i``
(guaranteed by :func:`visxai.features.graphs.generate_graph_representation`)
— a direct 1:1 assignment, no redistribution step needed, unlike the
fingerprint-bit or SMILES-token mapping problems solved elsewhere in
:mod:`visxai.utils.mapping`.

- :class:`IntegratedGradientsExplainer` — uses ``captum.attr.IntegratedGradients``
  to attribute node features (and, optionally, edge features — see below).
- :class:`GradCAMExplainer` — the graph adaptation of Grad-CAM: a single
  per-channel importance weight, averaged across every node's gradient at
  a caller-specified target layer, applied to that layer's own node
  activations. Requires an explicit ``target_layer`` — there is no
  reliable, convention-free way to identify "the last conv/GNN layer"
  generically across arbitrary architectures, the same "don't guess a
  model-specific detail" principle applied elsewhere in this package.
  Atom-level always; bond-level is opt-in via a second explicit
  ``edge_target_layer`` (see below) — unlike Integrated Gradients (which
  can differentiate any continuous tensor, including ``edge_attr``,
  without needing a specific layer's activations), Grad-CAM only has
  something to compute a channel weight/activation product over where a
  layer actually produces an edge-indexed intermediate tensor, which most
  standard GNN conv layers (``GCNConv``, ``GINConv``, etc.) don't expose —
  only message-passing-style layers that compute a per-edge message
  before aggregating into nodes do.

Edge-level attribution is computed by both explainers, under an opt-in
condition specific to each:

- :class:`IntegratedGradientsExplainer` computes it when
  ``model.uses_edge_attr`` (see
  :class:`~visxai.models.pytorch_wrapper.PyTorchGNNWrapper`) is ``True`` —
  i.e. only for models that actually consume ``edge_attr`` in their forward
  pass, and only for molecules built with a ``bond_featurizer``. Unlike the
  sequence path's bond attribution (inherently partial — most SMILES bonds
  have no explicit character to attribute a token to), this covers **every**
  bond that has an ``edge_attr`` vector, since Integrated Gradients can
  differentiate any continuous input regardless of whether it came from a
  SMILES character.
- :class:`GradCAMExplainer` computes it when a caller passes an explicit
  ``edge_target_layer`` whose output is an edge-indexed activation tensor
  (one row per *directed* edge in ``edge_index`` — the same "both
  ``(i,j)`` and ``(j,i)``" convention used everywhere else in this
  package). The same canonical formula (average gradient across the
  leading dimension as a shared per-channel weight, then ReLU'd
  weight-times-activation per row) is applied unchanged — it does not
  care whether that leading dimension indexes nodes or directed edges.
  Each bond's two directed-edge CAM scores are summed into one bond
  score, the same aggregation :class:`IntegratedGradientsExplainer` uses
  for ``edge_attr`` attributions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from captum.attr import IntegratedGradients
from torch_geometric.data import Data

from visxai.core.base_explainer import BaseExplainer
from visxai.core.base_model import BaseModelWrapper
from visxai.core.data_types import Explanation, MoleculeRepresentation
from visxai.models.pytorch_wrapper import PyTorchGNNWrapper


def _replicate_batch_metadata(
    edge_index: torch.Tensor,
    n_nodes: int,
    n_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replicate ``edge_index``/``batch`` into a mega-batch of ``n_steps`` copies.

    Captum's ``IntegratedGradients`` interpolates ``n_steps`` copies of the
    attributed input(s) between a baseline and the real input, then stacks
    them along dimension 0 and evaluates the forward function once on the
    stacked tensor(s). For a GNN, dimension 0 of the node (and edge)
    feature tensors is *nodes* (*edges*), not independent batch examples —
    so a naive forward call would see a mismatched node/edge count against
    the (unexpanded) ``edge_index``/``batch``. This helper builds the
    replicated ``edge_index``/``batch`` for ``n_steps`` identical-topology
    copies of the same graph (node indices offset per copy), mirroring how
    :meth:`torch_geometric.data.Batch.from_data_list` batches distinct
    graphs.

    Parameters
    ----------
    edge_index : torch.Tensor
        The molecule's ``edge_index``, shape ``(2, n_edges)``.
    n_nodes : int
        Number of nodes (atoms) in the molecule.
    n_steps : int
        Number of interpolation-step copies to replicate.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(edge_index_replicated, batch_replicated)``, shapes
        ``(2, n_steps * n_edges)`` and ``(n_steps * n_nodes,)``.
    """
    if n_steps == 1:
        return edge_index, torch.zeros(n_nodes, dtype=torch.long)

    edge_index_rep = torch.cat(
        [edge_index + i * n_nodes for i in range(n_steps)], dim=1
    )
    batch_rep = torch.cat(
        [torch.full((n_nodes,), i, dtype=torch.long) for i in range(n_steps)]
    )
    return edge_index_rep, batch_rep


def _replicated_forward(
    model: torch.nn.Module,
    edge_index: torch.Tensor,
    n_nodes: int,
):
    """Wrap a GNN's forward so Captum can interpolate node features alone.

    Parameters
    ----------
    model : torch.nn.Module
        The GNN being explained, exposing ``forward(x, edge_index, batch)``.
    edge_index : torch.Tensor
        The molecule's ``edge_index``, shape ``(2, n_edges)``.
    n_nodes : int
        Number of nodes (atoms) in the molecule.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        A function of ``x`` alone, suitable for
        ``captum.attr.IntegratedGradients``.
    """

    def forward_fn(x_expanded: torch.Tensor) -> torch.Tensor:
        n_steps = x_expanded.shape[0] // n_nodes
        edge_index_rep, batch_rep = _replicate_batch_metadata(
            edge_index, n_nodes, n_steps
        )
        return model(x_expanded, edge_index_rep, batch_rep)

    return forward_fn


def _replicated_forward_with_edge_attr(
    model: torch.nn.Module,
    edge_index: torch.Tensor,
    n_nodes: int,
):
    """Wrap a GNN's forward so Captum can jointly interpolate node and edge features.

    Parameters
    ----------
    model : torch.nn.Module
        The GNN being explained, exposing
        ``forward(x, edge_index, edge_attr, batch)``.
    edge_index : torch.Tensor
        The molecule's ``edge_index``, shape ``(2, n_edges)``.
    n_nodes : int
        Number of nodes (atoms) in the molecule.

    Returns
    -------
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        A function of ``(x, edge_attr)``, suitable for
        ``captum.attr.IntegratedGradients`` with multiple inputs.
    """

    def forward_fn(
        x_expanded: torch.Tensor, edge_attr_expanded: torch.Tensor
    ) -> torch.Tensor:
        n_steps = x_expanded.shape[0] // n_nodes
        edge_index_rep, batch_rep = _replicate_batch_metadata(
            edge_index, n_nodes, n_steps
        )
        return model(x_expanded, edge_index_rep, edge_attr_expanded, batch_rep)

    return forward_fn


class IntegratedGradientsExplainer(BaseExplainer):
    """Compute per-atom (and optionally per-bond) XAI scores using Integrated Gradients.

    Node-feature attributions are summed across feature dimensions to give
    one score per node, then assigned directly to the corresponding atom
    (node ``i`` = atom ``i`` by construction). When the wrapped model's
    ``uses_edge_attr`` is ``True``, edge-feature attributions are computed
    jointly and summed per bond the same way (each bond = two directed
    edges, summed together).

    Parameters
    ----------
    baseline : numpy.ndarray, optional
        A single feature vector of shape ``(n_node_features,)`` representing
        the "absence of an atom" reference point Integrated Gradients
        interpolates from, broadcast to every node in the molecule being
        explained. Defaults to ``None``, which uses an all-zeros baseline —
        Captum's own standard IG default and a property of the algorithm
        itself, not an assumption about the model's training data. Provide
        an explicit baseline if the target model's feature scheme makes an
        all-zeros vector a meaningless "absence" (see the architecture
        principle in the README on not assuming a specific feature
        scheme).
    edge_baseline : numpy.ndarray, optional
        The same idea as ``baseline``, but for edge features: a vector of
        shape ``(n_edge_features,)`` broadcast to every directed edge.
        Defaults to ``None`` (all-zeros). Only used when the wrapped
        model's ``uses_edge_attr`` is ``True``.
    n_steps : int, optional
        Number of interpolation steps between the baseline and the input.
        Defaults to ``50``, Captum's own common default.

    Attributes
    ----------
    _baseline : numpy.ndarray or None
        Stored per-feature node baseline vector.
    _edge_baseline : numpy.ndarray or None
        Stored per-feature edge baseline vector.
    _n_steps : int
        Stored interpolation step count.

    Examples
    --------
    >>> explainer = IntegratedGradientsExplainer()
    >>> explanation = explainer.explain(pytorch_gnn_wrapper, mol_rep)
    >>> explanation.atom_scores  # {atom_idx: ig_score, ...}
    >>> explanation.bond_scores  # {} unless model.uses_edge_attr is True
    """

    def __init__(
        self,
        baseline: Optional[np.ndarray] = None,
        edge_baseline: Optional[np.ndarray] = None,
        n_steps: int = 50,
    ) -> None:
        self._baseline: Optional[np.ndarray] = baseline
        self._edge_baseline: Optional[np.ndarray] = edge_baseline
        self._n_steps: int = n_steps

    def explain(
        self,
        model: BaseModelWrapper,
        mol_rep: MoleculeRepresentation,
    ) -> Explanation:
        """Compute atom-level (and optionally bond-level) IG attributions for a molecule.

        Parameters
        ----------
        model : BaseModelWrapper
            Must be a
            :class:`~visxai.models.pytorch_wrapper.PyTorchGNNWrapper`
            wrapping a GNN exposing ``forward(x, edge_index, batch)`` (or
            ``forward(x, edge_index, edge_attr, batch)`` if
            ``model.uses_edge_attr`` is ``True``).
        mol_rep : MoleculeRepresentation
            Featurised molecule. ``pyg_data`` must be populated (use
            :func:`visxai.features.graphs.generate_graph_representation`).
            If ``model.uses_edge_attr`` is ``True``,
            ``pyg_data.edge_attr`` must also be populated (pass a
            ``bond_featurizer`` when building the representation).

        Returns
        -------
        Explanation
            Dataclass with:

            - ``atom_scores`` — every RDKit atom index mapped to its
              Integrated Gradients contribution (sum across node feature
              dimensions). Every atom has a node, so every atom gets a
              real (non-default) value — no ``0.0``-fill step is needed.
            - ``bond_scores`` — every RDKit bond index mapped to its
              Integrated Gradients contribution (sum across edge feature
              dimensions, summed across both directed edges of the bond),
              if ``model.uses_edge_attr`` is ``True``. Empty dict
              otherwise.
            - ``metadata`` — contains ``"n_steps"`` (the interpolation step
              count used) and ``"predicted_value"`` (the model's raw
              prediction).

        Raises
        ------
        TypeError
            If ``model`` is not a
            :class:`~visxai.models.pytorch_wrapper.PyTorchGNNWrapper`.
        ValueError
            If ``mol_rep.pyg_data`` is ``None``; if a supplied ``baseline``
            does not have shape ``(n_node_features,)``; if a supplied
            ``edge_baseline`` does not have shape ``(n_edge_features,)``;
            or if ``model.uses_edge_attr`` is ``True`` but
            ``mol_rep.pyg_data.edge_attr`` is ``None``.
        """
        if not isinstance(model, PyTorchGNNWrapper):
            raise TypeError(
                f"IntegratedGradientsExplainer requires a PyTorchGNNWrapper, "
                f"got {type(model).__name__}"
            )
        if mol_rep.pyg_data is None:
            raise ValueError(
                "IntegratedGradientsExplainer requires pyg_data to be "
                "populated on mol_rep; use generate_graph_representation() "
                "to build it."
            )

        pyg_data = mol_rep.pyg_data
        x: torch.Tensor = pyg_data.x
        edge_index: torch.Tensor = pyg_data.edge_index
        n_nodes, n_features = x.shape

        baseline_t = self._resolve_baseline(self._baseline, n_features, x)

        model.model.eval()

        if model.uses_edge_attr:
            atom_scores, bond_scores = self._explain_with_edge_attr(
                model, pyg_data, x, edge_index, n_nodes, baseline_t
            )
        else:
            forward_fn = _replicated_forward(model.model, edge_index, n_nodes)
            ig = IntegratedGradients(forward_fn)
            attributions = ig.attribute(x, baselines=baseline_t, n_steps=self._n_steps)
            per_node_score = attributions.sum(dim=-1).detach().cpu().numpy()
            atom_scores = {i: float(per_node_score[i]) for i in range(n_nodes)}
            bond_scores = {}

        return Explanation(
            atom_scores=atom_scores,
            bond_scores=bond_scores,
            metadata={
                "n_steps": self._n_steps,
                "predicted_value": float(model.predict(mol_rep)[0]),
                # Node i is atom i, with no redistribution, so there is
                # nothing to itemise: atom_provenance/bond_provenance stay
                # empty and the descriptor alone carries the convention.
                "attribution": "graph/direct",
            },
        )

    @staticmethod
    def _resolve_baseline(
        baseline: Optional[np.ndarray],
        n_features: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Broadcast a per-feature baseline vector across ``reference``'s rows."""
        if baseline is None:
            return torch.zeros_like(reference)
        baseline_vec = torch.as_tensor(baseline, dtype=torch.float32)
        if tuple(baseline_vec.shape) != (n_features,):
            raise ValueError(
                f"baseline must have shape ({n_features},), got "
                f"{tuple(baseline_vec.shape)}"
            )
        return baseline_vec.unsqueeze(0).expand(reference.shape[0], -1).contiguous()

    def _explain_with_edge_attr(
        self,
        model: PyTorchGNNWrapper,
        pyg_data: Data,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        n_nodes: int,
        baseline_t: torch.Tensor,
    ) -> tuple[dict[int, float], dict[int, float]]:
        """Jointly attribute node and edge features, aggregating to atoms and bonds."""
        edge_attr = pyg_data.edge_attr
        if edge_attr is None:
            raise ValueError(
                "model.uses_edge_attr is True but mol_rep.pyg_data.edge_attr "
                "is None; pass a bond_featurizer to "
                "generate_graph_representation() to build it."
            )
        n_edges_directed, n_edge_features = edge_attr.shape

        edge_baseline_t = self._resolve_baseline(
            self._edge_baseline, n_edge_features, edge_attr
        )

        forward_fn = _replicated_forward_with_edge_attr(model.model, edge_index, n_nodes)
        ig = IntegratedGradients(forward_fn)
        x_attr, edge_attr_attr = ig.attribute(
            (x, edge_attr),
            baselines=(baseline_t, edge_baseline_t),
            n_steps=self._n_steps,
        )

        per_node_score = x_attr.sum(dim=-1).detach().cpu().numpy()
        atom_scores = {i: float(per_node_score[i]) for i in range(n_nodes)}

        per_edge_score = edge_attr_attr.sum(dim=-1).detach().cpu().numpy()
        n_bonds = n_edges_directed // 2
        bond_scores = {
            k: float(per_edge_score[2 * k] + per_edge_score[2 * k + 1])
            for k in range(n_bonds)
        }

        return atom_scores, bond_scores


class GradCAMExplainer(BaseExplainer):
    """Compute per-atom XAI scores for GNNs using the graph adaptation of Grad-CAM.

    Follows the canonical Grad-CAM formulation (as adapted for GNNs, e.g.
    Pope et al. 2019): a single per-channel importance weight is computed
    by averaging that channel's gradient *across every node* (the
    graph-adapted analogue of Grad-CAM's spatial global-average-pooling
    step), then applied uniformly to every node's own activation at the
    target layer:

    .. code-block:: text

        alpha_c = mean over all nodes i of grad(output, activation)[i, c]
        score_i = ReLU( sum_c alpha_c * activation[i, c] )

    ``alpha`` is the *same* vector for every node — only the activation
    term varies node to node. This is a different (and more standard)
    result than what ``captum.attr.LayerGradCam`` would give out of the
    box on a plain ``(n_nodes, hidden_dim)`` node-activation tensor: Captum
    treats a 2-D tensor's dimension 0 as independent batch examples with no
    "spatial" dimension to average over, so its default behaviour computes
    a *per-node local* gradient×activation with no cross-node weight
    sharing — a real deviation from Grad-CAM's defining class-discriminative
    global weighting. This explainer is therefore hand-rolled with plain
    PyTorch forward/backward hooks rather than wrapping Captum's
    ``LayerGradCam``.

    Scores are **non-negative** (the ``ReLU`` is part of the algorithm's
    definition, not a visualization choice) — like
    :class:`~visxai.explainers.attention.AttentionExplainer`'s raw
    attention scores, and unlike
    :class:`IntegratedGradientsExplainer`'s signed scores.
    :class:`~visxai.visualizers.rdkit_2d.RDKitSVGVisualizer` already
    renders an all-non-negative score range correctly (white-to-green
    gradient), so no visualizer changes are needed.

    Atom-level attribution is always computed, from ``target_layer``.
    Bond-level attribution is opt-in via a second, independent
    ``edge_target_layer`` — see the module docstring for why this is a
    separate parameter rather than automatically derived from
    ``target_layer``: it requires a layer whose activation is indexed by
    directed edge, not node, which most standard GNN conv layers don't
    expose. When supplied, the same canonical formula is applied to that
    layer's edge-indexed activation instead of a node-indexed one, and the
    two directed-edge scores per bond are summed into one bond score
    (mirroring how :class:`IntegratedGradientsExplainer` aggregates its
    own edge-feature attributions).

    Parameters
    ----------
    target_layer : torch.nn.Module
        The GNN layer (a submodule of the wrapped model) whose node
        activations Grad-CAM is computed over. Required, with no
        auto-detection. Must be a layer whose output is a 2-D tensor of
        shape ``(n_nodes, hidden_dim)`` — true for essentially all
        standard PyG conv layers (``GCNConv``, ``GINConv``, ``SAGEConv``,
        etc.), which return node-indexed embeddings.
    edge_target_layer : torch.nn.Module, optional
        A GNN layer (a submodule of the wrapped model) whose output is an
        edge-indexed activation tensor of shape
        ``(n_edges_directed, hidden_dim)`` — one row per directed edge in
        ``edge_index``, e.g. a message-passing layer's per-edge message
        computed before aggregation into nodes. Defaults to ``None``,
        which skips bond-level attribution entirely (``bond_scores`` stays
        ``{}``). Required, with no auto-detection, for the same reason as
        ``target_layer``.

    Attributes
    ----------
    _target_layer : torch.nn.Module
        Stored target layer for atom-level attribution.
    _edge_target_layer : torch.nn.Module or None
        Stored target layer for bond-level attribution, if any.

    Examples
    --------
    >>> explainer = GradCAMExplainer(target_layer=my_gnn_model.conv2)
    >>> explanation = explainer.explain(pytorch_gnn_wrapper, mol_rep)
    >>> explanation.atom_scores  # {atom_idx: cam_score >= 0.0, ...}
    >>> explanation.bond_scores  # {} — no edge_target_layer was given

    >>> explainer = GradCAMExplainer(
    ...     target_layer=my_gnn_model.conv2,
    ...     edge_target_layer=my_gnn_model.msg_lin,
    ... )
    >>> explanation = explainer.explain(pytorch_gnn_wrapper, mol_rep)
    >>> explanation.bond_scores  # {bond_idx: cam_score >= 0.0, ...}
    """

    def __init__(
        self,
        target_layer: torch.nn.Module,
        edge_target_layer: Optional[torch.nn.Module] = None,
    ) -> None:
        self._target_layer: torch.nn.Module = target_layer
        self._edge_target_layer: Optional[torch.nn.Module] = edge_target_layer

    def explain(
        self,
        model: BaseModelWrapper,
        mol_rep: MoleculeRepresentation,
    ) -> Explanation:
        """Compute atom-level Grad-CAM attributions for a molecule.

        Parameters
        ----------
        model : BaseModelWrapper
            Must be a
            :class:`~visxai.models.pytorch_wrapper.PyTorchGNNWrapper`
            wrapping a GNN whose ``target_layer`` (and ``edge_target_layer``,
            if given — both passed to this explainer's constructor) are
            among its own submodules.
        mol_rep : MoleculeRepresentation
            Featurised molecule. ``pyg_data`` must be populated (use
            :func:`visxai.features.graphs.generate_graph_representation`).
            If ``model.uses_edge_attr`` is ``True``,
            ``pyg_data.edge_attr`` must also be populated.

        Returns
        -------
        Explanation
            Dataclass with:

            - ``atom_scores`` — every RDKit atom index mapped to its
              (non-negative) Grad-CAM score.
            - ``bond_scores`` — every RDKit bond index mapped to its
              (non-negative) Grad-CAM score, if ``edge_target_layer`` was
              given to the constructor. Empty dict otherwise.
            - ``metadata`` — contains ``"predicted_value"`` (the model's
              raw prediction).

        Raises
        ------
        TypeError
            If ``model`` is not a
            :class:`~visxai.models.pytorch_wrapper.PyTorchGNNWrapper`.
        ValueError
            If ``mol_rep.pyg_data`` is ``None``; if ``target_layer`` or
            ``edge_target_layer`` is not a submodule of ``model.model``;
            if ``model.uses_edge_attr`` is ``True`` but
            ``mol_rep.pyg_data.edge_attr`` is ``None``; if no gradient
            reaches ``target_layer``'s or ``edge_target_layer``'s output
            (e.g. the layer was never invoked during the forward pass); or
            if ``edge_target_layer``'s activation isn't shaped one row per
            directed edge in ``edge_index``.
        """
        if not isinstance(model, PyTorchGNNWrapper):
            raise TypeError(
                f"GradCAMExplainer requires a PyTorchGNNWrapper, "
                f"got {type(model).__name__}"
            )
        if mol_rep.pyg_data is None:
            raise ValueError(
                "GradCAMExplainer requires pyg_data to be populated on "
                "mol_rep; use generate_graph_representation() to build it."
            )
        if self._target_layer not in model.model.modules():
            raise ValueError(
                "target_layer must be a submodule of the wrapped model "
                "(model.model); it was not found among model.model.modules()."
            )
        if (
            self._edge_target_layer is not None
            and self._edge_target_layer not in model.model.modules()
        ):
            raise ValueError(
                "edge_target_layer must be a submodule of the wrapped "
                "model (model.model); it was not found among "
                "model.model.modules()."
            )

        pyg_data = mol_rep.pyg_data
        x: torch.Tensor = pyg_data.x
        edge_index: torch.Tensor = pyg_data.edge_index
        n_nodes = x.shape[0]
        n_edges_directed = edge_index.shape[1]
        batch = torch.zeros(n_nodes, dtype=torch.long)

        captured: dict[str, torch.Tensor] = {}

        def make_hook(key: str):
            def forward_hook(
                _module: torch.nn.Module,
                _input: tuple,
                output: torch.Tensor,
            ) -> None:
                output.retain_grad()
                captured[key] = output

            return forward_hook

        handles = [self._target_layer.register_forward_hook(make_hook("node"))]
        if self._edge_target_layer is not None:
            handles.append(
                self._edge_target_layer.register_forward_hook(make_hook("edge"))
            )

        try:
            model.model.eval()
            if model.uses_edge_attr:
                if pyg_data.edge_attr is None:
                    raise ValueError(
                        "model.uses_edge_attr is True but "
                        "mol_rep.pyg_data.edge_attr is None; pass a "
                        "bond_featurizer to generate_graph_representation() "
                        "to build it."
                    )
                out = model.model(x, edge_index, pyg_data.edge_attr, batch)
            else:
                out = model.model(x, edge_index, batch)

            model.model.zero_grad()
            out.sum().backward()
        finally:
            for handle in handles:
                handle.remove()

        node_activation = captured.get("node")
        if node_activation is None or node_activation.grad is None:
            raise ValueError(
                "No gradient reached target_layer's output; it may not "
                "have been invoked during the forward pass."
            )

        alpha = node_activation.grad.mean(dim=0)  # (hidden_dim,), shared across nodes
        cam = torch.relu((node_activation * alpha.unsqueeze(0)).sum(dim=-1))  # (n_nodes,)
        cam_np = cam.detach().cpu().numpy()
        atom_scores = {i: float(cam_np[i]) for i in range(n_nodes)}

        bond_scores: dict[int, float] = {}
        if self._edge_target_layer is not None:
            edge_activation = captured.get("edge")
            if edge_activation is None or edge_activation.grad is None:
                raise ValueError(
                    "No gradient reached edge_target_layer's output; it "
                    "may not have been invoked during the forward pass."
                )
            if edge_activation.shape[0] != n_edges_directed:
                raise ValueError(
                    f"edge_target_layer's activation must have one row "
                    f"per directed edge in edge_index (expected "
                    f"{n_edges_directed}, got {edge_activation.shape[0]}); "
                    f"most standard node-only conv layers (GCNConv, "
                    f"GINConv, etc.) produce node-indexed activations "
                    f"instead — edge-level Grad-CAM requires a "
                    f"message-passing layer whose output is indexed by "
                    f"directed edge, e.g. a per-edge message submodule "
                    f"computed before aggregation into nodes."
                )

            edge_alpha = edge_activation.grad.mean(dim=0)
            edge_cam = torch.relu(
                (edge_activation * edge_alpha.unsqueeze(0)).sum(dim=-1)
            )
            edge_cam_np = edge_cam.detach().cpu().numpy()
            n_bonds = n_edges_directed // 2
            bond_scores = {
                k: float(edge_cam_np[2 * k] + edge_cam_np[2 * k + 1])
                for k in range(n_bonds)
            }

        return Explanation(
            atom_scores=atom_scores,
            bond_scores=bond_scores,
            metadata={
                "predicted_value": float(model.predict(mol_rep)[0]),
                # See IntegratedGradientsExplainer: 1:1 node-to-atom mapping
                # means there is no sharing to record.
                "attribution": "graph/direct",
            },
        )
