"""Interactive, in-notebook dashboard for comparing pre-computed explanations.

Phase 1 of interactive visualization (see
the project's design notes for the staged plan).
Wires standard ``ipywidgets`` controls around :class:`~visxai.visualizers.rdkit_2d.RDKitSVGVisualizer`
so a reader can flip between molecules/explainers and adjust the color
scale without editing and re-running notebook cells.

**This module is deliberately not a** :class:`~visxai.core.base_visualizer.BaseVisualizer`
**subclass.** That ABC's contract is ``visualize(mol_rep, explanation) -> str``
(an SVG string), which every existing caller relies on; a widget-returning
component would silently change what the ABC promises. :class:`ExplanationDashboard`
is a plain class that *uses* ``RDKitSVGVisualizer.visualize()`` internally
instead.

**Scope: re-rendering, not re-explaining.** The dashboard operates on a
collection of *already-computed* ``(mol_rep, explanation)`` results that
the caller registers up front via :meth:`ExplanationDashboard.add`.
Interactions recolor and re-render those existing scores; they never
re-run an explainer. This is a deliberate scope choice — re-running e.g.
``IntegratedGradientsExplainer`` (``n_steps`` forward/backward passes)
on every slider drag would be laggy for any realistically-sized model.

**Every interaction is a full recompute-and-redraw, not a DOM patch.**
RDKit's ``MolDraw2DSVG`` has no incremental/patch API — each redraw
regenerates a complete new SVG string, which then replaces the panel's
HTML wholesale. This is a perfectly workable interaction model at this
scale, but it is a request/response cycle per interaction rather than a
live-patching canvas.

``ipywidgets`` is an optional dependency (the ``[viz]`` extra) — importing
this module without it installed succeeds, and the error is raised at
:class:`ExplanationDashboard` construction time with an install hint.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

try:  # pragma: no cover - trivial import guard
    import ipywidgets as widgets
except ImportError:  # pragma: no cover - exercised only without the [viz] extra
    widgets = None  # type: ignore[assignment]

from visxai.core.data_types import Explanation, MoleculeRepresentation
from visxai.visualizers.hover_widget import MoleculeHoverWidget
from visxai.visualizers.hover_widget import anywidget as _anywidget
from visxai.visualizers.rdkit_2d import RDKitSVGVisualizer


# A registered result is keyed by (molecule name, explainer name) so the two
# form independent selector axes rather than one flattened label.
_ResultKey = Tuple[str, str]
_Result = Tuple[MoleculeRepresentation, Explanation]

# Shown in a panel whose (molecule, explainer) pair was never registered.
# Selector options are global (every molecule x every explainer), so a
# reachable-but-empty combination is normal, not an error -- e.g. when one
# molecule was explained with two explainers and another with only one.
_MISSING_RESULT_HTML = (
    "<pre style='color:#888;font-family:sans-serif'>"
    "No result registered for this molecule / explainer combination."
    "</pre>"
)

_INSTALL_HINT = (
    "ExplanationDashboard requires ipywidgets, which is an optional "
    "dependency. Install it with:  pip install 'visxai[viz]'"
)


def _score_range(explanations: List[Explanation], which: str) -> Optional[Tuple[float, float]]:
    """Compute the global ``(min, max)`` score across several explanations.

    Used to bound the dashboard's range sliders against every registered
    result at once, so dragging a slider can express any scale a user might
    reasonably want for the collection — rather than being clamped to
    whichever single explanation happens to be selected.

    Parameters
    ----------
    explanations : list[Explanation]
        The explanations to pool.
    which : {"atom", "bond"}
        Whether to pool ``atom_scores`` or ``bond_scores``.

    Returns
    -------
    tuple[float, float] or None
        The pooled ``(min, max)``, or ``None`` when no explanation has any
        score of that type at all (e.g. ``"bond"`` for a collection of
        atom-only explanations).

    Raises
    ------
    ValueError
        If ``which`` is not ``"atom"`` or ``"bond"``.
    """
    if which not in ("atom", "bond"):
        raise ValueError(f"which must be 'atom' or 'bond', got {which!r}")

    values: List[float] = []
    for explanation in explanations:
        scores = explanation.atom_scores if which == "atom" else explanation.bond_scores
        values.extend(scores.values())

    if not values:
        return None
    return (min(values), max(values))


def _pad_slider_bounds(score_range: Tuple[float, float]) -> Tuple[float, float]:
    """Widen a score range slightly so a slider can move past the observed extremes.

    A slider clamped exactly to the data's own min/max can only ever
    *narrow* the color scale, never widen it — but widening is a legitimate
    thing to want (e.g. pinning a deliberately generous shared scale so two
    molecules' colors stay comparable). Also guards the degenerate
    ``min == max`` case, which ``ipywidgets`` rejects outright as slider
    bounds.

    Parameters
    ----------
    score_range : tuple[float, float]
        The observed ``(min, max)``.

    Returns
    -------
    tuple[float, float]
        A padded ``(min, max)`` with ``min < max`` guaranteed.
    """
    low, high = score_range
    span = high - low
    if span <= 0.0:
        # Degenerate: pad by a fixed amount around the single value, scaled
        # to its own magnitude so this stays sensible for both 1e-6 and 1e6.
        pad = max(abs(low) * 0.5, 1.0)
    else:
        pad = span * 0.25
    return (low - pad, high + pad)


def _format_summary(explanation: Explanation) -> str:
    """Format an explanation's raw score sums and metadata as plain text.

    Deliberately reports **raw sums only**, with no completeness-axiom
    claim attached. ``sum(atom_scores) + sum(bond_scores)`` equals the
    model's prediction difference *only* for the graph path's
    ``IntegratedGradientsExplainer`` — it does not for the sequence path's
    same-named explainer (attribution mass on ``[CLS]``/``[SEP]`` tokens is
    dropped when aggregating onto atoms), nor for ``TreeSHAPExplainer``
    under its default ``bond_score_mode="duplicate"`` (which distributes
    each bit's full score twice, once to atoms and once to bonds). Labeling
    these sums as a completeness check would therefore be wrong on two of
    the three pathways, so the panel presents the numbers and leaves the
    interpretation to the reader.

    Parameters
    ----------
    explanation : Explanation
        The explanation to summarize.

    Returns
    -------
    str
        A newline-separated plain-text summary — score counts, raw sums,
        and every ``explanation.metadata`` entry.
    """
    lines: List[str] = []

    atom_scores = explanation.atom_scores
    lines.append(f"atoms: {len(atom_scores)} scored, sum = {sum(atom_scores.values()):+.4f}")

    bond_scores = explanation.bond_scores
    if bond_scores:
        lines.append(f"bonds: {len(bond_scores)} scored, sum = {sum(bond_scores.values()):+.4f}")
    else:
        lines.append("bonds: none scored")

    for key, value in explanation.metadata.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            lines.append(f"{key} = {value}")
        else:
            lines.append(f"{key} = {value:+.4f}")

    return "\n".join(lines)


def _summary_html(explanation: Explanation) -> str:
    """Wrap :func:`_format_summary`'s text in a monospace ``<pre>`` block.

    Parameters
    ----------
    explanation : Explanation
        The explanation to summarize.

    Returns
    -------
    str
        An HTML fragment suitable for an ``ipywidgets.HTML`` widget's value.
    """
    return (
        "<pre style='margin:0;font-size:11px;line-height:1.4'>"
        f"{_format_summary(explanation)}"
        "</pre>"
    )


class ExplanationDashboard:
    """An ipywidgets dashboard for viewing and comparing pre-computed explanations.

    Register results up front with :meth:`add` or :meth:`from_results`, then
    call :meth:`display` (in a notebook) or :meth:`widget` (to embed the root
    container yourself). Molecule and explainer are **independent selector
    axes**, so registering the same molecule under two explainers gives one
    molecule dropdown and one explainer dropdown rather than a single
    flattened list.

    ``n_panels`` chooses the layout: ``2`` (the default) renders two panels
    side by side, each with its own molecule and explainer selectors, so any
    two explanations can be compared directly — the same molecule under two
    explainers, two molecules under the same explainer, or any other
    pairing. ``n_panels=1`` renders a single panel for looking at one
    explanation at a time.

    **Color scaling has three modes**, selected at runtime, because "shared
    vs per-panel" and "automatic vs manual" are independent questions:

    - **Automatic, shared** (the default under ``legend_units="raw"``) —
      one range pooled across the explanations *currently displayed*,
      recomputed whenever a selector changes. This is what makes panels
      comparable. It pools over what's on screen rather than the whole
      registry on purpose: pooling over everything registered lets a
      molecule you never display flatten the ones you do.
    - **Automatic, per-panel** (the default under
      ``legend_units="normalized"``) — each panel normalizes against its
      own scores alone (``RDKitSVGVisualizer``'s own default). Best
      contrast within a panel; colors not comparable between panels. With
      ``n_panels=1`` this is identical to shared, so the selector collapses
      to Automatic/Manual.

    **The default mode follows ``legend_units``, because the two are not
    independent.** ``"shared"`` + ``"normalized"`` is degenerate: one
    pooled range means both panels apply the *same* mapping, so the
    normalized number is only the raw score rescaled — it carries nothing
    raw units don't — while each panel's colors silently depend on the
    other panel's scores. The normalized legend cannot reveal that, since
    it prints the output of the division and never the divisor, so its
    ticks read ``-1.00``/``0``/``+1.00`` whatever range is in force.
    Changing one panel's explainer then recolored the other with no visible
    cause. Normalized means "relative importance within *this* molecule",
    which is a per-panel idea; comparing magnitudes across panels is what
    raw units are for.
    - **Manual** — the ``atom_range``/``bond_range`` sliders take over, for
      deliberately clipping outliers so structure in the remaining scores
      becomes visible (the standard ``vmin``/``vmax`` need). The automatic
      modes keep the sliders in sync with the range actually in use, so
      switching to manual starts from there rather than jumping.

    Interactions re-render already-computed scores; they never re-run an
    explainer (see the module docstring for why).

    Parameters
    ----------
    width : int, optional
        Molecule canvas width in pixels for each panel. Defaults to ``400``.
    height : int, optional
        Molecule canvas height in pixels for each panel. Defaults to ``300``.
    show_tooltips : bool, optional
        Whether each rendered molecule carries native hover tooltips
        (see :class:`~visxai.visualizers.rdkit_2d.RDKitSVGVisualizer`).
        Defaults to ``True`` here — unlike the visualizer's own default of
        ``False`` — since per-element detail on hover is squarely the point
        of an interactive view. Safe to leave on with two panels in one
        document: every legend gradient id is already randomized per call,
        so the panels' SVGs cannot collide.
    legend_units : {"normalized", "raw"}, optional
        What the color bar's min/max ticks print. Defaults to
        ``"normalized"`` (the ``±1`` values that actually drive the color).
        Switchable at runtime from the "Legend" control; this only sets the
        initial value, which the control then owns. Raises ``ValueError``
        for any other value.

        The hover readout always shows **both** the raw score and the
        normalized value, so it reconciles with the bar either way — the
        two are different numbers, and reading one off a bar labelled in the
        other is the confusion this switch exists to resolve.
    hover_link : bool, optional
        Whether each panel is a hover-linked
        :class:`~visxai.visualizers.hover_widget.MoleculeHoverWidget` — the
        structure paired with an XSMILES-style SMILES strip, cross-highlighting
        on hover with a cursor-following readout. Defaults to ``True``.
        ``False`` renders the static SVG panel instead (and then
        ``show_tooltips`` still applies; with hover-linking on it is forced
        off, since the widget provides its own instant readout and a native
        ``<title>`` would stack the browser's slow tooltip on top).

        If ``anywidget`` isn't installed, this **warns and falls back** to
        the static panel rather than raising: the dashboard stays usable,
        but the degradation is announced rather than silent. ``hover_link``
        records what was asked for; the panels reflect what was available.
    n_panels : int, optional
        How many panels to render side by side. Defaults to ``2`` (a
        comparison view). Pass ``1`` for a single-explanation view — one
        molecule and one explainer at a time — which also collapses the
        color-scale selector to Automatic/Manual, since there is no second
        panel to share a scale with. Must be an integer ``>= 1``; values
        above ``2`` are allowed but get cramped at the default ``width``.
    **visualizer_kwargs : Any
        Any other :class:`~visxai.visualizers.rdkit_2d.RDKitSVGVisualizer`
        constructor argument (``color_scale``, ``legend_units``,
        ``scale_atom_radius``, ``show_legend``, ``legend_width``), applied
        identically to every panel. ``atom_range``/``bond_range`` are
        driven by the color-scale controls and must not be passed here.

    Raises
    ------
    ImportError
        If ``ipywidgets`` is not installed (the ``[viz]`` extra).
    ValueError
        If ``atom_range`` or ``bond_range`` is passed via
        ``visualizer_kwargs``, or if ``n_panels`` is not an integer ``>= 1``.

    Examples
    --------
    Incrementally, via :meth:`add`:

    >>> dashboard = ExplanationDashboard()
    >>> dashboard.add("aspirin", aspirin_rep, shap_explanation, explainer="TreeSHAP")
    >>> dashboard.add("aspirin", aspirin_rep, ig_explanation, explainer="IG")
    >>> dashboard.add("caffeine", caffeine_rep, caffeine_ig, explainer="IG")
    >>> dashboard.display()

    Or all at once via :meth:`from_results`, which writes each ``mol_rep``
    once per molecule rather than once per explanation:

    >>> dashboard = ExplanationDashboard.from_results({
    ...     "aspirin": (aspirin_rep, {"TreeSHAP": shap_explanation, "IG": ig_explanation}),
    ...     "caffeine": (caffeine_rep, {"IG": caffeine_ig}),
    ... })
    >>> dashboard.display()

    A single-panel view, for looking at one explanation at a time:

    >>> ExplanationDashboard.from_results(results, n_panels=1).display()
    """

    def __init__(
        self,
        width: int = 400,
        height: int = 300,
        show_tooltips: bool = True,
        n_panels: int = 2,
        hover_link: bool = True,
        legend_units: str = "normalized",
        **visualizer_kwargs: Any,
    ) -> None:
        if widgets is None:
            raise ImportError(_INSTALL_HINT)
        for reserved in ("atom_range", "bond_range"):
            if reserved in visualizer_kwargs:
                raise ValueError(
                    f"{reserved} is driven by the dashboard's color-scale controls "
                    "and cannot be passed to ExplanationDashboard; select "
                    "'Manual' and use the sliders instead."
                )
        # No visualizer_kwargs guard for legend_units, unlike atom_range and
        # bond_range: it's an explicit named parameter, so it binds there and
        # can never reach **visualizer_kwargs. A check would be unreachable.
        if legend_units not in ("raw", "normalized"):
            raise ValueError(
                f"legend_units must be 'raw' or 'normalized', got {legend_units!r}"
            )
        # Requested vs. actually available are tracked separately: falling
        # back must be visible, never a silent downgrade that leaves the
        # user believing hover-linking is on when it isn't.
        self.hover_link: bool = bool(hover_link)
        if hover_link and _anywidget is None:
            warnings.warn(
                "hover_link=True but anywidget is not installed, so the panels "
                "fall back to a static SVG with no hover-linking. Install it "
                "with:  pip install 'visxai[viz]'",
                RuntimeWarning,
                stacklevel=2,
            )
        self._hover_link_active: bool = bool(hover_link) and _anywidget is not None

        self.width: int = width
        self.height: int = height
        self.show_tooltips: bool = show_tooltips
        # Kept as instance state rather than read off the control widget, so
        # render_svg() works identically before the widget tree is built.
        # The control's observer writes back here.
        self.legend_units: str = legend_units
        self.visualizer_kwargs: Dict[str, Any] = dict(visualizer_kwargs)
        self._results: Dict[_ResultKey, _Result] = {}
        self._root: Optional[Any] = None
        self._n_panels: int = 0
        self.n_panels = n_panels  # validated by the property setter below
        self._controls: Dict[str, Any] = {}
        self._panels: List[Dict[str, Any]] = []
        # Re-entrancy guard: _refresh writes slider values in the auto modes,
        # which would otherwise re-trigger its own observer.
        self._refreshing: bool = False

    @property
    def n_panels(self) -> int:
        """int: How many panels the view renders side by side.

        Assignable after construction; doing so discards any
        already-built widget tree so the next :meth:`widget` or
        :meth:`display` call reflects the new count. Without that
        invalidation the assignment silently did nothing, since
        :meth:`widget` returns its cached root — the same staleness
        :meth:`add` already guards against.
        """
        return self._n_panels

    @n_panels.setter
    def n_panels(self, value: int) -> None:
        # bool is an int subclass, so `True` would otherwise satisfy
        # `isinstance(value, int) and value >= 1` and silently build a
        # one-panel dashboard from `n_panels=True`.
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"n_panels must be an integer >= 1, got {value!r}")
        if value != self._n_panels:
            self._n_panels = value
            self._root = None

    # -- registry ---------------------------------------------------------

    @classmethod
    def from_results(
        cls,
        results: Dict[str, Tuple[MoleculeRepresentation, Dict[str, Explanation]]],
        **dashboard_kwargs: Any,
    ) -> "ExplanationDashboard":
        """Build a dashboard from a nested ``{molecule: (mol_rep, {explainer: explanation})}`` dict.

        A convenience wrapper over repeated :meth:`add` calls, structured so
        each ``mol_rep`` is written **once per molecule** rather than once
        per explanation. ``mol_rep`` is a property of the molecule, not of
        any one explanation of it, so a flat call-per-explanation forces it
        to be restated for every explainer applied to the same molecule:

        .. code-block:: python

            # flat: aspirin_rep restated per explainer
            dashboard.add("aspirin", aspirin_rep, shap_expl, explainer="TreeSHAP")
            dashboard.add("aspirin", aspirin_rep, ig_expl, explainer="IG")

            # nested: written once
            ExplanationDashboard.from_results({
                "aspirin": (aspirin_rep, {"TreeSHAP": shap_expl, "IG": ig_expl}),
            })

        Purely additive sugar — this delegates to :meth:`add`, which stays
        the primitive. It does **not** run any explainer: every explanation
        passed here must already be computed, same as :meth:`add` (see the
        module docstring on why live re-explaining is out of scope).

        Molecules and explainers keep their dict insertion order, so the
        selectors' order is whatever the caller wrote. A molecule need not
        supply every explainer — a combination absent here is simply
        unregistered, and its panel renders a "no result" message.

        Parameters
        ----------
        results : dict[str, tuple[MoleculeRepresentation, dict[str, Explanation]]]
            Keyed by molecule display name. Each value is that molecule's
            representation paired with a dict of its explanations, keyed by
            explainer display name.
        **dashboard_kwargs : Any
            Passed straight to :class:`ExplanationDashboard`'s constructor
            (``width``, ``height``, ``show_tooltips``, and any
            ``RDKitSVGVisualizer`` keyword).

        Returns
        -------
        ExplanationDashboard
            A dashboard with every ``(molecule, explainer)`` pair registered.

        Raises
        ------
        ValueError
            If ``results`` is empty, if any molecule maps to an empty
            explanation dict, or (via :meth:`add`) if any name is empty.

        Examples
        --------
        >>> dashboard = ExplanationDashboard.from_results({
        ...     "aspirin": (aspirin_rep, {"TreeSHAP": shap_expl, "IG": ig_expl}),
        ...     "caffeine": (caffeine_rep, {"IG": caffeine_ig}),
        ... })
        >>> dashboard.display()
        """
        if not results:
            raise ValueError("results must not be empty")

        dashboard = cls(**dashboard_kwargs)
        for molecule, (mol_rep, explanations) in results.items():
            if not explanations:
                raise ValueError(
                    f"molecule {molecule!r} maps to an empty explanation dict; "
                    "give it at least one {explainer_name: explanation} entry"
                )
            for explainer, explanation in explanations.items():
                dashboard.add(molecule, mol_rep, explanation, explainer=explainer)
        return dashboard

    def add(
        self,
        molecule: str,
        mol_rep: MoleculeRepresentation,
        explanation: Explanation,
        explainer: str = "default",
    ) -> "ExplanationDashboard":
        """Register one pre-computed ``(mol_rep, explanation)`` result.

        Re-registering an existing ``(molecule, explainer)`` pair
        **overwrites** it rather than raising, so re-running a notebook
        cell that builds the dashboard is idempotent instead of an error.

        Parameters
        ----------
        molecule : str
            Display name for the molecule (one axis of the selectors),
            e.g. ``"aspirin"``.
        mol_rep : MoleculeRepresentation
            The featurised molecule the explanation was computed against.
        explanation : Explanation
            The already-computed explanation to render.
        explainer : str, optional
            Display name for the explainer (the other selector axis), e.g.
            ``"TreeSHAP"``. Defaults to ``"default"``, so a collection
            using only one explainer needn't name it.

        Returns
        -------
        ExplanationDashboard
            ``self``, so ``add`` calls can be chained.

        Raises
        ------
        ValueError
            If ``molecule`` or ``explainer`` is empty.
        """
        if not molecule:
            raise ValueError("molecule name must be a non-empty string")
        if not explainer:
            raise ValueError("explainer name must be a non-empty string")

        self._results[(molecule, explainer)] = (mol_rep, explanation)
        # Invalidate any previously-built widget tree so the next display()
        # reflects the new result rather than a stale selector list.
        self._root = None
        return self

    @property
    def molecules(self) -> List[str]:
        """list[str]: Registered molecule names, in first-registered order."""
        seen: List[str] = []
        for molecule, _ in self._results:
            if molecule not in seen:
                seen.append(molecule)
        return seen

    @property
    def explainers(self) -> List[str]:
        """list[str]: Registered explainer names, in first-registered order."""
        seen: List[str] = []
        for _, explainer in self._results:
            if explainer not in seen:
                seen.append(explainer)
        return seen

    def get(self, molecule: str, explainer: str) -> Optional[_Result]:
        """Look up a registered result.

        Parameters
        ----------
        molecule : str
            The molecule name used at :meth:`add` time.
        explainer : str
            The explainer name used at :meth:`add` time.

        Returns
        -------
        tuple[MoleculeRepresentation, Explanation] or None
            The registered pair, or ``None`` if that combination was never
            registered.
        """
        return self._results.get((molecule, explainer))

    # -- rendering --------------------------------------------------------

    def render_svg(
        self,
        molecule: str,
        explainer: str,
        atom_range: Optional[Tuple[float, float]] = None,
        bond_range: Optional[Tuple[float, float]] = None,
    ) -> Optional[str]:
        """Render one registered result to an SVG string.

        The dashboard's whole interaction model routes through this method:
        every selector change or slider drag re-calls it and swaps the
        resulting string into a panel. It is a plain, widget-free function
        of the dashboard's state, so the rendering behavior is fully
        testable without a notebook frontend.

        Parameters
        ----------
        molecule : str
            The molecule name to render.
        explainer : str
            The explainer name to render.
        atom_range : tuple[float, float], optional
            Fixed atom color-scale range, passed straight through to
            :class:`~visxai.visualizers.rdkit_2d.RDKitSVGVisualizer`.
            ``None`` (default) uses that visualizer's per-call
            auto-computed range.
        bond_range : tuple[float, float], optional
            Fixed bond color-scale range, mirroring ``atom_range``.

        Returns
        -------
        str or None
            The rendered SVG string, or ``None`` if that
            ``(molecule, explainer)`` combination isn't registered.
        """
        result = self.get(molecule, explainer)
        if result is None:
            return None
        mol_rep, explanation = result
        return self._make_visualizer(atom_range, bond_range).visualize(mol_rep, explanation)

    def _make_visualizer(
        self,
        atom_range: Optional[Tuple[float, float]],
        bond_range: Optional[Tuple[float, float]],
        show_tooltips: Optional[bool] = None,
    ) -> RDKitSVGVisualizer:
        """Build the visualizer for the current color-scale state.

        Shared by both render paths so the static and hover-linked panels
        can never disagree about width, tooltips, or normalization range —
        the hover widget's SMILES strip colors are derived from these same
        ranges, so a divergence here would desynchronize the strip from the
        structure it sits under.

        Parameters
        ----------
        atom_range : tuple[float, float] or None
            Fixed atom normalization range, or ``None`` for per-call.
        bond_range : tuple[float, float] or None
            Fixed bond normalization range, mirroring ``atom_range``.
        show_tooltips : bool, optional
            Override this dashboard's ``show_tooltips``. Only the
            hover-linked panel passes ``False`` here: its widget supplies
            an instant readout, and a native ``<title>`` would pop the
            browser's slow tooltip on top of it. Left ``None`` everywhere
            else — notably :meth:`render_svg`, which is a standalone SVG
            export path and must keep honoring ``show_tooltips`` as the
            caller set it, regardless of how the panels happen to render.

        Returns
        -------
        RDKitSVGVisualizer
            Configured from this dashboard's settings.
        """
        return RDKitSVGVisualizer(
            width=self.width,
            height=self.height,
            show_tooltips=self.show_tooltips if show_tooltips is None else show_tooltips,
            legend_units=self.legend_units,
            atom_range=atom_range,
            bond_range=bond_range,
            **self.visualizer_kwargs,
        )

    # -- widget construction ----------------------------------------------

    def _build_panel(self, index: int) -> Dict[str, Any]:
        """Build one panel of the view.

        Parameters
        ----------
        index : int
            Which panel this is (``0``-based), used only to pick a sensible
            differing default selection so multiple panels don't all open
            showing the identical figure. With ``n_panels=1`` this is always
            ``0`` and simply selects the first registered result.

        Returns
        -------
        dict[str, Any]
            The panel's widgets, keyed ``"molecule"``, ``"explainer"``,
            ``"figure"``, ``"summary"``, and ``"box"`` (the container).
        """
        molecules = self.molecules
        explainers = self.explainers

        # Default to the index-th *registered* pair, in registration order,
        # clamped to the last one. Selecting the molecule and explainer axes
        # independently (what this used to do) can name a combination that
        # was never registered -- with molecule A explained only by X and
        # molecule B by both X and Y, panel 1 would open on A/Y and greet
        # the user with "No result registered". Walking the registry instead
        # makes an occupied cell structurally guaranteed, and still gives
        # the intended defaults: for a molecule registered under two
        # explainers the first two pairs differ by explainer (the common
        # "compare two explainers on one structure" case), while for a
        # single-explainer collection they differ by molecule.
        keys = list(self._results.keys())
        molecule_default, explainer_default = keys[min(index, len(keys) - 1)]

        molecule_dropdown = widgets.Dropdown(
            options=molecules,
            value=molecule_default,
            description="Molecule:",
            layout=widgets.Layout(width="auto"),
        )
        explainer_dropdown = widgets.Dropdown(
            options=explainers,
            value=explainer_default,
            description="Explainer:",
            layout=widgets.Layout(width="auto"),
        )
        # The hover-linked widget renders its own SVG from a payload; the
        # static fallback takes an SVG string. _set_panel_result is the one
        # place that knows which is which.
        figure: Any = MoleculeHoverWidget() if self._hover_link_active else widgets.HTML(value="")
        # A dedicated widget for the "no result" notice, so both figure
        # types behave identically -- MoleculeHoverWidget has no `.value`
        # to write a message into.
        message = widgets.HTML(value="")
        summary = widgets.HTML(value="")
        box = widgets.VBox(
            [molecule_dropdown, explainer_dropdown, message, figure, summary]
        )

        return {
            "molecule": molecule_dropdown,
            "explainer": explainer_dropdown,
            "figure": figure,
            "message": message,
            "summary": summary,
            "box": box,
        }

    def _build_controls(self) -> Dict[str, Any]:
        """Build the color-scale controls driving every panel.

        The mode selector has three states, because "shared vs per-panel"
        and "automatic vs manual" are two independent questions that an
        earlier single checkbox conflated:

        - ``"shared"`` (the default when ``legend_units="raw"``, or with a
          single panel) — one range pooled across the explanations
          **currently displayed**, recomputed whenever a selector changes.
          This is what makes panels comparable, and it needs no user input
          at all. Deliberately scoped to the displayed panels rather than
          the whole registry: pooling over everything registered lets a
          molecule you never look at flatten the ones you do (a molecule
          spanning ``±5`` drags two molecules spanning ``±0.1`` down to
          ``|t| < 0.03``, i.e. uniformly white).
        - ``"per-panel"`` (the default when ``legend_units="normalized"``,
          which is itself the default) — each panel normalizes against its
          own scores alone (``RDKitSVGVisualizer``'s own default behavior).
          Maximum contrast within a panel, but colors are not comparable
          *between* panels. Pairing normalized units with a *shared* range
          is degenerate — see the class docstring — which is why this, not
          ``"shared"``, is what a default dashboard opens on.
        - ``"manual"`` — the sliders take over, for deliberately clipping
          outliers so structure in the remaining scores becomes visible
          (the standard ``vmin``/``vmax`` need). This is the only mode that
          genuinely requires a control.

        With ``n_panels == 1`` the first two modes are indistinguishable
        (there is only one panel to pool over), so the selector collapses to
        ``"Automatic"`` vs ``"Manual"``.

        Returns
        -------
        dict[str, Any]
            Keyed ``"scale_mode"``, ``"atom_slider"``, ``"bond_slider"``,
            ``"box"``, and the two ``"atom_available"``/``"bond_available"``
            flags. A score type is *unavailable* when no registered
            explanation has any score of that type at all (e.g. bonds for
            an atom-only collection); its slider is then permanently
            disabled. This is tracked as its own flag rather than read back
            off the slider's ``disabled`` attribute, because that attribute
            is also toggled whenever the mode changes — conflating the two
            left a slider stuck disabled after one mode round-trip.
        """
        explanations = [explanation for _, explanation in self._results.values()]
        atom_range = _score_range(explanations, "atom")
        bond_range = _score_range(explanations, "bond")

        # Labels name what each mode *does*, rather than leading two of the
        # three with the same uninformative word "Automatic" — which buried
        # the only choice that matters (compare across panels, or maximize
        # contrast within each?). Manual says what it is for, since its
        # purpose is the least guessable of the three: clipping an outlier
        # that is flattening everything else, or pinning a range to reuse
        # for a static figure.
        manual_label = "Manual range — clip an outlier, or pin a scale to reuse"
        if self.n_panels == 1:
            options = [("Automatic range", "shared"), (manual_label, "manual")]
        else:
            options = [
                ("Shared range — compare across panels", "shared"),
                ("Per panel — best contrast in each", "per-panel"),
                (manual_label, "manual"),
            ]
        # The default mode follows `legend_units`, because the two are not
        # independent: "shared" + "normalized" is a degenerate pairing.
        # Under one pooled range both panels apply the SAME mapping, so a
        # normalized number is just the raw score put through a transform
        # common to both -- it carries nothing the raw score doesn't, while
        # silently making each panel's colors depend on the *other* panel's
        # scores. Worse, the normalized legend cannot show it: it prints the
        # output of the division, never the divisor, so the ticks read
        # -1.00/0/+1.00 no matter what range is in force. Changing one
        # panel's explainer therefore recolored the other with no on-screen
        # explanation -- reported as a bug, and reasonably so.
        #
        # Normalized only means something per-panel ("relative importance
        # within THIS molecule"); shared magnitudes are what raw units are
        # for. With one panel there is nothing to pool, so the distinction
        # collapses and "shared" is the only automatic option.
        default_mode = (
            "shared"
            if self.n_panels == 1 or self.legend_units == "raw"
            else "per-panel"
        )
        scale_mode = widgets.RadioButtons(
            options=options,
            value=default_mode,
            description="Color scale:",
            layout=widgets.Layout(width="auto"),
        )
        # What the legend's min/max ticks print. The hover readout always
        # shows both the raw score and the normalized value, so this only
        # picks which one the color bar is labelled in -- but the two are
        # different numbers, and reading a raw score off a normalized bar
        # (or vice versa) is exactly the confusion this exposes a switch for.
        legend_units = widgets.ToggleButtons(
            options=[("Normalized (-1..+1)", "normalized"), ("Raw scores", "raw")],
            value=self.legend_units,
            description="Legend:",
            layout=widgets.Layout(width="auto"),
        )
        atom_slider = self._build_range_slider("Atom range:", atom_range)
        bond_slider = self._build_range_slider("Bond range:", bond_range)
        box = widgets.VBox([scale_mode, legend_units, atom_slider, bond_slider])

        return {
            "scale_mode": scale_mode,
            "legend_units": legend_units,
            "atom_slider": atom_slider,
            "bond_slider": bond_slider,
            "atom_available": atom_range is not None,
            "bond_available": bond_range is not None,
            "box": box,
        }

    def _build_range_slider(
        self, description: str, score_range: Optional[Tuple[float, float]]
    ) -> Any:
        """Build one ``FloatRangeSlider``, bounded by the pooled score range.

        Parameters
        ----------
        description : str
            The slider's label.
        score_range : tuple[float, float] or None
            The pooled observed range this slider covers. ``None`` (no
            registered explanation has this score type at all) yields a
            disabled placeholder slider rather than a missing control, so
            the layout stays stable.

        Returns
        -------
        ipywidgets.FloatRangeSlider
            The configured slider.
        """
        if score_range is None:
            return widgets.FloatRangeSlider(
                value=[0.0, 1.0], min=0.0, max=1.0, step=0.01,
                description=description, disabled=True, readout_format=".3f",
                layout=widgets.Layout(width="auto"),
            )

        low, high = _pad_slider_bounds(score_range)
        return widgets.FloatRangeSlider(
            value=[score_range[0], score_range[1]],
            min=low,
            max=high,
            step=(high - low) / 200.0,
            description=description,
            readout_format=".3f",
            continuous_update=False,
            layout=widgets.Layout(width="auto"),
        )

    def _displayed_explanations(self) -> List[Explanation]:
        """Collect the explanations currently selected across all panels.

        Skips panels whose ``(molecule, explainer)`` selection isn't
        registered, so an unfilled combination contributes nothing to the
        pooled range rather than erroring.

        Returns
        -------
        list[Explanation]
            One entry per panel with a registered selection.
        """
        displayed: List[Explanation] = []
        for panel in self._panels:
            result = self.get(panel["molecule"].value, panel["explainer"].value)
            if result is not None:
                displayed.append(result[1])
        return displayed

    def _refresh(self) -> None:
        """Re-render every panel from the current widget state.

        Called once at build time and again on every selector, mode, or
        slider change. Each call regenerates a complete new SVG string per
        panel (RDKit has no incremental redraw) and swaps it into that
        panel's ``HTML`` widget.

        In the two automatic modes this also writes the computed range back
        onto the sliders, so switching to ``"manual"`` starts from wherever
        automatic left off instead of jumping. That write would re-trigger
        this method through the sliders' own observer, hence the
        ``_refreshing`` re-entrancy guard.
        """
        if self._refreshing:
            return
        self._refreshing = True
        try:
            controls = self._controls
            mode = controls["scale_mode"].value
            # Mirror the control back onto the instance so render_svg() and
            # _make_visualizer() have one source of truth either way.
            self.legend_units = controls["legend_units"].value

            if mode == "manual":
                # A score type absent from every registered explanation has
                # no range to fix, so it stays on the visualizer's default.
                atom_range = (
                    tuple(controls["atom_slider"].value) if controls["atom_available"] else None
                )
                bond_range = (
                    tuple(controls["bond_slider"].value) if controls["bond_available"] else None
                )
            elif mode == "shared":
                # Pool over what's ON SCREEN, not the whole registry, so an
                # unviewed outlier can't flatten the panels being compared.
                displayed = self._displayed_explanations()
                atom_range = _score_range(displayed, "atom")
                bond_range = _score_range(displayed, "bond")
            else:  # "per-panel" -- let the visualizer recompute per call
                atom_range = bond_range = None

            # Sliders are live only in manual mode, and only for a score type
            # that actually exists. Derived fresh from the availability flag
            # each time rather than from the slider's own previous `disabled`
            # state, so leaving and re-entering manual mode restores an
            # available slider instead of leaving it stuck disabled.
            atom_on = mode == "manual" and controls["atom_available"]
            bond_on = mode == "manual" and controls["bond_available"]
            controls["atom_slider"].disabled = not atom_on
            controls["bond_slider"].disabled = not bond_on

            # Hide the sliders outside manual mode rather than showing them
            # greyed out. They are the only control the other two modes don't
            # need, and a permanently-visible pair of dead sliders reads as
            # broken UI. `layout.display` is used rather than rebuilding the
            # box's children so the widgets keep their identity -- observers
            # stay attached, and the range written below is still there when
            # manual mode reveals them again.
            slider_display = "" if mode == "manual" else "none"
            controls["atom_slider"].layout.display = slider_display
            controls["bond_slider"].layout.display = slider_display

            # Reflect an automatic range on the (disabled) sliders so the
            # numbers on screen always describe the rendering in front of
            # the user, and so manual mode inherits them.
            if mode != "manual":
                self._sync_slider(controls["atom_slider"], atom_range)
                self._sync_slider(controls["bond_slider"], bond_range)

            for panel in self._panels:
                self._set_panel_result(panel, atom_range, bond_range)
        finally:
            self._refreshing = False

    def _set_panel_result(
        self,
        panel: Dict[str, Any],
        atom_range: Optional[Tuple[float, float]],
        bond_range: Optional[Tuple[float, float]],
    ) -> None:
        """Render one panel's current selection, whichever figure type it has.

        The single place that knows the difference between the two figure
        widgets: :class:`~visxai.visualizers.hover_widget.MoleculeHoverWidget`
        takes a molecule and explanation and builds its own payload (SVG plus
        the SMILES strip), while the static fallback takes a finished SVG
        string. Keeping the branch here means :meth:`_refresh` and every
        caller stay identical across both modes.

        Parameters
        ----------
        panel : dict[str, Any]
            The panel to update, as built by :meth:`_build_panel`.
        atom_range : tuple[float, float] or None
            Fixed atom normalization range for this render.
        bond_range : tuple[float, float] or None
            Fixed bond normalization range, mirroring ``atom_range``.
        """
        molecule = panel["molecule"].value
        explainer = panel["explainer"].value
        result = self.get(molecule, explainer)

        if result is None:
            # Hide the figure rather than blanking it: a hover widget has no
            # `.value` to clear, and leaving a stale molecule on screen under
            # a "no result" notice would be worse than showing nothing.
            panel["message"].value = _MISSING_RESULT_HTML
            panel["figure"].layout.display = "none"
            panel["summary"].value = ""
            return

        mol_rep, explanation = result
        panel["message"].value = ""
        panel["figure"].layout.display = ""

        if self._hover_link_active:
            panel["figure"].update_explanation(
                mol_rep,
                explanation,
                visualizer=self._make_visualizer(atom_range, bond_range, show_tooltips=False),
                atom_range=atom_range,
                bond_range=bond_range,
            )
        else:
            panel["figure"].value = self._make_visualizer(
                atom_range, bond_range
            ).visualize(mol_rep, explanation)

        panel["summary"].value = _summary_html(explanation)

    @staticmethod
    def _sync_slider(slider: Any, score_range: Optional[Tuple[float, float]]) -> None:
        """Write an automatically-computed range onto a slider, clamped to its bounds.

        The slider's ``min``/``max`` come from the *whole registry* (so
        manual mode can express any range), while ``score_range`` here comes
        from the displayed subset — always within those bounds in practice,
        but clamped defensively since ipywidgets raises rather than clips.

        Parameters
        ----------
        slider : ipywidgets.FloatRangeSlider
            The slider to update.
        score_range : tuple[float, float] or None
            The range to display. ``None`` (that score type isn't present)
            leaves the slider untouched.
        """
        if score_range is None:
            return
        low = max(slider.min, min(slider.max, score_range[0]))
        high = max(slider.min, min(slider.max, score_range[1]))
        if low <= high:
            slider.value = (low, high)

    def widget(self) -> Any:
        """Build (once) and return the dashboard's root widget container.

        Returns
        -------
        ipywidgets.VBox
            The root container: the color-scale controls above a row of
            ``self.n_panels`` panels. Rebuilt automatically after any
            :meth:`add`, so the selectors always reflect the current
            registry.

        Raises
        ------
        ValueError
            If no results have been registered yet.
        """
        if not self._results:
            raise ValueError(
                "No results registered -- call add(molecule, mol_rep, explanation) "
                "at least once before displaying the dashboard."
            )
        if self._root is not None:
            return self._root

        self._controls = self._build_controls()
        self._panels = [self._build_panel(i) for i in range(self.n_panels)]

        def _on_change(_change: Any) -> None:
            self._refresh()

        for control_name in ("scale_mode", "legend_units", "atom_slider", "bond_slider"):
            self._controls[control_name].observe(_on_change, names="value")
        for panel in self._panels:
            panel["molecule"].observe(_on_change, names="value")
            panel["explainer"].observe(_on_change, names="value")

        self._root = widgets.VBox(
            [
                self._controls["box"],
                widgets.HBox([panel["box"] for panel in self._panels]),
            ]
        )
        self._refresh()
        return self._root

    def display(self) -> None:
        """Build the dashboard and display it in the current notebook cell.

        Thin wrapper over :meth:`widget` plus ``IPython.display.display``,
        so the common notebook call is a single line.

        Raises
        ------
        ValueError
            If no results have been registered yet.
        """
        from IPython.display import display as ipython_display

        ipython_display(self.widget())
