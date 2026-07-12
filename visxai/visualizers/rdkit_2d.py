"""2-D SVG molecule visualizer using RDKit's native rdMolDraw2D canvas.

Renders XAI atom- and bond-level scores as a red-to-green color gradient
directly onto a 2-D molecular structure.  All drawing is performed by
``rdkit.Chem.Draw.rdMolDraw2D.MolDraw2DSVG``; **matplotlib is not used**.

A color-bar legend (min/zero/max score tick labels alongside the actual
gradient) is composed onto the same SVG by default, so the mapping from
color to score is visible on the figure itself rather than only described
in prose elsewhere. The legend is plain hand-built SVG (``<linearGradient>``
+ ``<rect>`` + ``<text>``, no matplotlib) string-composed next to RDKit's
own molecule SVG — RDKit's own drawing primitives (``DrawRect`` etc.) were
tried first, but their coordinates go through the molecule's internal
scale/offset transform rather than raw pixels, which isn't a reliable way
to draw a fixed-geometry sidebar.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from rdkit.Chem import Mol
from rdkit.Chem.Draw import rdMolDraw2D

from visxai.core.base_visualizer import BaseVisualizer
from visxai.core.data_types import Explanation, MoleculeRepresentation


# Type alias used by rdMolDraw2D color arguments: RGB triple in [0, 1].
_RGB = Tuple[float, float, float]
_ColorMap = Dict[int, _RGB]


def _score_to_t(score: float, min_score: float, max_score: float) -> Optional[float]:
    """Compute the signed relative-magnitude ``t`` a score maps to, in ``[-1, 1]``.

    This is the same value :func:`_score_to_rgb` uses to pick a color,
    factored out so :func:`_build_legend_svg`'s ``legend_units="normalized"``
    tick labels are provably the exact number driving the color, not a
    re-derivation that could silently drift from it. Two different formulas
    are used depending on whether ``[min_score, max_score]`` straddles zero:

    - **Straddles zero** (``min_score < 0 < max_score``, a genuinely signed
      score set): diverging, zero-anchored. Each side of zero is scaled
      **independently** against its own actual extreme —
      ``t = score / max_score`` for ``score >= 0``, ``t = score /
      abs(min_score)`` for ``score < 0`` — so the larger-magnitude side
      never determines the smaller side's span. This fixes a real bug:
      under a single shared ``span = max(abs(min_score), abs(max_score))``
      (this function's original formula), a value that is *the maximum of
      its own sign* could still fail to reach the full color extreme
      purely because the opposite sign happened to have a bigger swing —
      e.g. IG's aspirin atom scores (``-1.65..+0.49``): the one positive
      atom is the largest positive value that exists, yet the old formula
      rendered it at only ``t=0.30`` (pale green) because the negative
      side's ``1.65`` set the shared span. It should read ``t=1.00`` (full
      green) — it *is* the maximum — and now does.
    - **Does not straddle zero** (all scores share one sign, or one bound
      is exactly ``0.0``): sequential, a plain min→max stretch across the
      *actual* observed range, not down to an unvisited absolute zero.
      Forcing a zero anchor when the data never comes near zero wastes
      most of the color range on a region with no data at all — e.g. IG's
      aspirin bond scores (``1.25..2.51``, all positive, none anywhere
      near ``0``): the old zero-anchored formula put every bond's ``t`` in
      ``[0.50, 1.0]``, a narrow slice of already-fairly-saturated green
      that made every bond look nearly identical. The sequential stretch
      instead maps the *actual* minimum to white (``t=0``) and the actual
      maximum to full green (``t=1``), using the whole available range —
      and mirrors to red (``t`` in ``[-1, 0]``) when every score is
      non-positive instead. The legend's raw-value ticks (or the
      min/max score printed alongside) still show the true, non-zero
      minimum, so a reader is never told the palest bond has "no effect".

    Parameters
    ----------
    score : float
        The score to map.
    min_score : float
        The minimum score in the dataset.
    max_score : float
        The maximum score in the dataset.

    Returns
    -------
    float or None
        The signed relative magnitude in ``[-1.0, 1.0]``, or ``None`` when
        ``min_score == max_score`` (including both being ``0.0``) — the
        degenerate case with no meaningful direction to normalize against.
    """
    if max_score == min_score:
        return None
    if min_score < 0.0 < max_score:
        span = max_score if score >= 0.0 else -min_score
        return max(-1.0, min(1.0, score / span))
    if min_score >= 0.0:
        t = (score - min_score) / (max_score - min_score)
        return max(0.0, min(1.0, t))
    t = (score - min_score) / (max_score - min_score) - 1.0
    return max(-1.0, min(0.0, t))


def _score_to_rgb(score: float, min_score: float, max_score: float) -> _RGB:
    """Map a scalar score to an RGB colour on a red–white–green gradient.

    Delegates to :func:`_score_to_t` for the actual score → magnitude
    mapping (see that function's docstring for the diverging-vs-sequential
    distinction), then converts the signed ``t`` to a color: ``t=0`` is
    always white, ``t>0`` interpolates toward pure green at ``t=1``,
    ``t<0`` interpolates toward pure red at ``t=-1``.

    Parameters
    ----------
    score : float
        The score to map.
    min_score : float
        The minimum score in the dataset. Maps to pure red ``(1, 0, 0)``
        only when ``[min_score, max_score]`` straddles zero (a genuinely
        signed score set) — otherwise (all one sign) it maps to white,
        since :func:`_score_to_t` uses a sequential, not diverging, scale
        in that case (see its docstring).
    max_score : float
        The maximum score in the dataset. Maps to pure green ``(0, 1, 0)``
        when ``score >= 0`` was ever possible; maps to white when every
        score (including this one) is non-positive.

    Returns
    -------
    tuple[float, float, float]
        An ``(R, G, B)`` triple with values in ``[0.0, 1.0]``.

    Notes
    -----
    When ``min_score == max_score`` (all scores identical), every atom is
    mapped to white ``(1, 1, 1)`` to avoid division by zero.
    """
    t = _score_to_t(score, min_score, max_score)
    if t is None:
        return (1.0, 1.0, 1.0)

    if t >= 0.0:
        # White (1,1,1) → Green (0,1,0)
        r = 1.0 - t
        g = 1.0
        b = 1.0 - t
    else:
        # Red (1,0,0) → White (1,1,1)
        t_abs = -t
        r = 1.0
        g = 1.0 - t_abs
        b = 1.0 - t_abs

    return (r, g, b)


# Highlight-circle radius bounds (RDKit mol-coordinate units; its own
# default highlightRadius is 0.3) used by _atom_radius below. A near-zero
# score still gets a small, visible circle rather than shrinking to
# nothing — it's already rendered white by _score_to_rgb, so the radius
# floor mostly matters for legibility of the circle's outline.
_MIN_ATOM_RADIUS = 0.2
_MAX_ATOM_RADIUS = 0.5


def _atom_radius(score: float, min_score: float, max_score: float) -> float:
    """Map a score to a highlight-circle radius, linear in ``|t|``.

    A second visual channel alongside color: the atom whose score is
    largest in magnitude (relative to the rest of its own score
    population, via :func:`_score_to_t`) gets the biggest highlight
    circle, regardless of whether the color gradient itself has enough
    contrast to make that magnitude difference obvious on its own.

    Parameters
    ----------
    score : float
        The score to map.
    min_score : float
        The minimum score in the dataset.
    max_score : float
        The maximum score in the dataset.

    Returns
    -------
    float
        A radius in ``[_MIN_ATOM_RADIUS, _MAX_ATOM_RADIUS]``. Degenerate
        ranges (``_score_to_t`` returns ``None``) get ``_MIN_ATOM_RADIUS``.
    """
    t = _score_to_t(score, min_score, max_score)
    magnitude = abs(t) if t is not None else 0.0
    return _MIN_ATOM_RADIUS + (_MAX_ATOM_RADIUS - _MIN_ATOM_RADIUS) * magnitude


def _rgb_to_hex(rgb: _RGB) -> str:
    """Format an ``(R, G, B)`` triple in ``[0, 1]`` as an ``#RRGGBB`` hex string."""
    r, g, b = (round(channel * 255) for channel in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def _gradient_stops(min_score: float, max_score: float) -> List[Tuple[float, _RGB]]:
    """Compute correctly-positioned color-gradient stops for a score range.

    Stop position ``0.0`` corresponds to ``min_score`` and ``1.0`` to
    ``max_score``. Unlike a naive fixed 0%/50%/100% split, the position of
    the "zero" stop (when the range straddles zero) reflects where zero
    actually falls in ``[min_score, max_score]`` — e.g. a ``-0.3..1.0``
    range puts white at position ``0.3 / 1.3 ≈ 0.23``, not the midpoint.

    Parameters
    ----------
    min_score : float
        The minimum score in the range being visualized.
    max_score : float
        The maximum score in the range being visualized.

    Returns
    -------
    list[tuple[float, tuple[float, float, float]]]
        ``(position, rgb)`` pairs, sorted by ascending position. A single
        white stop when ``min_score == max_score`` (mirrors
        :func:`_score_to_rgb`'s own divide-by-zero guard).
    """
    if max_score == min_score:
        return [(0.0, (1.0, 1.0, 1.0))]

    stops: List[Tuple[float, _RGB]] = [
        (0.0, _score_to_rgb(min_score, min_score, max_score))
    ]
    if min_score < 0.0 < max_score:
        zero_position = (0.0 - min_score) / (max_score - min_score)
        stops.append((zero_position, _score_to_rgb(0.0, min_score, max_score)))
    stops.append((1.0, _score_to_rgb(max_score, min_score, max_score)))
    return stops


def _build_legend_svg(
    min_score: float,
    max_score: float,
    width: int,
    height: int,
    title: Optional[str] = None,
    gradient_id: str = "visxaiLegendGradient",
    legend_units: Literal["raw", "normalized"] = "raw",
) -> str:
    """Build a standalone SVG fragment for a vertical color-bar legend.

    Pure string construction — no RDKit calls — producing a gradient-filled
    (or, in the degenerate ``min_score == max_score`` case, solid-filled)
    ``<rect>`` plus tick-label ``<text>`` elements for the max, zero (if in
    range), and min scores. Oriented with the maximum score at the top and
    the minimum at the bottom, the standard color-bar convention. Tick
    labels are vertically aligned with :func:`_gradient_stops`' positions,
    so a label lines up with where its color actually appears on the bar.

    Parameters
    ----------
    min_score : float
        The minimum score in the range being visualized.
    max_score : float
        The maximum score in the range being visualized.
    width : int
        Width in pixels available for the legend (bar + tick labels).
    height : int
        Height in pixels available for the legend; matches the molecule
        canvas height so the two compose into one rectangular figure, or
        half of it when stacked by :func:`_build_dual_legend_svg`.
    title : str, optional
        A short bold label drawn above the bar (e.g. ``"Atoms"``/``"Bonds"``)
        to distinguish it when two legends are stacked. ``None`` (default)
        draws no title, producing byte-identical output to before this
        parameter existed.
    gradient_id : str, optional
        The SVG ``id`` used for the ``<linearGradient>`` element. Must be
        unique within the final composed document — :func:`_build_dual_legend_svg`
        passes distinct ids for its two bars so both gradients render
        correctly side by side. Defaults to ``"visxaiLegendGradient"``, the
        pre-existing single-legend id.
    legend_units : {"raw", "normalized"}, optional
        Whether tick labels print the raw score (``"raw"``, the default —
        byte-identical to this function's output before this parameter
        existed) or the value ``t`` from :func:`_score_to_t`
        (``"normalized"`` — the exact same value that already determines
        the tick's color; ``[-1, 1]`` when the range straddles zero, or
        ``[0, 1]``/``[-1, 0]`` when it's entirely one sign — see that
        function's docstring for the full diverging-vs-sequential split).
        For a straddling range like ``-1.65..+0.49``, both the min and max
        ticks reach the full extreme (``-1.00``/``+1.00``) since each side
        is scaled independently. For a same-signed range like
        ``1.25..2.51`` (all positive), the min tick reads ``0.00`` (not
        the misleading ``0.50`` a shared-zero-anchored scale would give)
        and the max tick reads ``1.00``.

    Returns
    -------
    str
        An SVG fragment (``<defs>``/``<rect>``/``<text>`` elements, no
        wrapping ``<svg>`` tag) suitable for embedding inside a ``<g>``.
    """
    stops = _gradient_stops(min_score, max_score)

    margin = 20
    title_offset = 16 if title else 0
    bar_x = 8
    bar_width = 18
    bar_top = margin + title_offset
    bar_bottom = max(height - margin, bar_top + 1)
    bar_height = bar_bottom - bar_top
    text_x = bar_x + bar_width + 4

    if len(stops) == 1:
        # Degenerate range: nothing to gradient, just a solid swatch.
        _, color = stops[0]
        bar_fill = f"fill=\"{_rgb_to_hex(color)}\""
        defs = ""
    else:
        stop_elements = "\n".join(
            f'    <stop offset="{position * 100:.4f}%" stop-color="{_rgb_to_hex(color)}"/>'
            for position, color in stops
        )
        defs = (
            f'<defs>\n  <linearGradient id="{gradient_id}" x1="0" y1="1" x2="0" y2="0">\n'
            f"{stop_elements}\n  </linearGradient>\n</defs>\n"
        )
        bar_fill = f'fill="url(#{gradient_id})"'

    title_element = (
        f'<text x="{bar_x}" y="{margin - 6}" font-size="11" font-weight="bold" '
        f'font-family="sans-serif">{title}</text>\n'
        if title
        else ""
    )

    bar_rect = (
        f'<rect x="{bar_x}" y="{bar_top}" width="{bar_width}" height="{bar_height}" '
        f'{bar_fill} stroke="black" stroke-width="1"/>'
    )

    labels: List[str] = []
    for position, _ in stops:
        score = min_score + position * (max_score - min_score)
        if legend_units == "normalized":
            # len(stops) == 1 only when min_score == max_score, the same
            # degenerate case _score_to_t returns None for — print 0.00 to
            # match _score_to_rgb's "identical score -> neutral white"
            # semantics, rather than an undefined score/span.
            label_value = 0.0 if len(stops) == 1 else _score_to_t(score, min_score, max_score)
        else:
            label_value = score
        y = bar_bottom - position * bar_height
        labels.append(
            f'<text x="{text_x}" y="{y + 4:.1f}" font-size="11" '
            f'font-family="sans-serif">{label_value:.2f}</text>'
        )

    return title_element + defs + bar_rect + "\n" + "\n".join(labels)


def _build_dual_legend_svg(
    atom_min: float,
    atom_max: float,
    bond_min: float,
    bond_max: float,
    width: int,
    height: int,
    legend_units: Literal["raw", "normalized"] = "raw",
) -> str:
    """Stack an "Atoms" color-bar legend above a "Bonds" one.

    Used when atom and bond scores are normalized on independent scales
    (``RDKitSVGVisualizer(color_scale="separate")``, the default) and both
    are present — a single shared bar would otherwise mislabel one
    element type's colors with the other's tick values. Each half reuses
    :func:`_build_legend_svg` with its own ``gradient_id`` (SVG ids must be
    unique within one document) and a title so the two are distinguishable.

    Parameters
    ----------
    atom_min, atom_max : float
        The score range for the atom legend (top half).
    bond_min, bond_max : float
        The score range for the bond legend (bottom half).
    width : int
        Width in pixels available for each legend (bar + tick labels).
    height : int
        Total height in pixels available; split evenly between the two
        stacked legends.
    legend_units : {"raw", "normalized"}, optional
        Passed through unchanged to both nested :func:`_build_legend_svg`
        calls — each half stays normalized against its own independent
        atom/bond range (see that function's docstring). Defaults to
        ``"raw"``.

    Returns
    -------
    str
        An SVG fragment (two ``<g transform="translate(...)">``-wrapped
        legends) suitable for embedding inside a ``<g>``.
    """
    half_height = height // 2
    atom_svg = _build_legend_svg(
        atom_min, atom_max, width, half_height, title="Atoms",
        gradient_id="visxaiLegendGradientAtoms", legend_units=legend_units,
    )
    bond_svg = _build_legend_svg(
        bond_min, bond_max, width, half_height, title="Bonds",
        gradient_id="visxaiLegendGradientBonds", legend_units=legend_units,
    )
    return (
        f'<g transform="translate(0,0)">{atom_svg}</g>'
        f'<g transform="translate(0,{half_height})">{bond_svg}</g>'
    )


def _extract_svg_body(svg_text: str) -> str:
    """Return the inner content of an SVG string, between ``<svg ...>`` and ``</svg>``.

    Locates the boundary structurally (the first ``>`` that closes the root
    ``<svg`` opening tag, and the last ``</svg>``) rather than by matching
    RDKit's ``<!-- END OF HEADER -->`` comment literally, so this isn't
    tied to one specific RDKit version's exact SVG template.

    Parameters
    ----------
    svg_text : str
        A complete, self-contained SVG string.

    Returns
    -------
    str
        Everything between the root ``<svg>`` tag's ``>`` and the final
        ``</svg>``, stripped of leading/trailing whitespace.
    """
    open_tag_start = svg_text.index("<svg")
    open_tag_end = svg_text.index(">", open_tag_start) + 1
    close_tag_start = svg_text.rindex("</svg>")
    return svg_text[open_tag_end:close_tag_start].strip()


def _compose_svg_with_legend(
    mol_svg: str,
    legend_svg: str,
    mol_width: int,
    mol_height: int,
    legend_width: int,
) -> str:
    """Compose a molecule SVG and a legend SVG fragment into one SVG string.

    Wraps each half in its own ``<g transform="translate(...)">`` inside a
    new outer ``<svg>`` whose width is ``mol_width + legend_width`` — the
    molecule occupies ``x in [0, mol_width)``, the legend occupies
    ``x in [mol_width, mol_width + legend_width)``.

    Parameters
    ----------
    mol_svg : str
        The complete SVG string produced by ``rdMolDraw2D`` for the
        molecule alone.
    legend_svg : str
        The legend fragment from :func:`_build_legend_svg`.
    mol_width : int
        Width in pixels of the molecule canvas.
    mol_height : int
        Height in pixels of the molecule canvas (and the legend).
    legend_width : int
        Width in pixels reserved for the legend.

    Returns
    -------
    str
        One complete, self-contained SVG string containing both halves.
    """
    mol_body = _extract_svg_body(mol_svg)
    total_width = mol_width + legend_width
    return (
        "<svg xmlns='http://www.w3.org/2000/svg' "
        "xmlns:xlink='http://www.w3.org/1999/xlink' "
        f"width='{total_width}px' height='{mol_height}px' "
        f"viewBox='0 0 {total_width} {mol_height}'>"
        "<rect width='100%' height='100%' fill='white'/>"
        f"<g transform='translate(0,0)'>{mol_body}</g>"
        f"<g transform='translate({mol_width},0)'>{legend_svg}</g>"
        "</svg>"
    )


class RDKitSVGVisualizer(BaseVisualizer):
    """Render per-atom and per-bond XAI scores as a 2-D SVG molecule image.

    Uses ``rdkit.Chem.Draw.rdMolDraw2D.MolDraw2DSVG`` exclusively for the
    molecule itself.  Atom scores are mapped to a **red–white–green**
    colour gradient (negative → red, zero → white, positive → green) and
    passed directly to ``DrawMolecule`` via the ``highlightAtoms`` /
    ``highlightAtomColors`` arguments.  Bond scores follow the same mapping
    when present in the :class:`~visxai.core.data_types.Explanation`.

    A color-bar legend is composed alongside the molecule by default
    (``show_legend=True``) so the color-to-score mapping is visible on the
    figure itself. It's plain hand-built SVG (not RDKit-drawn, not
    matplotlib) string-composed next to RDKit's own molecule SVG — see the
    module docstring for why. Pass ``show_legend=False`` to get the exact
    same output as before this feature existed.

    **Atom and bond scores are normalized on independent scales by
    default** (``color_scale="separate"``). Atom scores and bond scores
    routinely come from different underlying distributions — e.g.
    ``IntegratedGradientsExplainer`` on a model that concatenates
    atom/edge features before every nonlinear layer tends to push most of
    the signed magnitude onto bonds, leaving atom scores in a much smaller
    range — so normalizing both against one shared min/max lets the
    larger-magnitude side (usually bonds) wash out the smaller side's
    contrast almost to invisibility, even though the smaller side may
    contain the more locally-informative signal. ``"separate"`` gives
    atoms and bonds their own min/max (and, when both are present, their
    own stacked "Atoms"/"Bonds" legend via :func:`_build_dual_legend_svg`)
    so each element type's own internal contrast is preserved. Pass
    ``color_scale="combined"`` to pool atom and bond scores into one
    shared min/max instead (one legend, one gradient for both). When only
    one of atom/bond scores is present, the two modes are identical —
    there's only one distribution to normalize against either way.

    **The color mapping itself is sign-aware** (see :func:`_score_to_t`
    for the full derivation) — this applies identically regardless of
    ``color_scale``. When a score population straddles zero (has both
    positive and negative values), the mapping stays diverging and
    zero-anchored, but each side is scaled to its *own* actual extreme —
    so a value that's the largest of its own sign always reaches the full
    color, never capped just because the opposite sign happens to swing
    further (this was a real bug in an earlier version: IG's aspirin atom
    scores span ``-1.65..+0.49``, and the one positive atom — genuinely
    the largest positive value present — used to render at only ``t=0.30``
    because the *negative* side's larger ``1.65`` set a shared span for
    both sides; it now correctly reaches ``t=1.00``). When a score
    population is entirely one sign (e.g. Grad-CAM, or IG bond scores that
    happen to all be positive), the mapping switches to sequential: a
    plain min→max stretch across the range that's actually present,
    rather than an unused zero anchor that wastes most of the color range
    on a region with no data (aspirin's IG bond scores, ``1.25..2.51``,
    used to all render within ``t ∈ [0.50, 1.0]`` — a narrow, already
    fairly-saturated slice that made every bond look nearly identical;
    they now correctly stretch across the full ``t ∈ [0.0, 1.0]``).

    **Legend tick labels print normalized values by default**
    (``legend_units="normalized"``). The number next to each tick is the
    exact same value ``t`` (see :func:`_score_to_t`) that already
    determines that tick's color, so a reader never has to reconcile a
    raw-unit number against a color computed on a different scale. Pass
    ``legend_units="raw"`` to print the raw score instead — this is
    **purely a display change**, composing with ``color_scale``
    independently of the sign-aware mapping above: neither setting
    touches ``explanation.atom_scores``/``bond_scores`` themselves, so
    e.g. ``IntegratedGradientsExplainer``'s completeness axiom (verified
    by ``tests/test_gradient_based.py``, which doesn't import this module
    at all) is unaffected regardless of either setting.

    **Highlighted atom-circle radius scales with score magnitude by
    default** (``scale_atom_radius=True``). A second visual channel
    alongside color — the atom with the largest-magnitude score (relative
    to its own score population, same ``t`` as the color) gets the
    biggest highlight circle, so a magnitude difference remains visible
    even where color contrast alone is subtle (e.g. two same-signed atoms
    whose colors are both fairly saturated but whose exact ``t`` differs).
    Radius is linear in ``|t|`` between ``_MIN_ATOM_RADIUS`` (0.2) and
    ``_MAX_ATOM_RADIUS`` (0.5) — RDKit's own default ``highlightRadius``
    is 0.3. Pass ``scale_atom_radius=False`` for a uniform radius on every
    highlighted atom (RDKit's own default), matching this visualizer's
    output before this parameter existed. Bond-highlight *width* is not
    similarly scaled: RDKit only varies highlight-band width per bond
    when using its unfilled "line" highlight style (``continuousHighlight
    =False``), which also switches every atom highlight from a filled
    circle to an unfilled ring — a much bigger rendering-style change than
    this parameter's scope, deferred rather than folded in silently.

    Parameters
    ----------
    width : int, optional
        Molecule canvas width in pixels.  Defaults to ``400``.
    height : int, optional
        Molecule canvas height in pixels.  Defaults to ``300``.
    show_legend : bool, optional
        Whether to compose a color-bar legend alongside the molecule.
        Defaults to ``True``. Has no effect if the explanation has no
        atom or bond scores at all.
    legend_width : int, optional
        Width in pixels reserved for the legend, added to ``width`` in the
        final rendered SVG. Defaults to ``80``. Unused if ``show_legend``
        is ``False``.
    color_scale : {"separate", "combined"}, optional
        Whether atom and bond scores are normalized independently
        (``"separate"``, the default) or against one shared min/max
        (``"combined"``). See above.
    legend_units : {"raw", "normalized"}, optional
        Whether legend tick labels print the raw score or the value ``t``
        that already drives the color.  Defaults to ``"normalized"``. See
        above.
    scale_atom_radius : bool, optional
        Whether highlighted atom-circle radius scales with ``|t|``.
        Defaults to ``True``. See above.

    Attributes
    ----------
    width : int
        SVG molecule-canvas width (unaffected by the legend).
    height : int
        SVG molecule-canvas height.

    Examples
    --------
    >>> viz = RDKitSVGVisualizer(width=600, height=400)
    >>> svg_str = viz.visualize(mol_rep, explanation)
    >>> open("molecule.svg", "w").write(svg_str)

    >>> viz_no_legend = RDKitSVGVisualizer(show_legend=False)
    >>> svg_str = viz_no_legend.visualize(mol_rep, explanation)

    >>> viz_combined_scale = RDKitSVGVisualizer(color_scale="combined")
    >>> svg_str = viz_combined_scale.visualize(mol_rep, explanation)

    >>> viz_raw_units = RDKitSVGVisualizer(legend_units="raw")
    >>> svg_str = viz_raw_units.visualize(mol_rep, explanation)

    >>> viz_uniform_radius = RDKitSVGVisualizer(scale_atom_radius=False)
    >>> svg_str = viz_uniform_radius.visualize(mol_rep, explanation)
    """

    def __init__(
        self,
        width: int = 400,
        height: int = 300,
        show_legend: bool = True,
        legend_width: int = 80,
        color_scale: Literal["separate", "combined"] = "separate",
        legend_units: Literal["raw", "normalized"] = "normalized",
        scale_atom_radius: bool = True,
    ) -> None:
        if color_scale not in ("separate", "combined"):
            raise ValueError(
                f"color_scale must be 'separate' or 'combined', got {color_scale!r}"
            )
        if legend_units not in ("raw", "normalized"):
            raise ValueError(
                f"legend_units must be 'raw' or 'normalized', got {legend_units!r}"
            )
        self.width: int = width
        self.height: int = height
        self.show_legend: bool = show_legend
        self.legend_width: int = legend_width
        self.color_scale: Literal["separate", "combined"] = color_scale
        self.legend_units: Literal["raw", "normalized"] = legend_units
        self.scale_atom_radius: bool = scale_atom_radius

    def visualize(
        self,
        mol_rep: MoleculeRepresentation,
        explanation: Explanation,
    ) -> str:
        """Render the explanation onto the molecule and return an SVG string.

        Parameters
        ----------
        mol_rep : MoleculeRepresentation
            Featurised molecule container.  ``mol_rep.mol`` (the RDKit
            ``Mol`` object) is used directly for drawing.
        explanation : Explanation
            XAI output holding ``atom_scores`` and optionally ``bond_scores``.
            All atoms present in ``mol_rep.mol`` should appear as keys in
            ``atom_scores`` (atoms absent from the dict receive no highlight
            colour).

        Returns
        -------
        str
            A complete, self-contained SVG string suitable for writing to a
            ``.svg`` file or rendering with ``IPython.display.SVG``. Includes
            a color-bar legend alongside the molecule unless
            ``self.show_legend`` is ``False`` or there are no scores at all.

        Notes
        -----
        When ``self.color_scale == "separate"`` (the default), atom scores
        and bond scores are normalized against their own independent
        min/max, so each element type's internal contrast is preserved
        even when the two come from very different distributions (see the
        class docstring). When ``self.color_scale == "combined"``, both are
        normalized against one shared min/max of the combined atom and
        bond scores.

        Within whichever min/max population is selected, the color mapping
        is sign-aware — diverging (zero-anchored, independent per-side
        span) when the population has both positive and negative scores,
        sequential (plain min→max stretch) when it's entirely one sign
        (see :func:`_score_to_t`).

        ``self.legend_units`` independently controls what number is
        *printed* at each legend tick — ``"normalized"`` (the default)
        prints the same value driving the tick's color; ``"raw"`` prints
        the raw score. ``self.scale_atom_radius`` independently controls
        whether highlighted atom-circle radius scales with score
        magnitude. None of these three settings touch
        ``explanation.atom_scores``/``bond_scores`` themselves.
        """
        mol: Mol = mol_rep.mol

        atom_scores: Dict[int, float] = explanation.atom_scores
        bond_scores: Dict[int, float] = explanation.bond_scores
        atom_values: List[float] = list(atom_scores.values())
        bond_values: List[float] = list(bond_scores.values())
        all_scores: List[float] = atom_values + bond_values

        # --- Compute score range(s) for the colour scale ---------------------
        if self.color_scale == "combined":
            if all_scores:
                shared_min, shared_max = min(all_scores), max(all_scores)
            else:
                shared_min = shared_max = 0.0
            atom_min = bond_min = shared_min
            atom_max = bond_max = shared_max
        else:
            atom_min, atom_max = (min(atom_values), max(atom_values)) if atom_values else (0.0, 0.0)
            bond_min, bond_max = (min(bond_values), max(bond_values)) if bond_values else (0.0, 0.0)

        # --- Build atom highlight lists -------------------------------------
        highlight_atoms: List[int] = list(atom_scores.keys())
        highlight_atom_colors: _ColorMap = {
            idx: _score_to_rgb(score, atom_min, atom_max)
            for idx, score in atom_scores.items()
        }
        highlight_atom_radii: Optional[Dict[int, float]] = (
            {
                idx: _atom_radius(score, atom_min, atom_max)
                for idx, score in atom_scores.items()
            }
            if self.scale_atom_radius
            else None
        )

        # --- Build bond highlight lists -------------------------------------
        highlight_bonds: List[int] = list(bond_scores.keys())
        highlight_bond_colors: _ColorMap = {
            idx: _score_to_rgb(score, bond_min, bond_max)
            for idx, score in bond_scores.items()
        }

        # --- Draw via rdMolDraw2D -------------------------------------------
        drawer = rdMolDraw2D.MolDraw2DSVG(self.width, self.height)
        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=highlight_atom_colors,
            highlightAtomRadii=highlight_atom_radii,
            highlightBonds=highlight_bonds,
            highlightBondColors=highlight_bond_colors,
        )
        drawer.FinishDrawing()
        raw_svg = drawer.GetDrawingText()

        if not self.show_legend or not all_scores:
            return raw_svg

        if self.color_scale == "separate" and atom_values and bond_values:
            legend_svg = _build_dual_legend_svg(
                atom_min, atom_max, bond_min, bond_max, self.legend_width, self.height,
                legend_units=self.legend_units,
            )
        elif atom_values:
            legend_svg = _build_legend_svg(
                atom_min, atom_max, self.legend_width, self.height, legend_units=self.legend_units
            )
        else:
            legend_svg = _build_legend_svg(
                bond_min, bond_max, self.legend_width, self.height, legend_units=self.legend_units
            )

        return _compose_svg_with_legend(raw_svg, legend_svg, self.width, self.height, self.legend_width)
