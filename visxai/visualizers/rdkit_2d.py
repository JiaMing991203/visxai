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

import re
import uuid
from typing import Dict, List, Literal, Optional, Tuple
from xml.sax.saxutils import escape as _xml_escape

from rdkit.Chem import Bond, Mol
from rdkit.Chem.Draw import rdMolDraw2D

from visxai.core.base_visualizer import BaseVisualizer
from visxai.core.data_types import (
    Explanation,
    MoleculeRepresentation,
    ScoreContribution,
)


# Type alias used by rdMolDraw2D color arguments: RGB triple in [0, 1].
_RGB = Tuple[float, float, float]
_ColorMap = Dict[int, _RGB]

# Below this fraction of the majority side's magnitude, the minority side of
# a straddling score population is treated as negligible (see _score_to_t's
# deadband branch) rather than scaled against its own tiny span.
_MINORITY_DEADBAND_RATIO = 0.05


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
    - **Straddles zero, but one side is negligible** (minority side's
      magnitude is under ``_MINORITY_DEADBAND_RATIO`` — 5% — of the majority
      side's): the independent-per-side formula above has its own edge case
      here. If the minority side collapses to a single near-zero value, that
      value gets divided by its own tiny span and forced to the full
      ``t=-1.0``/``t=1.00`` extreme, rendering pure red/green for a score
      that's really indistinguishable from zero — e.g. a population
      ``-0.0003..+1.68``: the old per-side formula gave the ``-0.0003`` atom
      ``t=-1.0`` (solid red), even though it's ``0.02%`` the size of the
      positive side's swing. In this regime both sides instead share one
      span, the majority side's own extreme (``span = majority_span`` for
      both signs) — the same shared-span formula the diverging branch above
      was fixed *away* from, deliberately reused here because it's actually
      correct when one side is this negligible. That same population now
      gives the ``-0.0003`` atom ``t = -0.0003 / 1.68 ≈ -0.0002`` (visually
      white, as it should be) while the positive side is unaffected. This is
      a plain ratio threshold, not a continuous blend, so a population
      sitting exactly at the 5% boundary can see a small color
      discontinuity as it crosses — an accepted, simple tradeoff, consistent
      with this module's other fixed-constant thresholds (e.g.
      ``_MIN_ATOM_RADIUS``/``_MAX_ATOM_RADIUS``).
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
        majority_span = max(max_score, -min_score)
        minority_span = min(max_score, -min_score)
        if minority_span < _MINORITY_DEADBAND_RATIO * majority_span:
            span = majority_span
        else:
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


def _gradient_stops(
    min_score: float,
    max_score: float,
    legend_units: Literal["raw", "normalized"] = "raw",
) -> List[Tuple[float, _RGB]]:
    """Compute correctly-positioned color-gradient stops for a score range.

    Stop position ``0.0`` corresponds to ``min_score`` and ``1.0`` to
    ``max_score``. When the range straddles zero, the "zero" stop's
    position depends on ``legend_units``:

    - ``"raw"`` (default): the position reflects where zero actually falls
      in ``[min_score, max_score]`` by raw magnitude — e.g. a ``-0.3..1.0``
      range puts white at position ``0.3 / 1.3 ≈ 0.23``, not the midpoint.
      This matches raw-unit tick labels, which are meant to reflect true
      relative magnitude via position too.
    - ``"normalized"``: the zero stop is fixed at the exact midpoint
      (``0.5``), regardless of raw magnitude. This is deliberately
      **not** proportional to raw score, because :func:`_score_to_t`
      independently rescales each side of a straddling range to its own
      full ``±1`` extent — so the printed labels always read
      ``-1.00``/``0.00``/``+1.00``, implying a symmetric scale. Positioning
      the zero stop by raw-score proportion instead would contradict that
      symmetry: e.g. a population dominated by negative scores with only a
      tiny positive extreme (real case: aspirin-adjacent IG atom scores
      spanning roughly ``-1.38..+0.03``) would put the zero stop at
      ``~98%`` — squeezing the entire green half into a cramped sliver a
      couple of percent tall, even though the label right next to it reads
      a clean ``0.00``/``+1.00``. Fixing the zero stop at the midpoint
      makes the green (or red) half of the bar always occupy a full,
      smoothly-graded half, matching what the symmetric ``-1..+1`` labels
      already imply — this is what fixes that "tiny colored sliver" look,
      not a change to any color value.

    Parameters
    ----------
    min_score : float
        The minimum score in the range being visualized.
    max_score : float
        The maximum score in the range being visualized.
    legend_units : {"raw", "normalized"}, optional
        Whether the zero stop (when present) is positioned by raw-score
        proportion (``"raw"``, the default — byte-identical to this
        function's output before this parameter existed) or fixed at the
        midpoint (``"normalized"``). Has no effect on stop colors, or on
        non-straddling ranges (no zero stop exists to reposition).

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
        if legend_units == "normalized":
            zero_position = 0.5
        else:
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
        and the max tick reads ``1.00``. Also determines the zero stop's
        *position* on the bar via :func:`_gradient_stops` — fixed at the
        midpoint under ``"normalized"``, matching the symmetric ``-1..+1``
        labels, rather than raw-score-proportional (see that function's
        docstring for why).

    Returns
    -------
    str
        An SVG fragment (``<defs>``/``<rect>``/``<text>`` elements, no
        wrapping ``<svg>`` tag) suitable for embedding inside a ``<g>``.
    """
    stops = _gradient_stops(min_score, max_score, legend_units=legend_units)

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

    # The score each stop represents is derived independently of its
    # position -- NOT by inverting position back into a score via linear
    # interpolation, which would be wrong whenever _gradient_stops uses a
    # non-proportional position (the "normalized" zero stop is fixed at
    # the midpoint regardless of raw magnitude; see that function's
    # docstring). _gradient_stops always builds stops in this exact order
    # (min, zero-if-straddling, max), so zipping against it directly keeps
    # each stop's label paired with the real score behind its color.
    if len(stops) == 1:
        stop_scores = [min_score]
    elif min_score < 0.0 < max_score:
        stop_scores = [min_score, 0.0, max_score]
    else:
        stop_scores = [min_score, max_score]

    labels: List[str] = []
    for (position, _), score in zip(stops, stop_scores):
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
    id_suffix: str = "",
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
    id_suffix : str, optional
        Appended to both gradient element ids (``"visxaiLegendGradientAtoms" +
        id_suffix`` / ``"visxaiLegendGradientBonds" + id_suffix``). Defaults
        to ``""`` — byte-identical to this function's output before this
        parameter existed. :meth:`RDKitSVGVisualizer.visualize` passes a
        per-call unique suffix so that two different rendered figures never
        share an ``id``, even if a notebook viewer ends up placing several
        of this visualizer's SVG outputs into one shared HTML document
        (``id`` references resolve document-wide in SVG/HTML, not per
        inline ``<svg>`` fragment — without a unique suffix, a viewer that
        doesn't isolate each output could resolve ``url(#...)`` to a
        *different* figure's gradient definition than the one actually
        inside that figure).

    Returns
    -------
    str
        An SVG fragment (two ``<g transform="translate(...)">``-wrapped
        legends) suitable for embedding inside a ``<g>``.
    """
    half_height = height // 2
    atom_svg = _build_legend_svg(
        atom_min, atom_max, width, half_height, title="Atoms",
        gradient_id=f"visxaiLegendGradientAtoms{id_suffix}", legend_units=legend_units,
    )
    bond_svg = _build_legend_svg(
        bond_min, bond_max, width, half_height, title="Bonds",
        gradient_id=f"visxaiLegendGradientBonds{id_suffix}", legend_units=legend_units,
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


# Matches a self-closing SVG element (tag + attributes, no nested content).
# RDKit's own attribute values never contain '<' or '>', so this is safe
# without needing re.DOTALL for the multi-line 'd' paths that atom-label
# glyphs use.
_SELF_CLOSING_ELEMENT_RE = re.compile(r"<(\w+)((?:[^<>])*)/>")
_CLASS_ATTR_RE = re.compile(r"class='([^']*)'")
_BOND_CLASS_TOKEN_RE = re.compile(r"\bbond-(\d+)\b")
_ATOM_CLASS_TOKEN_RE = re.compile(r"\batom-(\d+)\b")

# SMILES-style bond-order character, reused from the sequence path's own
# explicit-character convention (features/sequences.py) so a tooltip's bond
# symbol reads the same way a SMILES string would write it.
_BOND_ORDER_SYMBOLS: Dict[str, str] = {
    "SINGLE": "-",
    "DOUBLE": "=",
    "TRIPLE": "#",
    "AROMATIC": ":",
}


def _bond_order_symbol(bond: Bond) -> str:
    """Return a SMILES-style character for a bond's order (``-``/``=``/``#``/``:``).

    Parameters
    ----------
    bond : rdkit.Chem.Bond
        The bond to inspect.

    Returns
    -------
    str
        ``"-"``, ``"="``, ``"#"``, or ``":"`` for single/double/triple/aromatic
        bonds; ``"~"`` for any other RDKit bond type (e.g. dative, zero-order).
    """
    return _BOND_ORDER_SYMBOLS.get(str(bond.GetBondType()), "~")


def _format_provenance(
    contributions: Optional[List[ScoreContribution]],
) -> str:
    """Summarise how a score was assembled, as a short human-readable clause.

    Turns a list of
    :class:`~visxai.core.data_types.ScoreContribution` entries into text like
    ``"1 bit, split 6 ways"`` or ``"2 tokens, each in full"``. The wording
    keys off ``shared_among`` alone rather than any mode name, because the
    three sharing behaviours in this repo have no common vocabulary: the
    fingerprint path always divides, the sequence path never does, and
    ``bond_score_mode`` names only the atom-set-versus-bond-set axis.

    Parameters
    ----------
    contributions : list[ScoreContribution] or None
        The element's provenance entries, or ``None``/empty when the
        explainer did not record any.

    Returns
    -------
    str
        The clause, or ``""`` when there is nothing to report — callers append
        it only when non-empty, so an explainer that records no provenance
        produces exactly the tooltip text it did before this existed.

    Examples
    --------
    >>> _format_provenance([
    ...     {"source_kind": "bit", "source_index": 3, "source_score": -2.52,
    ...      "shared_among": 6, "contribution": -0.42},
    ... ])
    '1 bit, split 6 ways'
    >>> _format_provenance(None)
    ''
    """
    if not contributions:
        return ""

    n = len(contributions)
    noun = "bit" if contributions[0]["source_kind"] == "bit" else "token"
    subject = f"{n} {noun}" if n == 1 else f"{n} {noun}s"

    divisors = {c["shared_among"] for c in contributions}
    if divisors == {1}:
        # shared_among == 1 is duplication: the source handed over its whole
        # score without dividing.
        return f"{subject}, {'each ' if n > 1 else ''}in full"

    low, high = min(divisors), max(divisors)
    spread = f"{low} ways" if low == high else f"{low}-{high} ways"
    return f"{subject}, {'each ' if n > 1 else ''}split {spread}"


def _build_atom_tooltip_text(
    mol: Mol,
    atom_idx: int,
    score: float,
    contributions: Optional[List[ScoreContribution]] = None,
) -> str:
    """Build the hover-tooltip text for one atom.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The molecule ``atom_idx`` belongs to.
    atom_idx : int
        The atom's index within ``mol``.
    score : float
        The atom's score from ``explanation.atom_scores``.
    contributions : list[ScoreContribution], optional
        The atom's entry from ``explanation.atom_provenance``. When given and
        non-empty, a clause explaining how the score was assembled is
        appended. Optional so the many callers that have no provenance to
        offer keep producing byte-identical text.

    Returns
    -------
    str
        e.g. ``"Atom 3 (O): score -0.4200"``, or
        ``"Atom 3 (O): score -0.4200 — 1 bit, split 6 ways"``.
    """
    symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
    text = f"Atom {atom_idx} ({symbol}): score {score:.4f}"
    clause = _format_provenance(contributions)
    return f"{text} — {clause}" if clause else text


def _build_bond_tooltip_text(
    mol: Mol,
    bond_idx: int,
    score: float,
    contributions: Optional[List[ScoreContribution]] = None,
) -> str:
    """Build the hover-tooltip text for one bond.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The molecule ``bond_idx`` belongs to.
    bond_idx : int
        The bond's index within ``mol``.
    score : float
        The bond's score from ``explanation.bond_scores``.
    contributions : list[ScoreContribution], optional
        The bond's entry from ``explanation.bond_provenance``. See
        :func:`_build_atom_tooltip_text`.

    Returns
    -------
    str
        e.g. ``"Bond 5 (C=O): score 1.2000"``.
    """
    bond = mol.GetBondWithIdx(bond_idx)
    begin_symbol = bond.GetBeginAtom().GetSymbol()
    end_symbol = bond.GetEndAtom().GetSymbol()
    order_symbol = _bond_order_symbol(bond)
    text = (
        f"Bond {bond_idx} ({begin_symbol}{order_symbol}{end_symbol}): "
        f"score {score:.4f}"
    )
    clause = _format_provenance(contributions)
    return f"{text} — {clause}" if clause else text


def _inject_tooltips(
    svg_text: str,
    mol: Mol,
    atom_scores: Dict[int, float],
    bond_scores: Dict[int, float],
    atom_provenance: Optional[Dict[int, List[ScoreContribution]]] = None,
    bond_provenance: Optional[Dict[int, List[ScoreContribution]]] = None,
) -> str:
    """Inject a native-hover ``<title>`` child into every scored atom/bond shape.

    RDKit tags every atom- and bond-related SVG shape with a
    ``class='atom-N'`` / ``class='bond-K atom-I atom-J'`` attribute (bond
    shapes carry their own endpoint atoms' classes too, as a byproduct of
    how RDKit composes the class string — not a separate atom highlight).
    This walks every self-closing element, classifies it as bond-related
    (class contains a ``bond-K`` token) or atom-only (class contains only
    ``atom-N`` tokens), and — only when that index has an entry in the
    corresponding score dict — turns the element from self-closing into an
    open/close pair with a ``<title>`` child, which browsers render as a
    native hover tooltip. Elements with no ``class`` at all (e.g. the plain
    background ``<rect>``, unclassed stereo-wedge strokes) are left
    untouched.

    An index absent from ``atom_scores``/``bond_scores`` gets no tooltip at
    all, rather than an explicit "unattributed" one — the same "no entry =
    no information available" convention already used elsewhere in this
    repo for partial bond coverage (see ``AttentionExplainer`` in
    the README), so hovering an unscored element behaves exactly as it
    did before this feature existed.

    Parameters
    ----------
    svg_text : str
        A complete SVG string (or self-contained fragment) containing
        RDKit-drawn ``class='atom-N'``/``class='bond-K ...'`` elements.
    mol : rdkit.Chem.Mol
        The molecule the SVG was drawn from, used to look up atom/bond
        symbols for the tooltip text.
    atom_scores : dict[int, float]
        Scores keyed by atom index; only these atoms get a tooltip.
    bond_scores : dict[int, float]
        Scores keyed by bond index; only these bonds get a tooltip.
    atom_provenance : dict[int, list[ScoreContribution]], optional
        Per-atom provenance from the ``Explanation``. When supplied, each
        tooltip gains a clause explaining how its score was assembled.
        Elements missing from it simply get no clause, so a partially
        populated dict is fine.
    bond_provenance : dict[int, list[ScoreContribution]], optional
        Per-bond equivalent of ``atom_provenance``.

    Returns
    -------
    str
        ``svg_text`` with ``<title>`` children injected. Byte-identical to
        the input wherever no element matched, or matched but its index had
        no score entry.
    """
    atom_prov = atom_provenance or {}
    bond_prov = bond_provenance or {}

    def _replace(match: "re.Match[str]") -> str:
        tag, attrs = match.group(1), match.group(2)
        class_match = _CLASS_ATTR_RE.search(attrs)
        if class_match is None:
            return match.group(0)
        class_value = class_match.group(1)

        bond_match = _BOND_CLASS_TOKEN_RE.search(class_value)
        if bond_match is not None:
            bond_idx = int(bond_match.group(1))
            if bond_idx not in bond_scores:
                return match.group(0)
            title_text = _build_bond_tooltip_text(
                mol, bond_idx, bond_scores[bond_idx], bond_prov.get(bond_idx)
            )
        else:
            atom_match = _ATOM_CLASS_TOKEN_RE.search(class_value)
            if atom_match is None:
                return match.group(0)
            atom_idx = int(atom_match.group(1))
            if atom_idx not in atom_scores:
                return match.group(0)
            title_text = _build_atom_tooltip_text(
                mol, atom_idx, atom_scores[atom_idx], atom_prov.get(atom_idx)
            )

        return f"<{tag}{attrs}><title>{_xml_escape(title_text)}</title></{tag}>"

    return _SELF_CLOSING_ELEMENT_RE.sub(_replace, svg_text)


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
    by the test suite, which doesn't import this module
    at all) is unaffected regardless of either setting.

    Under ``legend_units="normalized"``, a straddling score population's
    zero point is drawn at the exact **midpoint** of the bar, matching the
    symmetric ``-1.00``/``0.00``/``+1.00`` labels — not at its raw-score-
    proportional position (which is what ``legend_units="raw"`` still
    uses, correctly, since its labels are raw magnitudes). Without this, a
    population dominated by one sign with only a tiny extreme on the other
    side (e.g. atom scores spanning roughly ``-1.38..+0.03``) would squeeze
    that whole side's color into a cramped few-percent-tall sliver right
    next to a label that reads a clean ``0.00``/``+1.00`` — this is what
    positioning at the midpoint fixes (see :func:`_gradient_stops` for the
    full derivation); no score or color value changes, only where the
    zero-crossing is drawn.

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

    **The color scale is per-call by default** — ``atom_min``/``atom_max``
    (and ``bond_min``/``bond_max``) are recomputed fresh from each
    ``explanation`` passed to :meth:`visualize`, so the exact same score
    can render a different color across two different molecules if their
    other scores happen to differ. This is fine for looking at one molecule
    in isolation, but is misleading when placing several rendered figures
    side by side for comparison — a reader's eye reads color, and two
    figures' colors aren't comparable unless they share a scale. Pass
    ``atom_range=(min, max)`` and/or ``bond_range=(min, max)`` to fix that
    element type's normalization range across every call made with this
    instance, instead of deriving it per call. Each override applies only
    to its own element type and composes with ``color_scale``: under
    ``"combined"``, giving only ``atom_range`` overrides just the atom half
    of the shared pool, leaving bonds on the auto-computed shared range —
    pass both if a single, fully fixed shared range is wanted. Leaving both
    ``None`` (the default) preserves the original per-call behavior exactly.

    **Native hover tooltips are opt-in via ``show_tooltips=True``.** When
    enabled, every atom/bond shape that has a score gets a ``<title>``
    child injected (:func:`_inject_tooltips`) reading e.g. ``"Atom 3 (O):
    score -0.4200"`` / ``"Bond 5 (C=O): score 1.2000"`` — a browser's
    native hover tooltip, no JavaScript or new dependency involved. An
    element whose index has no entry in ``atom_scores``/``bond_scores``
    (e.g. most bonds under the sequence path's inherently partial bond
    coverage) gets no tooltip at all, rather than an explicit
    "unattributed" one — matching this repo's existing "no entry = no
    information" convention. Defaults to ``False``: byte-identical to this
    visualizer's output before this parameter existed.

    **Every legend's gradient elements get a unique, random ``id`` per
    call to :meth:`visualize`.** SVG ``id`` references (``url(#...)``)
    resolve document-wide, not scoped to one inline ``<svg>`` fragment —
    so if a notebook viewer ever places several of this visualizer's SVG
    outputs into one shared HTML document without isolating each output
    (e.g. some notebook UIs keep every cell's rendered output alive in a
    single webview page for scrollback), a fixed, reused ``id`` across
    every rendered figure (as this module originally used —
    ``"visxaiLegendGradientAtoms"``/``"visxaiLegendGradientBonds"``, byte-
    identical on every call) means ``url(#visxaiLegendGradientAtoms)`` can
    resolve to a *different* figure's gradient than the one actually drawn
    for that figure, since a spec-compliant document-wide id lookup
    returns whichever matching element appears first, not the one local to
    the referencing figure. Found and diagnosed this way after a user
    report of legend colors that looked wrong in VS Code's Jupyter output
    panel but were confirmed correct (matching, byte-for-byte, MD5-hash-
    verified SVG data) once the same file was opened in a plain browser
    tab — which opens each SVG as its own isolated document, unaffected by
    any id collision with other cells' outputs. Every call to
    :meth:`visualize` now appends a random 8-hex-character suffix (via
    ``uuid.uuid4()``) to every gradient id it emits, so no two rendered
    figures can ever collide, regardless of how a given viewer composes
    multiple outputs into a document. Purely a rendering-robustness fix —
    doesn't change any score, color, or the rest of the SVG's structure.

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
    atom_range : tuple[float, float], optional
        Fixed ``(min, max)`` to normalize atom scores against, overriding
        the default of computing it fresh from each ``explanation``. Every
        call to :meth:`visualize` with a given instance then colors atom
        scores on the *same* scale, so the same score maps to the same
        color across different molecules/explanations — the per-call
        auto-computed range (the default, ``atom_range=None``) instead lets
        one molecule's outlier silently shift what an identical score in
        another molecule looks like, which is misleading when comparing
        several rendered figures side by side. Composes with
        ``color_scale``: under ``"combined"``, an explicit ``atom_range``
        overrides only the atom half of the shared pool, leaving
        ``bond_range`` (or the auto-computed bond range) independent — pass
        both if a single fully-shared fixed range is wanted. Independent of
        ``legend_units``, which controls what a tick's *number* looks like,
        not the range it's computed against. Raises ``ValueError`` at
        construction time if ``min > max``.
    bond_range : tuple[float, float], optional
        Fixed ``(min, max)`` to normalize bond scores against, mirroring
        ``atom_range``. ``None`` (default) preserves the existing
        per-call auto-computed behavior.
    show_tooltips : bool, optional
        Whether to inject a native-hover ``<title>`` element into every
        scored atom/bond shape. Defaults to ``False``. See above.

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

    >>> # Same atom color scale across every molecule rendered by this instance.
    >>> viz_fixed_scale = RDKitSVGVisualizer(atom_range=(-1.0, 1.0), bond_range=(0.0, 2.0))
    >>> svg_str = viz_fixed_scale.visualize(mol_rep, explanation)

    >>> # Native browser hover tooltips on every scored atom/bond.
    >>> viz_tooltips = RDKitSVGVisualizer(show_tooltips=True)
    >>> svg_str = viz_tooltips.visualize(mol_rep, explanation)
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
        atom_range: Optional[Tuple[float, float]] = None,
        bond_range: Optional[Tuple[float, float]] = None,
        show_tooltips: bool = False,
    ) -> None:
        if color_scale not in ("separate", "combined"):
            raise ValueError(
                f"color_scale must be 'separate' or 'combined', got {color_scale!r}"
            )
        if legend_units not in ("raw", "normalized"):
            raise ValueError(
                f"legend_units must be 'raw' or 'normalized', got {legend_units!r}"
            )
        if atom_range is not None and atom_range[0] > atom_range[1]:
            raise ValueError(f"atom_range must be (min, max) with min <= max, got {atom_range!r}")
        if bond_range is not None and bond_range[0] > bond_range[1]:
            raise ValueError(f"bond_range must be (min, max) with min <= max, got {bond_range!r}")
        self.width: int = width
        self.height: int = height
        self.show_legend: bool = show_legend
        self.legend_width: int = legend_width
        self.color_scale: Literal["separate", "combined"] = color_scale
        self.legend_units: Literal["raw", "normalized"] = legend_units
        self.scale_atom_radius: bool = scale_atom_radius
        self.atom_range: Optional[Tuple[float, float]] = atom_range
        self.bond_range: Optional[Tuple[float, float]] = bond_range
        self.show_tooltips: bool = show_tooltips

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

        # An explicit fixed range overrides whichever range was just computed
        # above (auto-computed per-call, or the "combined" shared pool) for
        # its element type only -- so the same score renders the same color
        # across different visualize() calls/molecules on this instance.
        if self.atom_range is not None:
            atom_min, atom_max = self.atom_range
        if self.bond_range is not None:
            bond_min, bond_max = self.bond_range

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
            final_svg = raw_svg
        else:
            # A per-call unique suffix on every gradient id, so that two
            # different rendered figures can never collide even if a notebook
            # viewer places several of this visualizer's SVG outputs into one
            # shared HTML document (see _build_dual_legend_svg's id_suffix
            # docstring for why a collision is possible at all: SVG id
            # references resolve document-wide, not per inline <svg> fragment).
            id_suffix = f"-{uuid.uuid4().hex[:8]}"

            if self.color_scale == "separate" and atom_values and bond_values:
                legend_svg = _build_dual_legend_svg(
                    atom_min, atom_max, bond_min, bond_max, self.legend_width, self.height,
                    legend_units=self.legend_units, id_suffix=id_suffix,
                )
            elif atom_values:
                legend_svg = _build_legend_svg(
                    atom_min, atom_max, self.legend_width, self.height, legend_units=self.legend_units,
                    gradient_id=f"visxaiLegendGradient{id_suffix}",
                )
            else:
                legend_svg = _build_legend_svg(
                    bond_min, bond_max, self.legend_width, self.height, legend_units=self.legend_units,
                    gradient_id=f"visxaiLegendGradient{id_suffix}",
                )

            final_svg = _compose_svg_with_legend(
                raw_svg, legend_svg, self.width, self.height, self.legend_width
            )

        if self.show_tooltips:
            final_svg = _inject_tooltips(
                final_svg,
                mol,
                atom_scores,
                bond_scores,
                explanation.atom_provenance,
                explanation.bond_provenance,
            )

        return final_svg
