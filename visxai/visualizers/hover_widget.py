"""XSMILES-style hover-linked molecule widget (Phase 2 of interactive visualization).

Pairs the 2-D structure with a **SMILES strip** — the SMILES string
rendered character by character, each carrying a score bar above it —
and links the two: hovering an atom in the structure highlights the
characters that spell it, and hovering a character highlights the atom.
This is the interaction a static SVG structurally cannot provide (see
the project's design notes for why native ``<title>`` tooltips fall
short), and the reason Phase 2 exists.

Two design points worth knowing before changing anything here:

**The character mapping is not new work.**
:func:`~visxai.features.sequences.compute_atom_char_spans` and
:func:`~visxai.features.sequences.compute_bond_char_spans` already derive
each atom's and bond's character span purely from SMILES syntax, for *any*
molecule regardless of paradigm. The strip is a rendering of mapping data
this repo already computes and tests — no new alignment logic is
introduced, and none should be.

**Colors are computed here, not in JavaScript.**
:func:`build_payload` runs the visualizer's own ``_score_to_t`` /
``_score_to_rgb`` and ships resolved hex colors, so a strip bar and its
atom in the structure render the same value. Reimplementing the score →
color math on the JS side would create two implementations that can
silently drift — exactly the failure this repo already fixed once by
factoring ``_score_to_t`` out of ``_score_to_rgb``. Note the final
float → byte step uses :func:`_rdkit_hex`, not ``_rgb_to_hex``: RDKit
truncates where the latter rounds, and matching it is what makes "the same
value" literally true rather than approximately so.

``anywidget`` is an optional dependency (the ``[viz]`` extra). Importing
this module without it succeeds; the error is raised at widget
construction with an install hint, mirroring :mod:`visxai.visualizers.interactive`.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional, Tuple

try:  # pragma: no cover - trivial import guard
    import anywidget
    import traitlets
except ImportError:  # pragma: no cover - exercised only without the [viz] extra
    anywidget = None  # type: ignore[assignment]
    traitlets = None  # type: ignore[assignment]

from visxai.core.data_types import (
    Explanation,
    MoleculeRepresentation,
    ScoreContribution,
)
from visxai.features.sequences import compute_atom_char_spans, compute_bond_char_spans
from visxai.visualizers.rdkit_2d import (
    RDKitSVGVisualizer,
    _RGB,
    _build_atom_tooltip_text,
    _build_bond_tooltip_text,
    _format_provenance,
    _score_to_rgb,
    _score_to_t,
)

_INSTALL_HINT = (
    "MoleculeHoverWidget requires anywidget, which is an optional "
    "dependency. Install it with:  pip install 'visxai[viz]'"
)

# Where the frontend lives. anywidget accepts a Path (and watches it during
# development); both files must ship as package data -- see pyproject.toml's
# [tool.setuptools.package-data]. anywidget validates these paths at
# CLASS-DEFINITION time, so a missing file breaks `import`, not just
# instantiation.
_STATIC = pathlib.Path(__file__).parent / "static"
_ESM_PATH = _STATIC / "hover_widget.js"
_CSS_PATH = _STATIC / "hover_widget.css"


def _rdkit_hex(rgb: _RGB) -> str:
    """Format an ``(R, G, B)`` triple in ``[0, 1]`` the way RDKit's SVG writer does.

    **Truncates** each channel (``int(v * 255)``) rather than rounding.
    This is not a style preference — it is what makes a strip bar and its
    atom in the structure render the *same* hex value, which is the entire
    reason colors are computed in Python rather than JavaScript.

    ``rdkit_2d._rgb_to_hex`` rounds instead, and the two disagree by one
    unit per channel on most values (verified against RDKit directly:
    ``0.5`` → RDKit ``0x7F`` = 127, rounding ``0x80`` = 128; ``0.27`` →
    ``0x44`` vs ``0x45``). Left as a separate helper rather than changing
    the shared one, because ``_rgb_to_hex`` also formats the legend
    gradient and several tests assert its exact output — a 1/255 shift is
    invisible but would churn tested bytes for no benefit.

    A consequence worth knowing: the **legend** therefore carries that same
    pre-existing 1/255 offset from the molecule it labels. Invisible, and
    not introduced here.

    Parameters
    ----------
    rgb : tuple[float, float, float]
        An ``(R, G, B)`` triple with channels in ``[0.0, 1.0]``.

    Returns
    -------
    str
        An ``#RRGGBB`` string byte-identical to what RDKit would emit for
        the same triple.
    """
    r, g, b = (int(channel * 255) for channel in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def _effective_ranges(
    explanation: Explanation,
    atom_range: Optional[Tuple[float, float]],
    bond_range: Optional[Tuple[float, float]],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Resolve the atom and bond normalization ranges the strip should use.

    Mirrors :class:`~visxai.visualizers.rdkit_2d.RDKitSVGVisualizer`'s
    ``color_scale="separate"`` default — atoms and bonds normalized
    independently — so a strip bar and its structure element are colored
    against the same span. An explicit range overrides its element type,
    exactly as the visualizer's ``atom_range``/``bond_range`` do; pass the
    same values here that were passed to the visualizer.

    Parameters
    ----------
    explanation : Explanation
        The explanation whose scores define the fallback ranges.
    atom_range : tuple[float, float] or None
        Explicit atom range, or ``None`` to compute it from
        ``explanation.atom_scores``.
    bond_range : tuple[float, float] or None
        Explicit bond range, or ``None`` to compute it from
        ``explanation.bond_scores``.

    Returns
    -------
    tuple[tuple[float, float], tuple[float, float]]
        ``((atom_min, atom_max), (bond_min, bond_max))``. An element type
        with no scores at all yields ``(0.0, 0.0)``, the degenerate range
        the visualizer already renders as neutral white.
    """
    atom_values = list(explanation.atom_scores.values())
    bond_values = list(explanation.bond_scores.values())
    atoms = atom_range if atom_range is not None else (
        (min(atom_values), max(atom_values)) if atom_values else (0.0, 0.0)
    )
    bonds = bond_range if bond_range is not None else (
        (min(bond_values), max(bond_values)) if bond_values else (0.0, 0.0)
    )
    return atoms, bonds


def _element_entry(
    mol_rep: MoleculeRepresentation,
    kind: str,
    index: int,
    score: float,
    low: float,
    high: float,
    contributions: Optional[List[ScoreContribution]] = None,
) -> Dict[str, Any]:
    """Compute the display data for one scored atom or bond.

    Single source for the label, normalized ``t``, color, and provenance
    clause, shared by :func:`build_char_map` and :func:`build_elements` so the
    SMILES strip and the hover readout can never disagree about the same
    element.

    Parameters
    ----------
    mol_rep : MoleculeRepresentation
        Supplies ``mol`` for symbol and bond-order lookup.
    kind : {"atom", "bond"}
        Which index space ``index`` belongs to.
    index : int
        The atom or bond index.
    score : float
        Its score.
    low, high : float
        The normalization range for this element type.
    contributions : list[ScoreContribution], optional
        This element's provenance entries from the ``Explanation``.

    Returns
    -------
    dict[str, Any]
        Keys ``score``, ``t``, ``color``, ``label``, ``provenance``, all
        JSON-native. ``provenance`` is a **separate key rather than being
        folded into ``label``** — unlike an SVG ``<title>``, which can only
        be one flat string, the readout renders ``·``-separated segments, so
        the clause is kept apart to sit alongside ``scaled …`` and
        ``no SMILES character``. It is ``""`` when the explainer recorded no
        provenance, which the frontend skips.
    """
    t = _score_to_t(score, low, high)
    label = (
        _build_atom_tooltip_text(mol_rep.mol, index, score)
        if kind == "atom"
        else _build_bond_tooltip_text(mol_rep.mol, index, score)
    )
    return {
        "score": float(score),
        "t": None if t is None else float(t),
        "color": _rdkit_hex(_score_to_rgb(score, low, high)),
        "label": label,
        "provenance": _format_provenance(contributions),
    }


def build_elements(
    mol_rep: MoleculeRepresentation,
    explanation: Explanation,
    strip_chars: List[Dict[str, Any]],
    atom_range: Optional[Tuple[float, float]] = None,
    bond_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Build display data for **every scored element**, strip or not.

    The SMILES strip can only represent what the SMILES string spells, and
    that is a strict subset of what the structure draws. Most bonds are
    *implicit* — a plain single or aromatic bond has no character at all —
    yet the tree path scores every bond and the visualizer colors every
    one of them. Measured on aspirin: 13 bonds scored, only 2 with an
    explicit character.

    Driving the hover readout off the strip therefore left **10 of 13
    visibly-colored bonds with no readout at all** — a regression against
    the native ``<title>`` tooltips, which covered every scored element.
    This map is what the readout reads instead; the strip stays
    SMILES-shaped, as it should be.

    Parameters
    ----------
    mol_rep : MoleculeRepresentation
        The molecule being rendered.
    explanation : Explanation
        Supplies the scores.
    strip_chars : list[dict[str, Any]]
        Output of :func:`build_char_map`, used only to mark which elements
        the strip actually shows.
    atom_range, bond_range : tuple[float, float], optional
        Fixed normalization ranges; see :func:`_effective_ranges`.

    Returns
    -------
    dict[str, dict[str, dict[str, Any]]]
        ``{"atom": {"<index>": entry}, "bond": {...}}``. Indices are
        **string** keys, since JSON object keys always are. Each entry adds
        ``in_strip`` to :func:`_element_entry`'s fields, so the frontend can
        say *why* nothing highlights in the strip for an implicit bond
        rather than appearing broken.
    """
    (atom_min, atom_max), (bond_min, bond_max) = _effective_ranges(
        explanation, atom_range, bond_range
    )
    shown = {(c["kind"], c["index"]) for c in strip_chars if c["kind"] is not None}

    elements: Dict[str, Dict[str, Dict[str, Any]]] = {"atom": {}, "bond": {}}
    for kind, scores, prov, low, high in (
        ("atom", explanation.atom_scores, explanation.atom_provenance, atom_min, atom_max),
        ("bond", explanation.bond_scores, explanation.bond_provenance, bond_min, bond_max),
    ):
        for index, score in scores.items():
            entry = _element_entry(
                mol_rep, kind, int(index), float(score), low, high, prov.get(int(index))
            )
            entry["in_strip"] = (kind, int(index)) in shown
            elements[kind][str(index)] = entry
    return elements


def build_char_map(
    smiles: str,
    mol_rep: MoleculeRepresentation,
    explanation: Explanation,
    atom_range: Optional[Tuple[float, float]] = None,
    bond_range: Optional[Tuple[float, float]] = None,
) -> List[Dict[str, Any]]:
    """Build the per-character SMILES strip data.

    One entry per character of ``smiles``, in order. A character that
    spells an atom carries that atom's score; a character that *is* a bond
    symbol (``=``, ``#``, a ring-closure digit, …) carries that bond's.
    Structural characters with no chemical owner — branch parens, most ring
    digits — carry no score and render as a gap in the strip, which is
    correct rather than missing data.

    Atoms take precedence over bonds when both claim a character. In
    practice the two span sets are disjoint (an atom's characters spell an
    element symbol; a bond's are the explicit order/closure symbols), but
    the precedence is fixed rather than incidental so the strip can never
    depend on dict iteration order.

    Parameters
    ----------
    smiles : str
        The SMILES string to lay out.
    mol_rep : MoleculeRepresentation
        Supplies ``mol`` for symbol/bond-order lookup in the readout labels.
    explanation : Explanation
        Supplies the scores.
    atom_range : tuple[float, float], optional
        Fixed atom normalization range; ``None`` computes it from the
        explanation (see :func:`_effective_ranges`).
    bond_range : tuple[float, float], optional
        Fixed bond normalization range, mirroring ``atom_range``.

    Returns
    -------
    list[dict[str, Any]]
        One dict per character with keys ``char``, ``kind``
        (``"atom"``/``"bond"``/``None``), ``index``, ``score``, ``t``,
        ``color``, and ``label``. Every value is JSON-native — floats are
        cast from any NumPy scalar the explainers may have produced, since
        traitlets serializes this straight to the frontend.
    """
    (atom_min, atom_max), (bond_min, bond_max) = _effective_ranges(
        explanation, atom_range, bond_range
    )

    owner: List[Optional[Tuple[str, int]]] = [None] * len(smiles)
    for bond_idx, (start, end) in compute_bond_char_spans(smiles, mol_rep.mol).items():
        for position in range(start, min(end, len(smiles))):
            owner[position] = ("bond", bond_idx)
    # Atoms last so they win any overlap -- see the precedence note above.
    for atom_idx, (start, end) in compute_atom_char_spans(smiles).items():
        for position in range(start, min(end, len(smiles))):
            owner[position] = ("atom", atom_idx)

    entries: List[Dict[str, Any]] = []
    for position, character in enumerate(smiles):
        held = owner[position]
        entry: Dict[str, Any] = {
            "char": character,
            "kind": None,
            "index": None,
            "score": None,
            "t": None,
            "color": None,
            "label": None,
            "provenance": None,
        }
        if held is not None:
            kind, index = held
            is_atom = kind == "atom"
            scores = explanation.atom_scores if is_atom else explanation.bond_scores
            prov = (
                explanation.atom_provenance if is_atom else explanation.bond_provenance
            )
            if index in scores:
                low, high = (atom_min, atom_max) if is_atom else (bond_min, bond_max)
                entry.update(
                    kind=kind,
                    index=int(index),
                    **_element_entry(
                        mol_rep,
                        kind,
                        int(index),
                        float(scores[index]),
                        low,
                        high,
                        prov.get(int(index)),
                    ),
                )
        entries.append(entry)
    return entries


def build_payload(
    mol_rep: MoleculeRepresentation,
    explanation: Explanation,
    visualizer: Optional[RDKitSVGVisualizer] = None,
    atom_range: Optional[Tuple[float, float]] = None,
    bond_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """Project a molecule and its explanation into a JSON-safe widget payload.

    This is the "JSON-safe projection" the interactive-visualization plan
    called out as genuinely new work.
    :class:`~visxai.core.data_types.MoleculeRepresentation` cannot cross the
    Python↔JavaScript boundary as-is — ``mol`` is a live RDKit C++ object,
    ``fingerprint_array`` is a NumPy array, and ``pyg_data`` (when present)
    is a ``torch_geometric`` object. This function keeps only what the
    frontend actually needs, in JSON-native types.

    Deliberately a **plain function**, not a widget method: it is the whole
    testable core of this module, and keeping it widget-free means the
    payload can be verified without ``anywidget`` installed or any
    frontend running — the same pattern
    ``_integrated_gradients_token_scores`` follows in the sequence path.

    Parameters
    ----------
    mol_rep : MoleculeRepresentation
        The molecule to render. ``mol_rep.smiles`` drives the strip.
    explanation : Explanation
        The already-computed explanation to visualize. Nothing here re-runs
        an explainer.
    visualizer : RDKitSVGVisualizer, optional
        The visualizer used to render the structure. ``None`` builds a
        default one with ``show_tooltips=False`` — the widget supplies its
        own richer hover, and a native ``<title>`` would additionally pop
        the browser's slow tooltip on top of it.
    atom_range : tuple[float, float], optional
        Fixed atom normalization range for the strip. Pass whatever was
        passed to ``visualizer`` so the two agree.
    bond_range : tuple[float, float], optional
        Fixed bond normalization range, mirroring ``atom_range``.

    Returns
    -------
    dict[str, Any]
        Keys ``svg``, ``smiles``, ``chars`` (see :func:`build_char_map`),
        ``elements`` (see :func:`build_elements`), ``atom_scores`` and
        ``bond_scores`` (index → score, with **string keys** — JSON object
        keys are always strings, so integer keys would silently change type
        in transit), and ``n_atoms``/``n_bonds``.

    Notes
    -----
    ``chars`` and ``elements`` are deliberately different sets, not
    redundant. ``chars`` is what the SMILES string spells; ``elements`` is
    everything the structure draws. Most bonds are implicit and appear only
    in the latter.
    """
    if visualizer is None:
        visualizer = RDKitSVGVisualizer(show_tooltips=False)

    smiles = mol_rep.smiles
    chars = build_char_map(smiles, mol_rep, explanation, atom_range, bond_range)
    return {
        "svg": visualizer.visualize(mol_rep, explanation),
        "smiles": smiles,
        "chars": chars,
        # Every scored element, including the many bonds the SMILES string
        # cannot spell. The readout reads THIS, not `chars` -- see
        # build_elements for why that distinction is load-bearing.
        "elements": build_elements(mol_rep, explanation, chars, atom_range, bond_range),
        "atom_scores": {str(k): float(v) for k, v in explanation.atom_scores.items()},
        "bond_scores": {str(k): float(v) for k, v in explanation.bond_scores.items()},
        "n_atoms": int(mol_rep.mol.GetNumAtoms()),
        "n_bonds": int(mol_rep.mol.GetNumBonds()),
    }


if anywidget is not None:  # pragma: no branch - the else is the install-hint stub

    class MoleculeHoverWidget(anywidget.AnyWidget):  # type: ignore[misc]
        """A hover-linked 2-D structure paired with an XSMILES-style SMILES strip.

        Renders the molecule SVG above a strip of the SMILES string, each
        character carrying a score bar. Hovering either view highlights the
        corresponding element in the other, and a readout follows the
        cursor — the interaction a static SVG cannot provide.

        Not a :class:`~visxai.core.base_visualizer.BaseVisualizer`, for the
        same reason :class:`~visxai.visualizers.interactive.ExplanationDashboard`
        isn't: that ABC promises ``visualize(mol_rep, explanation) -> str``
        and every caller relies on getting an SVG string back.

        Parameters
        ----------
        payload : dict
            Output of :func:`build_payload`. Prefer :meth:`from_explanation`
            over constructing this by hand.
        **kwargs : Any
            Forwarded to ``anywidget.AnyWidget`` (e.g. ``layout``).

        Attributes
        ----------
        payload : traitlets.Dict
            The synced widget data. Reassigning it re-renders the frontend,
            which is how the dashboard swaps molecules without rebuilding
            the widget.
        hovered : traitlets.Dict
            Written by the frontend on every hover, read-only from Python's
            perspective. Carries ``{"kind": "atom"|"bond", "index": int}``
            while the pointer is over an element, and ``{}`` when it leaves.
            Observe it to drive Python-side reactions to hover.

        Examples
        --------
        >>> widget = MoleculeHoverWidget.from_explanation(mol_rep, explanation)
        >>> widget.observe(lambda change: print(change["new"]), names="hovered")
        >>> widget
        """

        _esm = _ESM_PATH
        _css = _CSS_PATH
        payload = traitlets.Dict({}).tag(sync=True)
        hovered = traitlets.Dict({}).tag(sync=True)

        @classmethod
        def from_explanation(
            cls,
            mol_rep: MoleculeRepresentation,
            explanation: Explanation,
            visualizer: Optional[RDKitSVGVisualizer] = None,
            atom_range: Optional[Tuple[float, float]] = None,
            bond_range: Optional[Tuple[float, float]] = None,
            **kwargs: Any,
        ) -> "MoleculeHoverWidget":
            """Build a widget directly from a molecule and its explanation.

            Thin wrapper over :func:`build_payload` — the convenience path
            for the common case, so callers don't have to know the payload
            shape.

            Parameters
            ----------
            mol_rep : MoleculeRepresentation
                The molecule to render.
            explanation : Explanation
                The already-computed explanation.
            visualizer : RDKitSVGVisualizer, optional
                Visualizer for the structure; see :func:`build_payload`.
            atom_range, bond_range : tuple[float, float], optional
                Fixed normalization ranges; see :func:`build_payload`.
            **kwargs : Any
                Forwarded to the constructor.

            Returns
            -------
            MoleculeHoverWidget
                A widget ready to display.
            """
            payload = build_payload(
                mol_rep, explanation, visualizer, atom_range, bond_range
            )
            return cls(payload=payload, **kwargs)

        def update_explanation(
            self,
            mol_rep: MoleculeRepresentation,
            explanation: Explanation,
            visualizer: Optional[RDKitSVGVisualizer] = None,
            atom_range: Optional[Tuple[float, float]] = None,
            bond_range: Optional[Tuple[float, float]] = None,
        ) -> None:
            """Swap in a new molecule/explanation, re-rendering in place.

            Reassigns ``payload`` rather than rebuilding the widget, so the
            dashboard can change selection without tearing down and
            recreating the frontend on every interaction.

            Parameters
            ----------
            mol_rep : MoleculeRepresentation
                The new molecule.
            explanation : Explanation
                The new explanation.
            visualizer : RDKitSVGVisualizer, optional
                Visualizer for the structure; see :func:`build_payload`.
            atom_range, bond_range : tuple[float, float], optional
                Fixed normalization ranges; see :func:`build_payload`.
            """
            self.payload = build_payload(
                mol_rep, explanation, visualizer, atom_range, bond_range
            )

else:  # pragma: no cover - exercised only without the [viz] extra

    class MoleculeHoverWidget:  # type: ignore[no-redef]
        """Stub raising a helpful :class:`ImportError` when ``anywidget`` is absent.

        Mirrors :class:`~visxai.visualizers.interactive.ExplanationDashboard`'s
        guard: importing this module always succeeds, so the rest of the
        package stays usable without the ``[viz]`` extra, and the failure
        surfaces only when someone actually tries to build a widget.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(_INSTALL_HINT)

        @classmethod
        def from_explanation(cls, *args: Any, **kwargs: Any) -> "MoleculeHoverWidget":
            """Raise :class:`ImportError` — ``anywidget`` is not installed."""
            raise ImportError(_INSTALL_HINT)
