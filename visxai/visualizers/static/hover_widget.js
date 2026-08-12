// ESM frontend for visxai.visualizers.hover_widget.MoleculeHoverWidget.
//
// Renders the molecule SVG above an XSMILES-style strip: the SMILES string
// laid out character by character, each with a score bar above it. Hovering
// either view highlights the matching element in the other, and a readout
// follows the cursor.
//
// Two rules this file must keep:
//
//  1. NEVER recompute a score -> color here. Python already resolved every
//     color (matching RDKit's own float->byte truncation exactly), so the
//     strip and the structure agree by construction. Duplicating the score
//     math in JS creates two implementations that silently drift.
//
//  2. Element identity comes from RDKit's own `class` attributes on the SVG
//     shapes -- `atom-N`, and `bond-K atom-I atom-J` on bond shapes. A bond
//     shape also lists its endpoint atoms, so a bond token must be checked
//     FIRST; matching atoms first would attribute every bond to an atom.
//     This mirrors the same precedence rule in rdkit_2d._inject_tooltips.

const ATOM_RE = /\batom-(\d+)\b/;
const BOND_RE = /\bbond-(\d+)\b/;

/**
 * Identify which chemical element an SVG shape belongs to.
 *
 * @param {Element} el - an SVG shape carrying RDKit's class attribute
 * @returns {{kind: string, index: number}|null} null when the shape is
 *   structural (background rect, unclassed stroke) rather than chemical.
 */
export function identify(el) {
  const cls = el && el.getAttribute && el.getAttribute("class");
  if (!cls) return null;
  const bond = BOND_RE.exec(cls);
  if (bond) return { kind: "bond", index: Number(bond[1]) };
  const atom = ATOM_RE.exec(cls);
  if (atom) return { kind: "atom", index: Number(atom[1]) };
  return null;
}

/**
 * Collect every SVG shape belonging to a given element.
 *
 * One atom or bond is drawn as several shapes -- a highlight ellipse, one or
 * more stroke paths, and label glyphs -- so highlighting means touching all
 * of them, not just the one under the pointer.
 *
 * @param {Element} root - the container holding the rendered SVG
 * @param {string} kind - "atom" or "bond"
 * @param {number} index
 * @returns {Element[]}
 */
export function shapesFor(root, kind, index) {
  const out = [];
  for (const el of root.querySelectorAll("[class]")) {
    const id = identify(el);
    if (id && id.kind === kind && id.index === index) out.push(el);
  }
  return out;
}

/** Per-element saved stroke, so highlighting is exactly reversible. */
const SAVED = new WeakMap();

function applyHighlight(shapes) {
  for (const el of shapes) {
    if (!SAVED.has(el)) {
      SAVED.set(el, {
        stroke: el.style.stroke,
        width: el.style.strokeWidth,
      });
    }
    el.style.stroke = "#111";
    el.style.strokeWidth = "3px";
  }
}

function clearHighlight(shapes) {
  for (const el of shapes) {
    const saved = SAVED.get(el);
    if (saved) {
      el.style.stroke = saved.stroke;
      el.style.strokeWidth = saved.width;
    }
  }
}

/**
 * Build the SMILES strip: one column per character, bar above, glyph below.
 *
 * Bar geometry encodes the normalized score `t` -- height by |t|, direction
 * by sign, so positive and negative contributions read at a glance without
 * relying on color alone. Characters with no chemical owner (branch parens,
 * ring-closure digits) get a column with no bar, which is correct: they
 * carry no attribution rather than an attribution of zero.
 *
 * @param {Array<Object>} chars - payload.chars from the Python side
 * @param {(kind: string|null, index: number|null) => void} onHover
 * @returns {HTMLElement}
 */
export function buildStrip(chars, onHover) {
  const strip = document.createElement("div");
  strip.className = "vx-strip";

  chars.forEach((c, position) => {
    const col = document.createElement("div");
    col.className = "vx-col";
    col.dataset.position = String(position);
    if (c.kind !== null && c.kind !== undefined) {
      col.dataset.kind = c.kind;
      col.dataset.index = String(c.index);
    }

    const gutter = document.createElement("div");
    gutter.className = "vx-gutter";
    if (c.t !== null && c.t !== undefined) {
      const bar = document.createElement("div");
      bar.className = "vx-bar";
      // |t| in [0,1] -> half the gutter height; sign picks the half it
      // grows into, measured from the zero line at the gutter's middle.
      bar.style.height = `${Math.abs(c.t) * 50}%`;
      bar.style.background = c.color;
      bar.style[c.t >= 0 ? "bottom" : "top"] = "50%";
      gutter.appendChild(bar);
    }

    const glyph = document.createElement("div");
    glyph.className = "vx-char";
    glyph.textContent = c.char;
    if (c.color) glyph.style.borderBottomColor = c.color;
    if (c.kind === null || c.kind === undefined) glyph.classList.add("vx-inert");

    col.appendChild(gutter);
    col.appendChild(glyph);

    col.addEventListener("mouseenter", () =>
      onHover(c.kind ?? null, c.index ?? null),
    );
    col.addEventListener("mouseleave", () => onHover(null, null));
    strip.appendChild(col);
  });

  return strip;
}

/**
 * Columns in the strip that belong to a given element.
 *
 * @param {Element} strip
 * @param {string} kind
 * @param {number} index
 * @returns {Element[]}
 */
export function columnsFor(strip, kind, index) {
  return Array.from(
    strip.querySelectorAll(
      `.vx-col[data-kind="${kind}"][data-index="${index}"]`,
    ),
  );
}

export function render({ model, el }) {
  el.classList.add("vx-root");
  let current = null; // {kind, index} currently highlighted, or null

  const svgHost = document.createElement("div");
  svgHost.className = "vx-svg";
  const readout = document.createElement("div");
  readout.className = "vx-readout";
  readout.style.display = "none";

  el.replaceChildren(svgHost, readout);

  function draw() {
    const payload = model.get("payload") || {};
    const chars = payload.chars || [];
    svgHost.innerHTML = payload.svg || "";

    const strip = buildStrip(chars, (kind, index) => {
      highlight(kind === null ? null : { kind, index });
    });

    // Rebuilt on every payload change, so the old strip must go with it.
    const previous = el.querySelector(".vx-strip");
    if (previous) previous.remove();
    el.insertBefore(strip, readout);

    // Structure -> strip. Delegated from the host so it survives the
    // innerHTML swap above without re-binding per shape.
    svgHost.onmousemove = (ev) => {
      const id = identify(ev.target);
      highlight(id);
      if (id) position(ev);
    };
    svgHost.onmouseleave = () => highlight(null);

    highlight(null);
  }

  // Reads the `elements` map, NOT `chars`. The strip only holds what the
  // SMILES string spells, but the structure draws every scored atom and
  // bond -- most bonds are implicit and have no character. Looking up in
  // `chars` left the majority of colored bonds with no readout at all.
  function entryFor(kind, index) {
    const elements = (model.get("payload") || {}).elements || {};
    const byKind = elements[kind] || {};
    return byKind[String(index)] || null;
  }

  function highlight(id) {
    if (
      (current === null && id === null) ||
      (current && id && current.kind === id.kind && current.index === id.index)
    ) {
      return; // no change -- avoid churning the DOM on every mousemove
    }

    const strip = el.querySelector(".vx-strip");
    if (current) {
      clearHighlight(shapesFor(svgHost, current.kind, current.index));
      if (strip) {
        for (const c of columnsFor(strip, current.kind, current.index)) {
          c.classList.remove("vx-on");
        }
      }
    }

    current = id;
    model.set("hovered", id ? { kind: id.kind, index: id.index } : {});
    model.save_changes();

    if (!id) {
      // Leaving focus mode restores every column to full contrast.
      if (strip) strip.classList.remove("vx-focus");
      readout.style.display = "none";
      return;
    }

    applyHighlight(shapesFor(svgHost, id.kind, id.index));
    if (strip) {
      const columns = columnsFor(strip, id.kind, id.index);
      for (const c of columns) {
        c.classList.add("vx-on");
      }
      // Only dim the rest when there is actually a column to focus on --
      // an implicit bond has none, and fading the whole strip to say
      // "nothing here" would be worse than leaving it alone.
      strip.classList.toggle("vx-focus", columns.length > 0);
    }

    const entry = entryFor(id.kind, id.index);
    if (entry && entry.label) {
      const parts = [entry.label];
      // Show the raw score AND the normalized value driving the color, so
      // the readout reconciles with the legend whichever units it prints.
      if (entry.t !== null && entry.t !== undefined) {
        parts.push(`scaled ${entry.t >= 0 ? "+" : ""}${entry.t.toFixed(2)}`);
      }
      // How the score was assembled -- e.g. "1 bit, split 6 ways" vs
      // "2 tokens, each in full". Computed in Python (_format_provenance) so
      // the SVG tooltip and this readout can never word it differently.
      if (entry.provenance) {
        parts.push(entry.provenance);
      }
      // Explain the silence: an implicit bond has no SMILES character, so
      // nothing lights up in the strip. Saying so beats looking broken.
      if (entry.in_strip === false) {
        parts.push("no SMILES character");
      }
      readout.textContent = parts.join("   ·   ");
      readout.style.display = "block";
    } else {
      readout.style.display = "none";
    }
  }

  // Viewport coordinates, because the readout is position:fixed -- see the
  // CSS for why. Flips to the other side of the cursor near an edge so it
  // can't run off-screen.
  function position(ev) {
    const pad = 14;
    const width = readout.offsetWidth || 0;
    const height = readout.offsetHeight || 0;
    let x = ev.clientX + pad;
    let y = ev.clientY + pad;
    if (width && x + width > window.innerWidth) x = ev.clientX - width - pad;
    if (height && y + height > window.innerHeight) y = ev.clientY - height - pad;
    readout.style.left = `${Math.max(0, x)}px`;
    readout.style.top = `${Math.max(0, y)}px`;
  }

  // Keep the readout with the cursor even while it is over the strip.
  el.addEventListener("mousemove", (ev) => {
    if (readout.style.display !== "none") position(ev);
  });

  model.on("change:payload", draw);
  draw();

  return () => {
    model.off("change:payload", draw);
  };
}

export default { render };
