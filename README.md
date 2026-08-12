# VisXAI

**VisXAI** maps model-interpretation algorithms — SHAP, Integrated
Gradients, Grad-CAM, attention — back onto molecular structures (atoms,
bonds) for drug-discovery / molecular-property-prediction models. Instead
of a bar chart of feature importances, you get a 2D structure diagram
colored by how much each atom and bond contributed to a prediction.

It supports three model paradigms, each with its own feature
representation, model wrapper, and explainer(s):

| Paradigm | Typical models | Feature representation | Explainer(s) |
|---|---|---|---|
| **Tree / Tabular** | Random Forest, XGBoost | Morgan or MACCS fingerprints | `TreeSHAPExplainer` (SHAP) |
| **Sequence** | ChemBERTa-style Transformers, 1D CNNs | Tokenized SMILES | `AttentionExplainer`, `IntegratedGradientsExplainer`, `GradCAMExplainer` |
| **Graph** | GNNs (PyTorch Geometric) | Node/edge feature graphs | `IntegratedGradientsExplainer`, `GradCAMExplainer` |

All three paths converge on the same output type (an `Explanation` with
per-atom and per-bond scores) and the same 2D renderer
(`RDKitSVGVisualizer`), so switching between them — or comparing two
explainers on the same molecule — doesn't require learning a new API.

---

## Table of contents

- [Why this exists: the mapping problem](#why-this-exists-the-mapping-problem)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Interactive visualization](#interactive-visualization)
- [The algorithms, briefly](#the-algorithms-briefly)
- [Architecture](#architecture)
- [Design principle: VisXAI wraps a model, it doesn't invent one](#design-principle-visxai-wraps-a-model-it-doesnt-invent-one)
- [Examples](#examples)
- [Repository layout](#repository-layout)
- [Current status](#current-status)
- [License](#license)

---

## Why this exists: the mapping problem

Every interpretation algorithm here operates on whatever numeric
representation the model actually consumes — a fingerprint bit, a graph
node, a SMILES token — not on atoms directly. Turning "bit 1088 mattered"
or "token position 7 mattered" into "*this* atom and *this* bond mattered"
is a non-trivial translation step, and it's different for every feature
scheme:

- **Fingerprint bit → atoms/bonds**: a Morgan bit represents a whole
  circular substructure (possibly matched by more than one environment in
  the molecule); a MACCS bit represents a predefined SMARTS pattern. Both
  get resolved to the exact atom/bond indices they touched, and a
  multi-environment bit's score is split evenly across every atom/bond
  index it touched.
- **SMILES token → atom/bond**: a tokenizer's character offsets are
  aligned against each atom's and bond's own character span in the
  SMILES string, so token-level scores land on the right atom regardless
  of which tokenizer produced them.
- **Graph node/edge → atom/bond**: this one is easy by construction — node
  `i` is built to always be atom `i`, and directed edges `2k`/`2k+1` are
  always the two directions of bond `k`.

This translation layer (`visxai/utils/mapping.py`) is the actual reason
this package exists, more than any single explainer implementation — the
XAI algorithms themselves are largely off-the-shelf (SHAP, Captum), but
getting their output onto the right atom is not.

---

## Installation

Requires Python ≥ 3.9.

**Option 1 — everything, pinned versions (fastest way to run the example
notebooks):**

```bash
git clone <this-repo-url>
cd visxai
pip install -r requirements.txt
```

**Option 2 — editable install with `pyproject.toml` extras**, if you only
need a subset of paradigms:

```bash
pip install -e .                # core only: tree/tabular path (rdkit, shap, scikit-learn)
pip install -e ".[sequence]"    # + torch, captum — sequence path
pip install -e ".[graph]"       # + torch, torch_geometric, captum — graph path
pip install -e ".[viz]"         # + ipywidgets, anywidget — interactive notebook widgets
pip install -e ".[dev]"         # everything above + notebook tooling (nbformat/nbclient/ipykernel)
```

The core dependency set (`rdkit`, `shap`, `numpy`, `scikit-learn`) is
enough for the tree/tabular path on its own. The sequence and graph paths
each pull in `torch` (+ `captum` for gradient-based explainers on either
path); the graph path additionally needs `torch_geometric`. The `viz`
extra is only needed for the interactive widgets described
[below](#interactive-visualization) — static SVG output needs nothing
beyond the core set.

> **`ipywidgets` must be version 8 or newer**, which is why the `viz`
> extra pins `>=8.0`. An `ipywidgets` 7 kernel renders a **blank output
> area** on JupyterLab 4 / Notebook 7 — silently, with no error — because
> the two major versions speak different frontend protocols. If a widget
> doesn't appear, check this pairing before anything else.

**Not required by anything in this repo today:** `xgboost` and
`transformers` (any scikit-learn-compatible estimator or
HuggingFace-convention `forward(input_ids, attention_mask)` model works
without them — see [Quickstart](#quickstart)), and `py3Dmol` (3D
visualization isn't implemented yet).

**To run the example notebooks**, register a Jupyter kernel against
whichever interpreter you installed into:

```bash
python -m ipykernel install --user --name python3
```

---

## Quickstart

Every path follows the same four-step shape: **featurize → wrap the model
→ explain → visualize.**

### Tree / Tabular (Morgan or MACCS fingerprints)

```python
from sklearn.ensemble import RandomForestClassifier
from visxai.features.fingerprints import generate_morgan_representation
from visxai.models.sklearn_wrapper import SklearnModelWrapper
from visxai.explainers.tree_shap import TreeSHAPExplainer
from visxai.visualizers.rdkit_2d import RDKitSVGVisualizer

mol_rep = generate_morgan_representation("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
wrapper = SklearnModelWrapper(my_trained_sklearn_model)  # any .predict(X)-compatible estimator
explanation = TreeSHAPExplainer().explain(wrapper, mol_rep)

svg = RDKitSVGVisualizer().visualize(mol_rep, explanation)
```

Swap in `generate_maccs_representation(smiles)` for MACCS keys instead —
same downstream pipeline, no other code changes.

### Sequence (tokenized SMILES)

```python
from visxai.features.sequences import generate_sequence_representation
from visxai.models.pytorch_wrapper import PyTorchSequenceWrapper
from visxai.explainers.attention import AttentionExplainer
from visxai.visualizers.rdkit_2d import RDKitSVGVisualizer

mol_rep = generate_sequence_representation("CC(=O)Oc1ccccc1C(=O)O")
wrapper = PyTorchSequenceWrapper(my_trained_transformer)  # forward(input_ids, attention_mask)
explanation = AttentionExplainer().explain(wrapper, mol_rep)

svg = RDKitSVGVisualizer().visualize(mol_rep, explanation)
```

`IntegratedGradientsExplainer`/`GradCAMExplainer` also work here — import
them from `visxai.explainers.gradient_based_sequence` (see the next
section for why the module matters, since these class names are
intentionally reused across paths).

**Using a real pretrained checkpoint** (e.g. ChemBERTa) instead of a
hand-rolled model: `visxai.models.hf_wrapper` loads one by name and hands
back both a ready `PyTorchSequenceWrapper` and a tokenizer adapter in one
call.

```python
from transformers import AutoModelForSequenceClassification
from visxai.models.hf_wrapper import load_pretrained_sequence_model

wrapper, tokenizer = load_pretrained_sequence_model(
    "seyonec/ChemBERTa-zinc-base-v1", AutoModelForSequenceClassification, num_labels=1
)
mol_rep = generate_sequence_representation("CC(=O)Oc1ccccc1C(=O)O", tokenizer=tokenizer)
explanation = AttentionExplainer().explain(wrapper, mol_rep)
```

### Graph (PyTorch Geometric)

```python
from visxai.features.graphs import generate_graph_representation
from visxai.models.pytorch_wrapper import PyTorchGNNWrapper
from visxai.explainers.gradient_based import IntegratedGradientsExplainer
from visxai.visualizers.rdkit_2d import RDKitSVGVisualizer

def atom_featurizer(atom):
    return [atom.GetAtomicNum(), atom.GetDegree()]  # must match your model's training scheme

mol_rep = generate_graph_representation("CC(=O)Oc1ccccc1C(=O)O", atom_featurizer)
wrapper = PyTorchGNNWrapper(my_trained_gnn)  # forward(x, edge_index, batch)
explanation = IntegratedGradientsExplainer().explain(wrapper, mol_rep)

svg = RDKitSVGVisualizer().visualize(mol_rep, explanation)
```

> **Note on the two `IntegratedGradientsExplainer`/`GradCAMExplainer` pairs:**
> the sequence path (`visxai.explainers.gradient_based_sequence`) and the
> graph path (`visxai.explainers.gradient_based`) each define their own
> class of the same name, adapted to that path's model shape (token
> sequences vs. graphs). Import from the module matching your model type.

`svg` is a self-contained SVG string with a color-bar legend built in —
`display(SVG(svg))` in a notebook, or write it straight to a `.svg` file.

---

## Interactive visualization

Requires the `viz` extra (`pip install -e ".[viz]"`) and a **live kernel**.
A notebook rendered on GitHub or nbviewer shows a blank area where the
widget would be.

### Hover tooltips on the static SVG

The cheapest option needs no extra dependency at all — `show_tooltips=True`
injects a native SVG `<title>` into every scored atom and bond:

```python
svg = RDKitSVGVisualizer(show_tooltips=True).visualize(mol_rep, explanation)
```

Be aware this uses the browser's own tooltip timer: roughly a second of
*stationary* hover, unstyled, with no highlight. Sweeping the pointer shows
nothing. That is the native behaviour, not a fault in the output.

### Hover-linked widget

`MoleculeHoverWidget` pairs the structure with a SMILES strip — one score
bar per character — and cross-highlights as the cursor moves, with a
readout that follows it:

```python
from visxai.visualizers.hover_widget import MoleculeHoverWidget

MoleculeHoverWidget.from_explanation(mol_rep, explanation)
```

### Comparing explanations side by side

`ExplanationDashboard` puts two such widgets next to each other on a shared
color scale, with dropdowns for molecule and explainer:

```python
from visxai.visualizers.interactive import ExplanationDashboard

dashboard = ExplanationDashboard.from_results({
    "aspirin": (mol_rep, {"TreeSHAP": explanation, "TreeSHAP (split)": split_explanation}),
})
dashboard.display()
```

It **re-renders explanations you already computed and never re-runs an
explainer**, so switching panels is instant regardless of how expensive the
attribution was. It also works for all three paradigms without
paradigm-specific code, since it only ever touches `MoleculeRepresentation`
and `Explanation`.

### Reading a score's provenance

An atom's score is a *sum of shares* from every source that touched it, and
the readout reports how that sum was reached:

```
Atom 3 (O): score -0.4200   ·   1 bit, split 6 ways   ·   scaled -1.00
```

That last detail matters because the paradigms do not share a convention.
The fingerprint path **splits** — a bit's score is divided among the atoms
of the substructure that triggered it. The sequence path **duplicates** — a
token hands its whole score to every atom it overlaps, undivided, so the
atom scores sum to more than the token scores did. The graph path does
neither, because node *i* simply is atom *i*.

Available programmatically too, on any explanation from the tree or
sequence paths:

```python
explanation.atom_provenance[3]
# [{'source_kind': 'bit', 'source_index': 314, 'source_score': -2.52,
#   'shared_among': 6, 'contribution': -0.42}]
```

`shared_among` is the divisor that was applied: `1` means the score arrived
undivided, anything larger means it was split that many ways.

---

## The algorithms, briefly

### TreeSHAP (SHAP — Lundberg & Lee, 2017; TreeSHAP — Lundberg et al., 2020)

**SHAP** (SHapley Additive exPlanations) assigns each input feature a
share of a prediction based on Shapley values from cooperative game
theory: roughly, "how much does the prediction change, on average, across
every possible subset of features, when this one feature is added?" This
gives a fair, theoretically-grounded credit split, but is normally
exponential to compute exactly. **TreeSHAP** is a polynomial-time
algorithm specific to tree ensembles (Random Forest, XGBoost, etc.) that
computes exact Shapley values by exploiting the tree structure. Here, the
"features" are fingerprint bits; `TreeSHAPExplainer` runs TreeSHAP via the
`shap` package, then redistributes each bit's score onto the atoms (and
bonds) that bit's substructure environment actually touched.

### Integrated Gradients (Sundararajan, Taly & Yan, 2017)

**Integrated Gradients (IG)** attributes a prediction by integrating the
model's gradient along a straight-line path from a *baseline* input
(e.g. all-zero features, representing "absence") to the real input,
accumulating how much each input dimension's gradient contributed along
the way. Unlike a single raw gradient, this satisfies a **completeness
axiom**: the sum of all attributions equals `prediction(input) -
prediction(baseline)` exactly (up to integration approximation error) —
a useful sanity check, and one the project verifies
numerically rather than just asserting. IG works on any differentiable
input, which is why it applies to both the graph path (node/edge features)
and the sequence path (token embeddings, via Captum's
`LayerIntegratedGradients`).

### Grad-CAM (Selvaraju et al., 2017; graph adaptation — Pope et al., 2019)

**Grad-CAM** (Gradient-weighted Class Activation Mapping) was originally
designed for CNNs on images: it takes a specific layer's activations,
weights each channel by how much that channel's *average gradient*
affects the output, and produces a heatmap from the weighted, ReLU'd
sum. This repo implements the graph- and sequence-adapted versions
by hand (not via Captum's default `LayerGradCam`, which doesn't share
weights across nodes/positions the way the original algorithm's
global-average-pooling step does) — one score per graph node or per
token position, non-negative, reflecting a *specific layer's* learned
representation rather than a full end-to-end attribution.

### Attention weights (+ Attention Rollout — Abnar & Zuidema, 2020)

The simplest of the four: for a Transformer-style model, just read off
its own softmax attention weights. `AttentionExplainer` offers two
aggregation strategies — the last layer's attention (head-averaged) from
a summarizing query token, or full **Attention Rollout** across every
layer, which accounts for how attention composes through the residual
stream rather than looking at one layer in isolation. Since attention
weights are a real (if debated) part of what the model computed, this is
the cheapest explainer here to run — no gradients or interpolation loop
needed.

### Fingerprints as a feature representation (Morgan / MACCS)

Not an XAI algorithm itself, but worth knowing: the tree path doesn't
work on raw atoms, it works on **fingerprints** — fixed-length bit
vectors summarizing a molecule's substructures. **Morgan fingerprints**
(a.k.a. circular/ECFP-style) hash arbitrarily-sized circular atom
environments into a configurable number of bits (`n_bits`, default
`2048` in this repo's demos). **MACCS keys** are a fixed, predefined
166-bit vocabulary (e.g. "has a 6-membered ring") — smaller,
human-interpretable, but less expressive than a large Morgan vector.
Both are supported identically by the rest of the pipeline.

---

## Architecture

Everything passes through two dataclasses (`visxai/core/data_types.py`):

- **`MoleculeRepresentation`** — every featurized view of one molecule:
  SMILES, RDKit `Mol`, the feature representation itself (fingerprint
  array / graph / token ids), and the bit-to-atom / token-to-atom mapping
  metadata needed to translate an explanation back to atom/bond indices.
- **`Explanation`** — the universal XAI output: `atom_scores`,
  `bond_scores` (optional), and free-form `metadata` (e.g. the model's
  raw prediction).

Three abstract base classes (`visxai/core/`) define the plugin contract
every implementation follows, so any wrapper/explainer/visualizer
combination on the same paradigm is interchangeable:

```python
BaseModelWrapper.predict(mol_rep) -> np.ndarray
BaseExplainer.explain(model, mol_rep) -> Explanation
BaseVisualizer.visualize(mol_rep, explanation) -> str  # SVG
```

```
SMILES string
     │
     ▼
feature extractor          (visxai/features/*.py)
     │  fingerprints.py  -> Morgan/MACCS bit vector + bit_info
     │  sequences.py     -> token ids + token_to_atom_map
     │  graphs.py        -> PyG Data + node i == atom i
     ▼
MoleculeRepresentation
     │
     ▼
model wrapper               (visxai/models/*.py)
     │  SklearnModelWrapper / PyTorchSequenceWrapper / PyTorchGNNWrapper
     ▼
explainer                   (visxai/explainers/*.py)
     │  TreeSHAPExplainer / AttentionExplainer /
     │  IntegratedGradientsExplainer / GradCAMExplainer
     ▼
Explanation (atom_scores + bond_scores + metadata)
     │
     ▼
RDKitSVGVisualizer.visualize()  ->  colored 2D SVG + legend
```

---

## Design principle: VisXAI wraps a model, it doesn't invent one

VisXAI explains an **already-trained** model; it does not choose the
feature scheme that model was trained on. Every feature-extraction module
separates two concerns:

1. **Numeric feature content** — what a feature vector's columns mean,
   which tokenizer/vocabulary is used, which atom/edge features a GNN
   expects. This must come from *you*, since VisXAI has no way to verify
   it matches your specific model's training-time scheme — a mismatch
   here produces a meaningless explanation that may not even raise an
   error.
2. **Index/alignment bookkeeping** — node `i` = atom `i`; which SMILES
   characters belong to which atom; which environment a fingerprint bit
   matched. This is universal regardless of feature content, and is the
   part VisXAI actually handles for you.

This is why `generate_sequence_representation`'s tokenizer is a pluggable
parameter (not a fixed scheme) and why `generate_graph_representation`'s
`atom_featurizer`/`bond_featurizer` are **required** arguments with no
default at all — there's no single dominant convention for GNN node/edge
features the way there is for fingerprints (Morgan/MACCS are the one
acknowledged exception, since they're the dominant real-world convention
for tree-based cheminformatics models).

---

## Examples

Six runnable Jupyter notebooks in `examples/`. The first three, one per
paradigm, each train a small real model (actual gradient descent, not
just random weights, for the sequence/graph paths) on a synthetic
aromaticity-detection task and then explain a prediction on aspirin —
fully offline, no network access needed. Each closes with an interactive
dashboard built from the explanations it just computed:

- **`examples/tree_demo.ipynb`** — Morgan fingerprints and MACCS keys →
  RandomForest → `TreeSHAPExplainer`, including a `bond_score_mode`
  comparison (`"duplicate"` vs. `"split"`) worked through on a single
  fingerprint bit.
- **`examples/sequence_demo.ipynb`** — a tiny attention model explained
  with `AttentionExplainer` (both aggregation strategies), plus a second
  small CNN explained with `IntegratedGradientsExplainer` and
  `GradCAMExplainer`.
- **`examples/graph_demo.ipynb`** — a tiny message-passing GNN explained
  with both `IntegratedGradientsExplainer` and `GradCAMExplainer`, atom-
  and bond-level, including the `uses_edge_attr` opt-in/opt-out contrast.

Three further notebooks are different on purpose:

- **`examples/logp_usecase.ipynb`** — asks whether the colors are *right*,
  rather than how to produce them. Crippen logP is defined as a sum of
  per-atom contributions, so RDKit supplies ground truth at exactly the
  granularity VisXAI produces. A graph network and a fingerprint forest are
  put to the same test on 900 molecules that ship with RDKit (no download).
  Both recover the reference only partially, and the model that predicts far
  better does not explain better — worth reading before trusting an
  atom-level number. Closes with a dashboard putting the best, median and
  worst agreement cases next to the reference itself, which can occupy a
  panel because ground truth expressed per atom is the same kind of object as
  a model explanation. Fully offline.
- **`examples/dashboard_demo.ipynb`** — the interactive layer on its own:
  `ExplanationDashboard`, the hover-linked widget, the color-scale modes,
  and how a widget-based notebook can still be verified without a browser.
- **`examples/hf_demo.ipynb`** — loads a real, publicly-hosted pretrained
  checkpoint (`seyonec/ChemBERTa-zinc-base-v1`) via `visxai.models.hf_wrapper`
  and explains aspirin with it, using `AttentionExplainer` and
  `IntegratedGradientsExplainer`. **Requires internet access** (downloads
  the checkpoint on first run) — unlike the other four, which never touch
  the network.

Open any of them with `jupyter lab examples/` after installing the `dev`
extra (or `requirements.txt`) and registering a kernel (see
[Installation](#installation)).

---

## Repository layout

```
visxai/
├── core/            # Abstract base classes + the two shared dataclasses
├── features/        # SMILES -> MoleculeRepresentation (fingerprints, sequences, graphs)
├── models/          # Uniform model-wrapper interface per framework
├── explainers/       # TreeSHAP, attention, Integrated Gradients, Grad-CAM
├── visualizers/      # 2D SVG renderer, hover-linked widget, comparison dashboard
│   └── static/      # The widget's frontend (ES module + CSS)
└── utils/           # The bit/token/node -> atom/bond mapping layer, SMILES I/O
examples/            # One notebook per paradigm, plus the dashboard and HF demos
```

---

## Current status

All three paradigms (tree/tabular, sequence, graph) are implemented,
tested, and demonstrated end-to-end, as is the interactive visualization
layer (hover tooltips, the hover-linked widget, and the comparison
dashboard). Not yet built:

- 3D visualization (`visualizers/py3dmol_3d.py`).
- `visxai/utils/graph_utils.py` (deferred until a concrete need emerges).
- Saving an interactive dashboard to a file. `visualize()` returns an SVG
  string you can write yourself, but there is no export path for a live
  widget.

One limitation worth stating plainly: **nothing checks appearance
automatically.** The widget's hover and linking logic is verified
headlessly, with no layout engine, so colors, bar geometry, and readout placement
are verified as the values the code sets — never as rendered pixels.

## License

[MIT](LICENSE).
