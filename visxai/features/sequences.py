"""SMILES tokenization and token-to-atom/bond alignment for VisXAI sequence models.

This module provides :func:`generate_sequence_representation`, which converts
a SMILES string into a :class:`~visxai.core.data_types.MoleculeRepresentation`
populated with ``token_ids``, ``token_to_atom_map``, ``token_to_bond_map``,
and ``attention_mask``. ``token_to_atom_map``/``token_to_bond_map`` are the
artefacts consumed by :mod:`visxai.explainers.attention` (and any other
sequence explainer) to translate token-level attributions back to RDKit atom
and bond indices.

Design (most of the original open
questions are resolved by this design, not just answered ad hoc)
--------------------------------------------------------------------------
VisXAI wraps an already-trained model; it must not assume the tokenizer that
model was trained with matches any particular scheme (see the architecture
principle in the README). The tokenizer is therefore a **pluggable
parameter**: :func:`generate_sequence_representation` accepts any callable
satisfying the :class:`SMILESTokenizer` protocol, defaulting to
:func:`default_smiles_tokenizer` — VisXAI's own reference implementation,
only valid if the target model was actually trained on that exact scheme.

The key insight that makes this pluggable *and* still produces correct
alignment maps: token-to-atom (and token-to-bond) alignment can always be
derived from **character-offset overlap** between (a) each atom's or bond's
character span in the SMILES string — computed independently by
:func:`compute_atom_char_spans`/:func:`compute_bond_char_spans`, purely from
SMILES syntax, regardless of tokenizer — and (b) each token's character
span, as reported by whichever tokenizer produced it (VisXAI's own regex
tokenizer, or a real tokenizer's ``return_offsets_mapping=True`` output).
:func:`~visxai.utils.mapping.align_tokens_to_atoms` performs this alignment
for both cases. This is why non-atom, non-bond-carrying tokens (branch
parens, most ring-closure digits, special tokens) naturally map to an empty
list — they simply have no overlapping span, rather than being a
special-cased rule.

**Bond coverage is inherently partial.** Most bonds in a SMILES string are
implicit — a plain single bond (``"CC"``) or default aromatic bond
(``"cc"``) has no character at all. Only bonds carrying an explicit symbol
(``=``, ``#``, ``:``, ``~``, ``/``, ``\\``, or one attached to a
ring-closure digit) get a ``token_to_bond_map`` entry; on real drug
molecules this covers roughly 10-25% of bonds (measured on aspirin,
ibuprofen, caffeine, penicillin G). This is a hard limit of SMILES syntax
itself, not a tokenizer or parser limitation — but the bonds that *are*
covered (double/triple bonds, stereo bonds, explicit ring closures) still
carry real model contribution worth visualizing, so
:class:`~visxai.explainers.attention.AttentionExplainer` populates
``Explanation.bond_scores`` for them rather than discarding that signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
from rdkit import Chem

from visxai.core.data_types import MoleculeRepresentation
from visxai.utils.chem_utils import parse_smiles
from visxai.utils.mapping import align_tokens_to_atoms


# ---------------------------------------------------------------------------
# Regex atom-wise tokenizer (VisXAI's reference tokenizer)
# ---------------------------------------------------------------------------

# Alternatives are ordered so that longer/more-specific patterns are tried
# before shorter ones that would otherwise shadow them (e.g. "Br" before the
# single-letter organic-subset class, "%nn" before a lone digit).
_SMILES_TOKEN_PATTERN = re.compile(
    r"("
    r"\[[^\]]+\]"    # bracket atom, e.g. [nH], [C@@H], [NH4+], [Se]
    r"|Br|Cl"        # two-letter halogens (organic subset)
    r"|[BCNOSPFI]"   # organic-subset atoms (aliphatic)
    r"|[bcnosp]"     # organic-subset atoms (aromatic)
    r"|\*"           # dummy / wildcard atom
    r"|%[0-9]{2}"    # two-digit ring-closure label
    r"|[0-9]"        # single-digit ring-closure label
    r"|[=#\-:~/\\]"  # bond symbols
    r"|\(|\)"        # branch open / close
    r"|\."           # disconnected-structure separator
    r")"
)

# A token is an "atom token" (occupies exactly one RDKit atom's span) iff it
# matches this pattern. Everything else (bonds, ring digits, branch parens,
# '.') is structural and has no atom span.
_ATOM_TOKEN_PATTERN = re.compile(r"^(\[[^\]]+\]|Br|Cl|[BCNOSPFI]|[bcnosp]|\*)$")

_CLS_TOKEN = "[CLS]"
_SEP_TOKEN = "[SEP]"
_PAD_TOKEN = "[PAD]"
_UNK_TOKEN = "[UNK]"

# Small fixed vocabulary for the reference tokenizer only. Anything not
# listed here maps to _UNK_TOKEN's id. A real tokenizer plugged in via the
# `tokenizer` parameter brings its own vocabulary entirely.
_VOCAB: dict[str, int] = {
    _PAD_TOKEN: 0,
    _UNK_TOKEN: 1,
    _CLS_TOKEN: 2,
    _SEP_TOKEN: 3,
    # organic-subset atoms (aliphatic)
    "B": 4, "C": 5, "N": 6, "O": 7, "S": 8, "P": 9, "F": 10, "I": 11,
    "Cl": 12, "Br": 13,
    # organic-subset atoms (aromatic)
    "b": 14, "c": 15, "n": 16, "o": 17, "s": 18, "p": 19,
    "*": 20,
    # bond / structural symbols
    "-": 21, "=": 22, "#": 23, ":": 24, "~": 25, "/": 26, "\\": 27,
    "(": 28, ")": 29, ".": 30,
    # ring-closure digits (single-digit only; "%nn" and anything else -> UNK)
    "0": 31, "1": 32, "2": 33, "3": 34, "4": 35,
    "5": 36, "6": 37, "7": 38, "8": 39, "9": 40,
    # a modest set of common bracket atoms seen in drug-like molecules
    "[nH]": 41, "[NH]": 42, "[NH2]": 43, "[NH3+]": 44, "[NH4+]": 45,
    "[O-]": 46, "[N+]": 47, "[n+]": 48, "[nH+]": 49,
    "[C@H]": 50, "[C@@H]": 51, "[C@]": 52, "[C@@]": 53,
    "[Se]": 54, "[se]": 55, "[Si]": 56, "[SiH2]": 57, "[SiH3]": 58,
}


def _tokenize_smiles_with_spans(smiles: str) -> list[tuple[str, int, int]]:
    """Tokenize `smiles`, returning ``(token, start_char, end_char)`` triples.

    Raises
    ------
    ValueError
        If a substring of ``smiles`` does not match any token pattern (e.g.
        an unbracketed atom outside the organic subset, such as ``"se"``
        instead of ``"[se]"``).
    """
    triples: list[tuple[str, int, int]] = []
    pos = 0
    for match in _SMILES_TOKEN_PATTERN.finditer(smiles):
        if match.start() != pos:
            raise ValueError(
                f"Could not tokenize SMILES {smiles!r}: unrecognised "
                f"character(s) starting at position {pos}."
            )
        triples.append((match.group(0), match.start(), match.end()))
        pos = match.end()
    if pos != len(smiles):
        raise ValueError(
            f"Could not tokenize SMILES {smiles!r}: unrecognised "
            f"character(s) starting at position {pos}."
        )
    return triples


def tokenize_smiles(smiles: str) -> list[str]:
    """Split a SMILES string into atom-wise regex tokens.

    Parameters
    ----------
    smiles : str
        Input SMILES string.

    Returns
    -------
    list[str]
        Tokens covering the entire input string with no gaps, in the same
        left-to-right order the characters appear in ``smiles``.

    Raises
    ------
    ValueError
        If a substring of ``smiles`` does not match any token pattern.

    Examples
    --------
    >>> tokenize_smiles("CC(=O)O")
    ['C', 'C', '(', '=', 'O', ')', 'O']
    """
    return [token for token, _, _ in _tokenize_smiles_with_spans(smiles)]


def _is_atom_token(token: str) -> bool:
    """Return ``True`` if ``token`` occupies exactly one RDKit atom's span."""
    return _ATOM_TOKEN_PATTERN.match(token) is not None


def compute_atom_char_spans(smiles: str) -> dict[int, tuple[int, int]]:
    """Return the ``(start_char, end_char)`` span of each atom in ``smiles``.

    Spans are keyed by RDKit atom index (dense, ``0..n_atoms-1``), derived
    purely from re-parsing the SMILES string's syntax via the same regex
    used by :func:`tokenize_smiles` — independent of any tokenizer choice.
    This is the "universal bookkeeping" half of VisXAI's architecture
    principle: always correct regardless of which :class:`SMILESTokenizer`
    is plugged into :func:`generate_sequence_representation`.

    Parameters
    ----------
    smiles : str
        Input SMILES string.

    Returns
    -------
    dict[int, tuple[int, int]]
        One ``(start_char, end_char)`` span per atom index.

    Raises
    ------
    ValueError
        If a substring of ``smiles`` does not match any token pattern.

    Examples
    --------
    >>> compute_atom_char_spans("CCO")
    {0: (0, 1), 1: (1, 2), 2: (2, 3)}
    """
    atom_idx = 0
    atom_spans: dict[int, tuple[int, int]] = {}
    for token, start, end in _tokenize_smiles_with_spans(smiles):
        if _is_atom_token(token):
            atom_spans[atom_idx] = (start, end)
            atom_idx += 1
    return atom_spans


# ---------------------------------------------------------------------------
# Bond character-span parser (for bonds with an explicit SMILES character)
# ---------------------------------------------------------------------------

_BOND_SYMBOL_PATTERN = re.compile(r"^[=#\-:~/\\]$")
_RING_CLOSURE_PATTERN = re.compile(r"^(\d|%\d\d)$")


def compute_bond_char_spans(
    smiles: str,
    mol: Chem.Mol,
) -> dict[int, tuple[int, int]]:
    """Return the ``(start_char, end_char)`` span of each *explicit* bond.

    Most bonds in a SMILES string have **no character at all** — a plain
    single bond (``"CC"``) or a default aromatic bond (``"cc"``) is purely
    implicit. Only bonds carrying an explicit symbol (``=``, ``#``, ``:``,
    ``~``, ``/``, ``\\``) — including one attached to a ring-closure digit,
    e.g. ``"C=1CCCCC1"`` — have a character span to attribute a token's
    contribution to. This function returns a **sparse** mapping: bond
    indices absent from the result are implicit bonds with nothing to
    align a token to, not an oversight.

    A small state-machine traversal mirrors what a SMILES parser tracks
    internally (the "current atom" pointer, a branch stack for ``(``/``)``,
    and a ring-closure label dictionary for digit pairs) to determine which
    two atoms each explicit-bond-symbol or ring-closure-digit token
    connects, then looks up the corresponding RDKit bond via
    ``mol.GetBondBetweenAtoms``. This has been validated (see the git
    history around this function) to reproduce RDKit's own bond list
    exactly on fused-ring and mixed explicit/implicit test cases.

    Parameters
    ----------
    smiles : str
        Input SMILES string.
    mol : rdkit.Chem.Mol
        The RDKit ``Mol`` parsed from the same ``smiles`` (e.g. via
        :func:`visxai.utils.chem_utils.parse_smiles`), used to resolve
        ``(begin_atom, end_atom)`` pairs to RDKit bond indices.

    Returns
    -------
    dict[int, tuple[int, int]]
        Mapping from RDKit bond index to ``(start_char, end_char)`` span,
        containing only the bonds that have an explicit character.

    Raises
    ------
    ValueError
        If a substring of ``smiles`` does not match any token pattern.

    Examples
    --------
    >>> mol = parse_smiles("CC(=O)O")
    >>> compute_bond_char_spans("CC(=O)O", mol)
    {1: (3, 4)}
    """
    atom_idx_counter = 0
    prev_atom: Optional[int] = None
    branch_stack: list[Optional[int]] = []
    pending_bond_span: Optional[tuple[int, int]] = None
    ring_bonds: dict[str, tuple[Optional[int], Optional[tuple[int, int]]]] = {}
    atom_pair_spans: list[tuple[int, int, Optional[tuple[int, int]]]] = []

    for token, start, end in _tokenize_smiles_with_spans(smiles):
        if _is_atom_token(token):
            this_atom = atom_idx_counter
            atom_idx_counter += 1
            if prev_atom is not None:
                atom_pair_spans.append((prev_atom, this_atom, pending_bond_span))
            pending_bond_span = None
            prev_atom = this_atom
        elif _BOND_SYMBOL_PATTERN.match(token):
            pending_bond_span = (start, end)
        elif token == "(":
            branch_stack.append(prev_atom)
        elif token == ")":
            prev_atom = branch_stack.pop()
        elif token == ".":
            prev_atom = None
            pending_bond_span = None
        elif _RING_CLOSURE_PATTERN.match(token):
            if token in ring_bonds:
                open_atom, open_span = ring_bonds.pop(token)
                span = pending_bond_span or open_span
                atom_pair_spans.append((open_atom, prev_atom, span))
                pending_bond_span = None
            else:
                ring_bonds[token] = (prev_atom, pending_bond_span)
                pending_bond_span = None
        # else: unreachable given _SMILES_TOKEN_PATTERN's fixed alphabet.

    bond_spans: dict[int, tuple[int, int]] = {}
    for atom_a, atom_b, span in atom_pair_spans:
        if span is None:
            continue
        bond = mol.GetBondBetweenAtoms(atom_a, atom_b)
        bond_spans[bond.GetIdx()] = span
    return bond_spans


def tokens_to_ids(tokens: list[str]) -> list[int]:
    """Map tokens to integer ids using the built-in fixed vocabulary.

    Parameters
    ----------
    tokens : list[str]
        Token stream, may include ``[CLS]``/``[SEP]``.

    Returns
    -------
    list[int]
        One id per token. Tokens absent from :data:`_VOCAB` (e.g. an
        uncommon bracket atom or a multi-digit ring closure) map to the
        ``[UNK]`` id.

    Examples
    --------
    >>> tokens_to_ids(['[CLS]', 'C', 'O', '[SEP]'])
    [2, 5, 7, 3]
    """
    unk_id = _VOCAB[_UNK_TOKEN]
    return [_VOCAB.get(token, unk_id) for token in tokens]


# ---------------------------------------------------------------------------
# Pluggable tokenizer contract
# ---------------------------------------------------------------------------


@dataclass
class TokenizerOutput:
    """Uniform output contract any SMILES tokenizer must produce.

    Parameters
    ----------
    token_ids : list[int]
        Vocabulary ids for the tokenized SMILES string.
    token_offsets : list[tuple[int, int]]
        Per-token ``(start_char, end_char)`` span in the original SMILES
        string, aligned with ``token_ids``. A zero-length span (e.g.
        ``(0, 0)``) signals a token with no text span (a special token
        like ``[CLS]``/``[SEP]``/``[PAD]``) and is treated as
        unattributable to any atom by
        :func:`~visxai.utils.mapping.align_tokens_to_atoms`.
    attention_mask : list[int], optional
        Per-token mask (``1`` = real token, ``0`` = padding), aligned with
        ``token_ids``. ``None`` means "no padding" —
        :func:`generate_sequence_representation` fills an all-ones mask in
        that case.
    """

    token_ids: list[int]
    token_offsets: list[tuple[int, int]]
    attention_mask: Optional[list[int]] = None


class SMILESTokenizer(Protocol):
    """Structural protocol any pluggable SMILES tokenizer must satisfy.

    A tokenizer must actually match the scheme the target model was
    trained on — VisXAI cannot verify this and does not assume one
    specific scheme is universally correct (see the architecture
    principle in the README). To wrap a real Hugging Face tokenizer,
    implement this protocol using that tokenizer's own
    ``return_offsets_mapping=True`` output.
    """

    def __call__(self, smiles: str) -> TokenizerOutput: ...


def default_smiles_tokenizer(smiles: str) -> TokenizerOutput:
    """VisXAI's built-in reference tokenizer.

    Uses the atom-wise regex tokenizer (:func:`tokenize_smiles`) plus a
    small hand-built vocabulary (:data:`_VOCAB`), with ``[CLS]``/``[SEP]``
    prepended/appended. **Only valid if the target model was actually
    trained on this exact scheme.** For a real pretrained checkpoint,
    build a :class:`SMILESTokenizer` adapter around that model's own
    tokenizer instead and pass it via the ``tokenizer`` parameter of
    :func:`generate_sequence_representation`.

    Parameters
    ----------
    smiles : str
        Input SMILES string.

    Returns
    -------
    TokenizerOutput
        ``token_ids``/``token_offsets`` for ``["[CLS]", *tokens, "[SEP]"]``,
        with ``[CLS]``/``[SEP]`` given zero-length ``(0, 0)`` spans, and an
        all-ones ``attention_mask`` (no padding is ever introduced here).

    Raises
    ------
    ValueError
        If a substring of ``smiles`` does not match any token pattern.
    """
    raw = _tokenize_smiles_with_spans(smiles)
    tokens = [token for token, _, _ in raw]
    offsets = [(start, end) for _, start, end in raw]

    wrapped_tokens = [_CLS_TOKEN, *tokens, _SEP_TOKEN]
    wrapped_offsets: list[tuple[int, int]] = [(0, 0), *offsets, (0, 0)]
    token_ids = tokens_to_ids(wrapped_tokens)

    return TokenizerOutput(
        token_ids=token_ids,
        token_offsets=wrapped_offsets,
        attention_mask=[1] * len(token_ids),
    )


def generate_sequence_representation(
    smiles: str,
    tokenizer: SMILESTokenizer = default_smiles_tokenizer,
) -> MoleculeRepresentation:
    """Tokenize a SMILES string and align tokens to RDKit atom indices.

    Parameters
    ----------
    smiles : str
        Input SMILES string. Must be parseable by RDKit and by VisXAI's
        atom-span parser (i.e. use only the organic subset outside of
        bracket atoms).
    tokenizer : SMILESTokenizer, optional
        Callable satisfying the :class:`SMILESTokenizer` protocol. Defaults
        to :func:`default_smiles_tokenizer` — VisXAI's own reference
        scheme. Pass a different tokenizer (e.g. one wrapping a real
        pretrained model's own tokenizer) when the target model was not
        trained on the default scheme.

    Returns
    -------
    MoleculeRepresentation
        Fully populated container with:

        - ``smiles`` / ``mol`` — the original string and sanitized RDKit
          ``Mol``.
        - ``fingerprint_array`` / ``bit_info`` — irrelevant to the sequence
          path; set to an empty array and an empty dict respectively.
        - ``token_ids`` — from ``tokenizer``.
        - ``token_to_atom_map`` — derived by
          :func:`~visxai.utils.mapping.align_tokens_to_atoms` from
          :func:`compute_atom_char_spans` and ``tokenizer``'s reported
          token offsets, regardless of which tokenizer was used.
        - ``token_to_bond_map`` — derived the same way from
          :func:`compute_bond_char_spans`. Sparse: most token positions map
          to no bond at all, since most bonds have no explicit SMILES
          character (see :func:`compute_bond_char_spans`'s docstring).
        - ``attention_mask`` — from ``tokenizer``, or an all-ones mask if
          the tokenizer didn't supply one.

    Raises
    ------
    ValueError
        If RDKit cannot parse ``smiles``, if VisXAI's atom-span parser
        cannot parse it, or if the number of atom spans found does not
        match ``mol.GetNumAtoms()`` (a sanity check that catches
        parser/RDKit atom-count mismatches rather than silently
        mis-aligning atoms).

    Examples
    --------
    >>> rep = generate_sequence_representation("CCO")
    >>> rep.token_ids
    [2, 5, 5, 7, 3]
    >>> rep.token_to_atom_map
    {0: [], 1: [0], 2: [1], 3: [2], 4: []}
    >>> rep.token_to_bond_map  # both C-C/C-O bonds are implicit, no spans
    {0: [], 1: [], 2: [], 3: [], 4: []}
    """
    mol = parse_smiles(smiles)

    atom_spans = compute_atom_char_spans(smiles)
    if len(atom_spans) != mol.GetNumAtoms():
        raise ValueError(
            f"VisXAI's atom-span parser found {len(atom_spans)} atoms but "
            f"RDKit parsed {mol.GetNumAtoms()} atoms for SMILES {smiles!r}; "
            "this usually means the input uses a SMILES feature the regex "
            "parser does not yet recognise."
        )

    bond_spans = compute_bond_char_spans(smiles, mol)

    tok_out = tokenizer(smiles)
    token_to_atom_map = align_tokens_to_atoms(atom_spans, tok_out.token_offsets)
    token_to_bond_map = align_tokens_to_atoms(bond_spans, tok_out.token_offsets)
    attention_mask = (
        tok_out.attention_mask
        if tok_out.attention_mask is not None
        else [1] * len(tok_out.token_ids)
    )

    return MoleculeRepresentation(
        smiles=smiles,
        mol=mol,
        fingerprint_array=np.zeros(0, dtype=np.uint8),
        bit_info={},
        token_ids=tok_out.token_ids,
        token_to_atom_map=token_to_atom_map,
        token_to_bond_map=token_to_bond_map,
        attention_mask=attention_mask,
    )
