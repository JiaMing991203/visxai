"""Convenience adapter + loader for real pretrained Hugging Face checkpoints.

The sequence path's other modules already work against any model/tokenizer
satisfying their respective calling conventions:
:class:`~visxai.models.pytorch_wrapper.PyTorchSequenceWrapper` accepts any
``forward(input_ids, attention_mask) -> object with .logits``-shaped model
(which is exactly how a real Hugging Face model is called), and
:func:`~visxai.features.sequences.generate_sequence_representation` accepts
any :class:`~visxai.features.sequences.SMILESTokenizer`-satisfying callable.
Nothing here is a *new* wrapper class — it's purely a convenience layer for
the one genuinely missing piece: turning a real HF tokenizer into a
``SMILESTokenizer``, and bundling "load a checkpoint by name" into one call.

Per VisXAI's architecture principle (see the README), this module never
guesses a model-specific detail on the caller's behalf. In particular,
:func:`load_pretrained_sequence_model` requires an explicit ``model_class``
(e.g. ``transformers.AutoModelForSequenceClassification``) rather than
defaulting to one — a real checkpoint's task head varies (classification,
regression, masked-LM-only with no head at all), and picking one silently
would be exactly the kind of unverifiable scheme mismatch this package's
other "no default" parameters (``atom_featurizer``, ``target_layer``,
``embedding_layer``) already avoid.

**A real, version-specific gotcha found while building this**: on
``transformers`` versions whose default attention implementation is
``"sdpa"`` (the modern default for most architectures, including RoBERTa-
family checkpoints like ChemBERTa), requesting ``output_attentions=True``
(what :class:`~visxai.explainers.attention.AttentionExplainer` does)
currently triggers an automatic runtime fallback to the eager attention
implementation, with a `UserWarning` — verified this still correctly
returns real per-layer attention tensors, not empty/``None`` ones. But the
warning states this fallback is **removed in transformers v5.0.0**, at
which point requesting attentions from an ``"sdpa"``-implementation model
may raise instead of falling back. If you're using
:class:`~visxai.explainers.attention.AttentionExplainer` against a
real checkpoint, pass ``attn_implementation="eager"`` through
``**model_kwargs`` to :func:`load_pretrained_sequence_model` to sidestep
this entirely (not done by default here, since it's an opinionated choice
outside the scope of "just load the checkpoint").
"""

from __future__ import annotations

from typing import Any, Optional

from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from visxai.features.sequences import TokenizerOutput
from visxai.models.pytorch_wrapper import PyTorchSequenceWrapper


class HFTokenizerAdapter:
    """Adapts a real Hugging Face tokenizer to VisXAI's ``SMILESTokenizer`` protocol.

    Wraps ``tokenizer(smiles, return_offsets_mapping=True)`` and repackages
    its ``input_ids``/``attention_mask``/``offset_mapping`` into a
    :class:`~visxai.features.sequences.TokenizerOutput` — the exact contract
    :func:`~visxai.features.sequences.generate_sequence_representation`'s
    ``tokenizer`` parameter expects, so
    :func:`~visxai.utils.mapping.align_tokens_to_atoms` can derive
    ``token_to_atom_map``/``token_to_bond_map`` from this tokenizer's own
    character offsets exactly the same way it does for VisXAI's built-in
    reference tokenizer.

    Only valid, as with any :class:`~visxai.features.sequences.SMILESTokenizer`
    implementation, if this is actually the tokenizer the target model was
    trained on — VisXAI cannot verify that; a mismatch produces a
    meaningless explanation that may not even error (see the architecture
    principle in the README).

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizerBase
        A real Hugging Face tokenizer instance. Must be a "fast"
        (Rust-backed) tokenizer — ``return_offsets_mapping=True`` is not
        supported by "slow" (pure-Python) tokenizers.

    Attributes
    ----------
    _tokenizer : transformers.PreTrainedTokenizerBase
        The wrapped tokenizer.

    Raises
    ------
    ValueError
        If ``tokenizer.is_fast`` is ``False``.

    Examples
    --------
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    >>> adapter = HFTokenizerAdapter(tokenizer)
    >>> output = adapter("CCO")
    >>> output.token_ids  # doctest: +SKIP
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        if not tokenizer.is_fast:
            raise ValueError(
                "HFTokenizerAdapter requires a 'fast' (Rust-backed) tokenizer, "
                f"since return_offsets_mapping=True is not supported by 'slow' "
                f"tokenizers; got {type(tokenizer).__name__} (is_fast=False). "
                "Most checkpoints default to a fast tokenizer already; if not, "
                "pass use_fast=True to AutoTokenizer.from_pretrained()."
            )
        self._tokenizer: PreTrainedTokenizerBase = tokenizer

    def __call__(self, smiles: str) -> TokenizerOutput:
        """Tokenize ``smiles`` and return a :class:`TokenizerOutput`.

        Parameters
        ----------
        smiles : str
            Input SMILES string.

        Returns
        -------
        TokenizerOutput
            ``token_ids``/``token_offsets``/``attention_mask`` from the
            wrapped tokenizer's own ``return_offsets_mapping=True`` output.
            Special tokens (e.g. ``<s>``/``</s>``) get whatever zero-length
            offset the underlying tokenizer itself reports for them —
            typically ``(0, 0)``, which :func:`~visxai.utils.mapping.align_tokens_to_atoms`
            already treats as unattributable to any atom.
        """
        encoding = self._tokenizer(smiles, return_offsets_mapping=True)
        return TokenizerOutput(
            token_ids=list(encoding["input_ids"]),
            token_offsets=[tuple(span) for span in encoding["offset_mapping"]],
            attention_mask=list(encoding["attention_mask"]),
        )


def load_pretrained_sequence_model(
    checkpoint: str,
    model_class: type[PreTrainedModel],
    output_attr: str = "logits",
    tokenizer_kwargs: Optional[dict[str, Any]] = None,
    **model_kwargs: Any,
) -> tuple[PyTorchSequenceWrapper, HFTokenizerAdapter]:
    """Load a real pretrained checkpoint into a ready-to-use wrapper + tokenizer adapter.

    Convenience wrapper around ``model_class.from_pretrained(checkpoint,
    **model_kwargs)`` and ``AutoTokenizer.from_pretrained(checkpoint,
    **tokenizer_kwargs)``, bundled into the two objects every other VisXAI
    sequence-path call needs:
    :func:`~visxai.features.sequences.generate_sequence_representation`'s
    ``tokenizer`` parameter, and any sequence explainer's ``model`` parameter.

    Parameters
    ----------
    checkpoint : str
        A Hugging Face Hub model id (e.g. ``"seyonec/ChemBERTa-zinc-base-v1"``)
        or a local checkpoint directory path — anything
        ``from_pretrained`` accepts.
    model_class : type[transformers.PreTrainedModel]
        The Hugging Face model class to instantiate (e.g.
        ``transformers.AutoModelForSequenceClassification``). Required, with
        no default — a real checkpoint's task head varies (classification,
        regression, masked-LM-only with no head at all), and this package
        never guesses a model-specific detail on the caller's behalf (see
        the architecture principle in the README).
    output_attr : str, optional
        Passed through to
        :class:`~visxai.models.pytorch_wrapper.PyTorchSequenceWrapper`.
        Defaults to ``"logits"``, correct for essentially every standard HF
        model output type.
    tokenizer_kwargs : dict, optional
        Extra keyword arguments forwarded to
        ``AutoTokenizer.from_pretrained``. Defaults to ``None`` (no extras).
    **model_kwargs : Any
        Extra keyword arguments forwarded to ``model_class.from_pretrained``
        (e.g. ``num_labels=1`` for a regression head, or
        ``attn_implementation="eager"`` — see this module's docstring for
        why that one matters if you plan to use
        :class:`~visxai.explainers.attention.AttentionExplainer`).

    Returns
    -------
    tuple[PyTorchSequenceWrapper, HFTokenizerAdapter]
        ``(wrapper, tokenizer_adapter)`` — pass ``wrapper`` to any sequence
        explainer's ``explain()``, and ``tokenizer_adapter`` as
        :func:`~visxai.features.sequences.generate_sequence_representation`'s
        ``tokenizer`` argument when building the ``MoleculeRepresentation``
        for a molecule this model will explain.

    Raises
    ------
    ValueError
        If the loaded tokenizer is not a "fast" tokenizer (raised by
        :class:`HFTokenizerAdapter`).

    Examples
    --------
    >>> from transformers import AutoModelForSequenceClassification
    >>> from visxai.features.sequences import generate_sequence_representation
    >>> from visxai.explainers.attention import AttentionExplainer
    >>> wrapper, tokenizer = load_pretrained_sequence_model(
    ...     "seyonec/ChemBERTa-zinc-base-v1",
    ...     AutoModelForSequenceClassification,
    ...     num_labels=1,
    ... )  # doctest: +SKIP
    >>> mol_rep = generate_sequence_representation("CCO", tokenizer=tokenizer)  # doctest: +SKIP
    >>> explanation = AttentionExplainer().explain(wrapper, mol_rep)  # doctest: +SKIP
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, **(tokenizer_kwargs or {}))
    model = model_class.from_pretrained(checkpoint, **model_kwargs)

    wrapper = PyTorchSequenceWrapper(model, output_attr=output_attr)
    tokenizer_adapter = HFTokenizerAdapter(tokenizer)
    return wrapper, tokenizer_adapter
