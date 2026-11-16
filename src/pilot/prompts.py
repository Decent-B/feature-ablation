"""Chat templating, the post-instruction suffix (intervention positions) and answer tokens."""

from __future__ import annotations

import re

PROBE = "PILOT_PROBE_CONTENT_7f3a"


def render(tok, user_text: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def encode(tok, user_text: str) -> list[int]:
    return tok(render(tok, user_text), add_special_tokens=False)["input_ids"]


def template_suffix(tok) -> dict:
    """Everything the template appends after the user content. Its tokens are the
    intervention positions; the last one is the final prompt token."""
    text = render(tok, PROBE)
    j = text.rindex(PROBE) + len(PROBE)
    suffix_text = text[j:]
    suffix_ids = tok(suffix_text, add_special_tokens=False)["input_ids"]
    full = tok(text, add_special_tokens=False)["input_ids"]
    if full[-len(suffix_ids):] != suffix_ids:
        raise RuntimeError("template suffix does not tokenize independently of the content")
    return {"suffix_text": suffix_text, "suffix_ids": suffix_ids, "T": len(suffix_ids),
            "suffix_tokens": tok.convert_ids_to_tokens(suffix_ids),
            "has_empty_think_block": "<think>" in suffix_text and "</think>" in suffix_text}


def encode_checked(tok, user_text: str, suffix_ids: list[int]) -> list[int]:
    ids = encode(tok, user_text)
    if ids[-len(suffix_ids):] != suffix_ids:
        raise RuntimeError("prompt does not end with the template suffix; positions would shift")
    return ids


def answer_variants(tok, words=("True", "False")) -> dict:
    """First token id of each answer word, without and with a leading space."""
    out = {}
    for name, prefix in (("bare", ""), ("space", " ")):
        ids = [tok.encode(prefix + w, add_special_tokens=False)[0] for w in words]
        out[name] = {"ids": ids, "tokens": tok.convert_ids_to_tokens(ids), "distinct": ids[0] != ids[1]}
    return out


_ANSWER_RE = re.compile(r"^\s*(true|false)\b", re.IGNORECASE)


def starts_with_answer(text: str) -> bool:
    return bool(_ANSWER_RE.match(text)) and not text.lstrip().startswith("<think>")


def starts_with_thinking(text: str) -> bool:
    return text.lstrip().startswith("<think>")
