"""Shared handling of model replies that must be exactly one JSON object."""

from __future__ import annotations

import re


_SINGLE_FENCE = re.compile(r"^```[A-Za-z]*[ \t]*\r?\n?(?P<body>.*?)\r?\n?```$", re.DOTALL)


def strip_single_code_fence(text: str) -> str:
    """Remove one Markdown code fence wrapping the whole reply, if present.

    Parser revision for the v2 prompts: a reply that is exactly one JSON object
    inside one fence (for example a ```json block) is accepted. Any other prose
    around or inside the fence still fails JSON parsing downstream.
    """

    if not isinstance(text, str):
        return text
    stripped = text.strip()
    match = _SINGLE_FENCE.fullmatch(stripped)
    return match.group("body").strip() if match else stripped
