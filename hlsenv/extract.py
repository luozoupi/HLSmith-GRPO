"""Pull compilable C++ out of an LLM completion."""

from __future__ import annotations

import re

# Match EVERY fenced block regardless of tag (```python, ```Cpp, untagged, ...).
# Matching only C++-tagged fences would mis-pair a foreign block's closing ```
# with the next block's opening ``` and extract the prose in between.
_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
_CPP_TAGS = {"cpp", "c++", "cxx", "cc", "c"}


def extract_cpp(completion: str | None) -> str | None:
    """Return the most plausible C++ source block, or None if there is no code.

    Preference order: last block explicitly tagged C/C++ (case-insensitive),
    then last block of any tag that looks like C++, then any fenced block,
    then the raw text if it itself looks like a function definition.
    """
    if not completion:
        return None

    blocks = [(tag.strip().lower(), body) for tag, body in _FENCE_RE.findall(completion)]

    for tag, body in reversed(blocks):
        if tag in _CPP_TAGS:
            return body.strip() + "\n"
    for _tag, body in reversed(blocks):
        if "void" in body or "#include" in body:
            return body.strip() + "\n"
    if blocks:
        return blocks[-1][1].strip() + "\n"

    text = completion.strip()
    if ("void" in text or "#include" in text) and "{" in text:
        return text + "\n"
    return None
