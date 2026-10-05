"""
src/utils/helpers.py
--------------------
Shared utility functions used across the project.
"""

import re
import unicodedata
from typing import Any


def normalise_text(text: str) -> str:
    """Normalise unicode, collapse whitespace, fix common PDF artifacts."""
    # Unicode normalisation
    text = unicodedata.normalize("NFKC", text)
    # Fix ligatures (fi, fl, ff etc.)
    ligatures = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
    for lig, repl in ligatures.items():
        text = text.replace(lig, repl)
    # Collapse multiple spaces/newlines
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_text(text: str, max_chars: int = 300) -> str:
    """Truncate text to max_chars, appending ellipsis if cut."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def format_confidence(score: float) -> str:
    """Return a human-readable confidence label."""
    if score >= 0.8:
        return f"🟢 High ({score:.0%})"
    elif score >= 0.5:
        return f"🟡 Medium ({score:.0%})"
    else:
        return f"🔴 Low ({score:.0%})"


def confidence_colour(score: float) -> str:
    """Return a CSS hex colour for a confidence score."""
    if score >= 0.8:
        return "#22c55e"   # green-500
    elif score >= 0.5:
        return "#f59e0b"   # amber-500
    else:
        return "#ef4444"   # red-500


def sanitise_filename(name: str) -> str:
    """Strip characters unsafe for filenames."""
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def flatten_metadata(metadata: dict[str, Any]) -> str:
    """Render metadata dict as a readable single-line string."""
    parts = []
    for key in ("book", "chapter", "page"):
        if key in metadata:
            parts.append(f"{key.capitalize()}: {metadata[key]}")
    return " | ".join(parts) if parts else "Unknown source"


def sigmoid(x: float) -> float:
    """Sigmoid function for score normalisation."""
    import math
    return 1.0 / (1.0 + math.exp(-x))
