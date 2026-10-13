"""Label sanitization — the ONLY path external text may take into prompts,
golden files, or the site.

Moved out of eval/harvest.py so the tag-harvest tool and the enrichment
layer (afrec/enrich.py) share one implementation. Last.fm tags are
crowd-sourced (prompt-injection surface); MusicBrainz and Musicae strings
go through the same filter defensively.

Contract: anything that survives clean_label() is `[A-Za-z0-9 .&'-]` only,
whitespace-collapsed, ≤ maxlen chars — no quotes, no angle brackets, no
unicode punctuation, no control characters. It can be embedded in
JSON/markdown/JS without escaping tricks and could never carry
instructions.
"""

from __future__ import annotations

import re

# Allowed in any label we keep: letters, digits, spaces, . & ' -
# (covers "jazz hiphop", "neo-soul", "R&B"→"R&B", "post-punk", "Miles Davis")
_LABEL_BAD = re.compile(r"[^A-Za-z0-9 .&'\-]+")


def clean_label(value, maxlen: int = 60) -> str | None:
    """Whitelist-sanitize one string. Returns None when nothing usable remains.

    This is deliberately dumb and strict.
    """
    if value is None:
        return None
    s = str(value).replace("\u2019", "'").replace("\u2018", "'")
    s = re.sub(r"[\t\n\r]+", " ", s)                      # whitespace controls → space
    s = "".join(ch for ch in s if ch.isprintable() and ch != "\x7f")
    s = _LABEL_BAD.sub(" ", s)
    s = " ".join(s.split())
    s = s[:maxlen].strip()
    return s or None
