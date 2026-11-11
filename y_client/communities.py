"""Community actions of the forum agents (opt-in): found a new community, cross-post to a related one.

In the forum a community is a topic. The client config enables the two actions in a ``communities`` section::

    "communities": {
        "creation":  {"enabled": false, "probability": 0.05, "max_topics": 0},
        "crosspost": {"enabled": false, "probability": 0.15, "max_per_post": 1}
    }

``creation``: before writing a post an agent founds, with ``probability``, a new community (the model proposes its
name, the server creates it or answers the existing one) and the post is its first. ``max_topics`` caps the
communities of the forum (0 = no cap). ``crosspost``: after a post an agent shows it, with ``probability``, to the model
together with the other communities; the model names those where it is clearly on topic (at most ``max_per_post``)
and the server adds a cross-post of the thread in each. Without the section nothing here is active.
"""

import random
import re

DEFAULT_CREATION = {"enabled": False, "probability": 0.05, "max_topics": 0}
DEFAULT_CROSSPOST = {"enabled": False, "probability": 0.15, "max_per_post": 1}
NAME_MAX = 20  # the forum stores a community name in 20 characters
NONE_WORDS = {"", "none", "null", "nessuna", "nessuno", "n/a", "-"}

_state = {"creation": dict(DEFAULT_CREATION), "crosspost": dict(DEFAULT_CROSSPOST)}


def _section(raw, defaults):
    raw = raw if isinstance(raw, dict) else {}
    out = dict(defaults)
    out["enabled"] = bool(raw.get("enabled", defaults["enabled"]))
    out["probability"] = min(1.0, max(0.0, float(raw.get("probability", defaults["probability"]))))
    for key in ("max_topics", "max_per_post"):
        if key in defaults:
            out[key] = max(0, int(raw.get(key, defaults[key]) or 0))
    return out


def install(config):
    """Read the ``communities`` section of the client config (absent: both actions off)."""
    section = (config or {}).get("communities") or {}
    _state["creation"] = _section(section.get("creation"), DEFAULT_CREATION)
    _state["crosspost"] = _section(section.get("crosspost"), DEFAULT_CROSSPOST)


def creation_enabled():
    return _state["creation"]["enabled"]


def crosspost_enabled():
    return _state["crosspost"]["enabled"]


def max_topics():
    return _state["creation"]["max_topics"]


def max_per_post():
    return max(1, _state["crosspost"]["max_per_post"])


def draw_creation(rng=random):
    return creation_enabled() and rng.random() < _state["creation"]["probability"]


def draw_crosspost(rng=random):
    return crosspost_enabled() and rng.random() < _state["crosspost"]["probability"]


def _clean_name(raw):
    name = str(raw or "").strip().strip("\"'`*.,;:").strip()
    name = re.sub(r"^(?:[yr]/|#)", "", name, flags=re.IGNORECASE).strip()
    return re.sub(r"\s+", " ", name)


def parse_name(reply):
    """The community name in a model reply (first non-empty line), or None for NONE / nothing usable."""
    for line in str(reply or "").splitlines():
        name = _clean_name(line)
        if not name:
            continue
        if name.lower().rstrip(".") in NONE_WORDS:
            return None
        name = name[:NAME_MAX].strip()
        return name if len(name) >= 3 else None
    return None


def parse_choice(reply, candidates, limit=1):
    """The communities of ``candidates`` named in a model reply, in the order of the reply, at most ``limit``.

    A name that is not in the list is ignored (the model may only choose among the existing communities)."""
    by_lower = {str(c).strip().lower(): c for c in candidates}
    picked = []
    for part in re.split(r"[,\n;]", str(reply or "")):
        name = _clean_name(part).lower().rstrip(".")
        if name in NONE_WORDS:
            continue
        found = by_lower.get(name)
        if found is not None and found not in picked:
            picked.append(found)
        if len(picked) >= limit:
            break
    return picked


def crosspost_text(post_text, note, limit=480):
    """The text of a cross-post: the title and the start of the body of the original, then a note naming where it
    comes from. Short enough for a post (500 characters)."""
    text = str(post_text or "").strip()
    match = re.match(r"(?is)\s*TITLE:\s*(.+?)(?:\n\s*\n(.*))?$", text)
    title, body = (match.group(1).strip(), (match.group(2) or "").strip()) if match else ("", text)
    head = f"TITLE: {title}\n\n" if title else ""
    tail = f"\n\n{note}"
    room = max(0, limit - len(head) - len(tail))
    if len(body) > room:
        body = body[: max(0, room - 1)].rstrip() + "…"
    return f"{head}{body}{tail}".strip()
