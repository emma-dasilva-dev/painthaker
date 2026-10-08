"""Explicit user preferences: reply language, explanation level, reply length.

Set only with the /prefs command and saved in the history database's
`preferences` table (see history.py). Nothing is inferred from conversations.
"""

from dataclasses import dataclass, replace

OPTIONS: dict[str, tuple[str, ...]] = {
    "language": ("auto", "fr", "en"),
    "level": ("beginner", "intermediate", "advanced"),
    "length": ("concise", "detailed"),
}


@dataclass(frozen=True)
class Preferences:
    language: str = "auto"
    level: str = "beginner"
    length: str = "concise"

    def with_value(self, key: str, value: str) -> "Preferences":
        if key not in OPTIONS or value not in OPTIONS[key]:
            raise ValueError(f"invalid preference {key}={value}")
        return replace(self, **{key: value})

    @classmethod
    def from_stored(cls, stored: dict[str, str]) -> "Preferences":
        """Known keys with valid values; anything else falls back to the default."""
        prefs = cls()
        for key, value in stored.items():
            if key in OPTIONS and value in OPTIONS[key]:
                prefs = prefs.with_value(key, value)
        return prefs


_LEVEL_GUIDANCE = {
    "beginner": (
        "a beginner: define technical terms when you first use them, prefer "
        "plain words and a concrete example, and don't assume prior knowledge. "
        "Simpler must stay true: an analogy may not claim more than the fact "
        '(for example, "permanent" for something that is only usually fixed)'
    ),
    "intermediate": (
        "an intermediate learner: assume the basics (HTTP, shells, common "
        "vulnerability names) and focus on mechanisms and trade-offs"
    ),
    "advanced": (
        "an advanced practitioner: skip introductions, use precise terminology, "
        "and go straight to details, edge cases and references to standards"
    ),
}
_LENGTH_GUIDANCE = {
    "concise": (
        "concise: the shortest answer that is still correct and complete enough "
        "to act on safely. Never drop a caveat, a security warning, the main "
        "exception to a general statement or a step needed for correctness just "
        "to be shorter"
    ),
    "detailed": (
        "detailed: a fuller explanation with reasoning, an example and the main "
        "exceptions, still without padding"
    ),
}


def preferences_note(prefs: Preferences) -> str:
    """Per-call system note for explanation level and reply length. The reply
    language is set by its own note, so this one never mentions language."""
    return (
        "User preferences (set by the user with /prefs): explain as if to "
        f"{_LEVEL_GUIDANCE[prefs.level]}. Reply length: "
        f"{_LENGTH_GUIDANCE[prefs.length]}. If the user's latest message clearly "
        "asks for something else (for example a simpler or more advanced "
        "explanation, or more or less detail), follow that request for this "
        "reply only; it does not change these preferences."
    )
