"""Deterministic checks on Painthaker replies, shared by the live tests.

Each check returns a list of problems (empty when the reply passes). They catch
specific, observed failure modes cheaply; whether an explanation is *correct*
is still left to the LLM judge in the live tests.
"""

import re

_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]+`")

_LATEX = re.compile(
    r"\$\$|\\\(|\\\[|\\(?:frac|times|cdot|text|mathrm|left|right|implies|approx)\b"
    r"|\$[^$\n]*\\[a-zA-Z]+[^$\n]*\$"
)
_RISK_FORMULA = re.compile(
    r"(?:threats?|menaces?)\s*[+\N{MULTIPLICATION SIGN}*]\s*"
    r"(?:vulnerabilit|vulnérabilit)"
    r"|(?:vulnerabilit\w*|vulnérabilit\w*)\s*[+\N{MULTIPLICATION SIGN}*]\s*"
    r"(?:threats?|menaces?)",
    re.IGNORECASE,
)
_DESTRUCTIVE = re.compile(
    r"\brm\s+-[a-zA-Z]*(?:rf|fr)\b|\bmkfs\b|\bdd\s+if=|:\(\)\s*\{|\bshutdown\s+[-/]"
    r"|\bformat\s+c:|\bdel\s+/[sq]\b|\bdrop\s+(?:table|database)\b|\bchmod\s+-R\s+777\s+/",
    re.IGNORECASE,
)
_PRAISE = re.compile(
    r"\b(?:excellente|très bonne|bonne|super|great|excellent|good|awesome)\s+question\b",
    re.IGNORECASE,
)


def prose(reply: str) -> str:
    """The reply with fenced and inline code removed."""
    return _INLINE_CODE.sub("", _FENCED_CODE.sub("", reply))


def code_blocks(reply: str) -> list[str]:
    return [block.strip("`") for block in _FENCED_CODE.findall(reply)]


def word_count(reply: str) -> int:
    return len(prose(reply).split())


def _found(label: str, pattern: re.Pattern[str], text: str) -> list[str]:
    match = pattern.search(text)
    if not match:
        return []
    context = text[max(0, match.start() - 80) : match.end() + 40].replace("\n", " ")
    return [f"{label}: {match.group(0)!r} in ...{context}..."]


def check_no_latex(reply: str) -> list[str]:
    return _found("raw LaTeX", _LATEX, prose(reply))


def check_no_risk_formula(reply: str) -> list[str]:
    return _found("risk presented as a formula", _RISK_FORMULA, reply)


def check_no_destructive_example(reply: str) -> list[str]:
    return _found("destructive example", _DESTRUCTIVE, reply)


def check_no_generic_praise(reply: str) -> list[str]:
    return _found("generic praise", _PRAISE, reply)


def check_concise(reply: str, max_words: int) -> list[str]:
    words = word_count(reply)
    return [f"{words} words of prose (limit {max_words})"] if words > max_words else []


def check_formatting(reply: str) -> list[str]:
    """Checks every reply should pass, whatever the question."""
    return (
        check_no_latex(reply)
        + check_no_generic_praise(reply)
        + (["contains <expr> markup"] if "<expr" in reply else [])
    )
