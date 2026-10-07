# Offline tests for tests/response_checks.py: no LLM calls. The "bad" samples
# are shaped on replies Painthaker actually produced.

from response_checks import (
    check_concise,
    check_formatting,
    check_no_destructive_example,
    check_no_latex,
    check_no_risk_formula,
)


def test_flags_raw_latex() -> None:
    assert check_no_latex(r"On peut l'écrire $$Risque = Menace \times Vulnérabilité$$")
    assert check_no_latex(r"where \( R = P \cdot I \)")


def test_ignores_dollar_signs_in_code() -> None:
    reply = 'Run it with:\n```bash\necho $HOME && cat "$FILE"\n```\nThen `$PATH`.'
    assert check_no_latex(reply) == []


def test_flags_risk_formula_in_both_languages() -> None:
    assert check_no_risk_formula("**Risque = Menace + Vulnérabilité**")
    assert check_no_risk_formula("Threat + Vulnerability = Risk")
    assert check_no_risk_formula("Risk = Threat \N{MULTIPLICATION SIGN} Vulnerability")


def test_accepts_risk_explained_without_formula() -> None:
    reply = (
        "Risk depends on how likely a threat is to exploit a vulnerability "
        "and how bad the impact would be."
    )
    assert check_no_risk_formula(reply) == []


def test_flags_destructive_examples() -> None:
    assert check_no_destructive_example("an attacker could type `file.txt; rm -rf /`")
    assert check_no_destructive_example("Robert'); DROP TABLE students;--")


def test_accepts_harmless_demonstrations() -> None:
    assert (
        check_no_destructive_example("for example `notes.txt; id` runs `id` too") == []
    )


def test_flags_generic_praise() -> None:
    assert check_formatting("Excellente question ! Une menace est...")
    assert check_formatting("Great question! A threat is...")


def test_concision_counts_prose_only() -> None:
    long_code = "```python\n" + "x = 1\n" * 500 + "```"
    assert check_concise("Short answer." + long_code, max_words=20) == []
    assert check_concise("word " * 50, max_words=20)
