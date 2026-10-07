# Offline tests for the reply-language selection in src/agent.py (no LLM calls).
# Several messages are ones that previously got a reply in the wrong language.

from agent import detect_language, reply_language

CODE = (
    "```python\nimport os\ndef check(host):\n    return os.system('ping ' + host)\n```"
)


def test_defaults_to_french() -> None:
    assert reply_language([]) == "fr"
    assert reply_language(["Bonjour !"]) == "fr"
    assert reply_language(["ok"]) == "fr"


def test_short_english_question_next_to_code_is_english() -> None:
    message = (
        "Is this endpoint helper exploitable? `host` comes from a query parameter.\n"
        + CODE
    )
    assert detect_language(message) == "en"
    assert (
        detect_language("Please review this Python code for security:\n" + CODE) == "en"
    )


def test_code_alone_does_not_decide_the_language() -> None:
    assert detect_language(CODE) is None
    assert reply_language(["Can we continue in English please?", CODE]) == "en"


def test_french_with_code_is_french() -> None:
    assert detect_language("Est-ce que ce code est vulnérable ?\n" + CODE) == "fr"


def test_explicit_requests_switch_language() -> None:
    assert reply_language(["Salut", "Can we continue in English please?"]) == "en"
    assert reply_language(["What is XSS?", "Réponds en français s'il te plaît"]) == "fr"


def test_language_sticks_until_the_user_switches() -> None:
    history = ["Can we continue in English please? What is XSS?", "ok", "thanks"]
    assert reply_language(history) == "en"
    assert reply_language([*history, "Et c'est quoi une injection SQL ?"]) == "fr"
