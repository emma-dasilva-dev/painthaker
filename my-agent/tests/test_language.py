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


def test_short_english_imperatives_are_english() -> None:
    # These got French replies in the 2026-10-20 live workspace-tools test.
    for message in [
        "Read .env.local.",
        "Read ../something-outside-the-workspace.",
        "Search node_modules for something.",
        "Find auth.py",
        "Show git status",
        "Check pyproject.toml",
        "Please read src/agent.py",
        "List the files in src/",
    ]:
        assert detect_language(message) == "en", message
    assert reply_language(["Read .env.local."]) == "en"


def test_short_french_imperatives_are_french() -> None:
    for message in [
        "Lis .env.local.",
        "Cherche node_modules",
        "Trouve auth.py",
        "Montre le statut git",
        "Vérifie pyproject.toml",
        "Puis affiche src/agent.py",
    ]:
        assert detect_language(message) == "fr", message


def test_technical_text_alone_does_not_decide_the_language() -> None:
    for message in [
        ".env.local",
        "node_modules",
        "src/agent.py",
        "git status",
        "../the/path/of/this/file.py",
        "https://example.com/what-is-the-answer",
        "ok",
    ]:
        assert detect_language(message) is None, message
    assert reply_language(["Can we continue in English please?", "src/agent.py"]) == (
        "en"
    )
    assert reply_language(["Bonjour !", "git status"]) == "fr"


def test_a_verb_alone_does_not_beat_the_other_language() -> None:
    # Franglais: an English verb with a French cue stays undecided (sticky).
    assert detect_language("Check le fichier") is None
    assert reply_language(["Bonjour !", "Check le fichier"]) == "fr"
    # A verb later in a sentence isn't an imperative cue.
    assert detect_language("Je veux read") is None


def test_language_sticks_until_the_user_switches() -> None:
    history = ["Can we continue in English please? What is XSS?", "ok", "thanks"]
    assert reply_language(history) == "en"
    assert reply_language([*history, "Et c'est quoi une injection SQL ?"]) == "fr"
