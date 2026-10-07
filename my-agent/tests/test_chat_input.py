# Offline checks for the src/chat.py input: no LLM or network calls. Keys are fed
# through a pipe as a terminal would send them, including bracketed paste
# (ESC[200~ ... ESC[201~), which is how VS Code's terminal delivers a paste.

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from chat import build_prompt_session

ENTER = "\r"
ALT_ENTER = "\x1b\r"
SNIPPET = (
    "def load(path):\n"
    "    with open(path) as f:\n"
    "        for line in f:\n"
    "\t\tprint(line)\n"
)


def _paste(text: str) -> str:
    return f"\x1b[200~{text}\x1b[201~"


def _submit(keys: str) -> str:
    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        session = build_prompt_session(input=pipe, output=DummyOutput())
        return session.prompt("vous> ")


def test_ordinary_paste_is_sent_on_enter() -> None:
    text = "Bonjour, peux-tu m'expliquer ce qu'est une injection SQL ?"
    assert _submit(_paste(text) + ENTER) == text


def test_multiline_paste_keeps_newlines_and_indentation_as_one_message() -> None:
    assert _submit(_paste(SNIPPET) + ENTER) == SNIPPET


def test_paste_does_not_submit_by_itself() -> None:
    # Text typed after the paste ends up in the same message, so the paste's
    # newlines didn't send anything.
    message = _submit(_paste(SNIPPET) + "Est-ce sûr ?" + ENTER)
    assert message == SNIPPET + "Est-ce sûr ?"


def test_windows_line_endings_are_normalized() -> None:
    assert _submit(_paste("a = 1\r\nb = 2") + ENTER) == "a = 1\nb = 2"


def test_alt_enter_inserts_a_newline_while_typing() -> None:
    assert _submit("ligne 1" + ALT_ENTER + "ligne 2" + ENTER) == "ligne 1\nligne 2"
