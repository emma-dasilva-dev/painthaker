# Offline checks of the documented first-run path: `cp .env.example .env.local`,
# then start the chat. No model or network calls.

import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent


def test_env_example_lists_every_credential_the_chat_needs() -> None:
    keys = {
        line.split("=", 1)[0].strip()
        for line in (PROJECT / ".env.example").read_text().splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    }
    assert {
        "GOOGLE_API_KEY",
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
    } <= keys


def test_missing_google_key_stops_cleanly_before_creating_history(
    tmp_path: Path,
) -> None:
    # Same as a fresh clone where .env.local was copied from .env.example but the
    # key wasn't filled in: run from a directory with that .env.local.
    (tmp_path / ".env.local").write_text((PROJECT / ".env.example").read_text())
    db = tmp_path / "data" / "history.sqlite3"
    env = {k: v for k, v in os.environ.items() if k != "GOOGLE_API_KEY"}
    env.update(PAINTHAKER_HISTORY_DB=str(db), HOME=str(tmp_path))

    run = subprocess.run(
        [sys.executable, str(PROJECT / "src" / "chat.py")],
        cwd=tmp_path,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
    )

    output = run.stdout + run.stderr
    assert run.returncode == 1
    assert "GOOGLE_API_KEY manquante" in output
    assert "Traceback" not in output
    assert not db.exists()
