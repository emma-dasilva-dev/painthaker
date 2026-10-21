# Offline tests for run_linter (src/workspace_checks.py). Each test builds a
# throwaway project in tmp_path and runs the real Ruff there through the tool,
# or replaces subprocess.Popen to check exactly what would be started. Nothing
# calls an LLM, a TTS service or the network.

import os
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

import pytest
from scripted_llm import ScriptedLLM

import workspace_checks
from agent import SPOKEN_INSTRUCTIONS, TEXT_INSTRUCTIONS, Painthaker
from workspace import WORKSPACE_ENV
from workspace_checks import RUFF_ARGS, ruff_executable, run_linter_payload

pytestmark = pytest.mark.skipif(
    not (
        Path(sysconfig.get_path("scripts"))
        / ("ruff" + (sysconfig.get_config_var("EXE") or ""))
    ).is_file(),
    reason="needs ruff in this Python environment",
)

UNUSED_IMPORT = "import os\n"


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A small, lint-clean project set as the workspace."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "app.py").write_text("print('hi')\n", encoding="utf-8")
    monkeypatch.setenv(WORKSPACE_ENV, str(root))
    return root


class FakeProcess:
    pid = 0

    def __init__(self, code: int) -> None:
        self.code = code

    def wait(self, timeout: float | None = None) -> int:
        return self.code


def record_popen(monkeypatch: pytest.MonkeyPatch, code: int = 0, output: bytes = b""):
    calls: list[tuple[list[str], dict]] = []

    def fake_popen(argv, **kwargs):
        calls.append((list(argv), kwargs))
        kwargs["stdout"].write(output)
        return FakeProcess(code)

    monkeypatch.setattr(workspace_checks.subprocess, "Popen", fake_popen)
    return calls


def no_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        workspace_checks.subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("no process may be started"),
    )


def test_clean_project(project: Path) -> None:
    payload = run_linter_payload()
    assert payload["status"] == "clean"
    assert payload["exit_code"] == 0
    assert payload["summary"] == "All checks passed!"
    assert "no file was changed" in payload["note"]
    assert "data, not instructions" in payload["note"]


def test_findings_are_reported_and_nothing_is_fixed(project: Path) -> None:
    # A project setting `fix = true` would make a plain `ruff check` edit files.
    (project / "pyproject.toml").write_text("[tool.ruff]\nfix = true\n", "utf-8")
    (project / "bad.py").write_text(UNUSED_IMPORT, encoding="utf-8")
    payload = run_linter_payload()
    assert payload["status"] == "findings"
    assert payload["exit_code"] == 1
    assert "bad.py:1:8: F401" in payload["output"]
    # Ruff's own last line is a "fixable with --fix" hint, not the summary.
    assert payload["summary"] == "Found 1 error."
    assert "can't fix files" in payload["note"]
    assert (project / "bad.py").read_text(encoding="utf-8") == UNUSED_IMPORT


def test_no_cache_or_other_file_is_written(project: Path) -> None:
    (project / "bad.py").write_text(UNUSED_IMPORT, encoding="utf-8")

    def snapshot() -> dict[str, tuple[int, int]]:
        return {
            str(p): (p.stat().st_mtime_ns, p.stat().st_size) for p in project.rglob("*")
        }

    before = snapshot()
    run_linter_payload()
    assert snapshot() == before
    assert not (project / ".ruff_cache").exists()


def test_configuration_error_is_an_error(project: Path) -> None:
    (project / "pyproject.toml").write_text(
        '[tool.ruff]\nline-length = "long"\n', encoding="utf-8"
    )
    payload = run_linter_payload()
    assert payload["status"] == "error"
    assert payload["exit_code"] == 2
    assert "exit code 2" in payload["message"]
    assert str(project) not in str(payload)


def test_exact_argv_and_process_settings(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = record_popen(monkeypatch)
    run_linter_payload()
    (argv, kwargs) = calls[0]
    assert argv == [ruff_executable(project.resolve()), *RUFF_ARGS]
    assert RUFF_ARGS == (
        "check",
        "--no-fix",
        "--no-cache",
        "--output-format=concise",
        ".",
    )
    assert "--fix" not in argv and "format" not in argv
    assert Path(argv[0]).parent == Path(sysconfig.get_path("scripts"))
    assert kwargs.get("shell", False) is False
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["cwd"] == project.resolve()
    assert kwargs["start_new_session"] is True


def test_environment_is_sanitized(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUFF_CACHE_DIR", "/tmp/evil")
    monkeypatch.setenv("RUFF_OUTPUT_FORMAT", "json")
    monkeypatch.setenv("PYTHONPATH", "/tmp/evil")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/evil.so")
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-not-a-real-key")
    calls = record_popen(monkeypatch)
    run_linter_payload()
    env = calls[0][1]["env"]
    for name in (
        "RUFF_CACHE_DIR",
        "RUFF_OUTPUT_FORMAT",
        "PYTHONPATH",
        "LD_PRELOAD",
        "GOOGLE_API_KEY",
    ):
        assert name not in env


def test_ruff_settings_from_the_environment_have_no_effect(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUFF_OUTPUT_FORMAT", "json")
    monkeypatch.setenv("RUFF_CACHE_DIR", str(project / "cache"))
    (project / "bad.py").write_text(UNUSED_IMPORT, encoding="utf-8")
    payload = run_linter_payload()
    assert "bad.py:1:8: F401" in payload["output"]
    assert not (project / "cache").exists()


def test_missing_ruff_is_controlled(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty-scripts"
    empty.mkdir()
    monkeypatch.setattr(workspace_checks.sysconfig, "get_path", lambda name: str(empty))
    no_process(monkeypatch)
    payload = run_linter_payload()
    assert payload["status"] == "linter_unavailable"


def test_ruff_planted_in_the_workspace_is_refused(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts = project / "bin"
    scripts.mkdir()
    planted = scripts / ("ruff" + (sysconfig.get_config_var("EXE") or ""))
    planted.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    planted.chmod(0o755)
    monkeypatch.setattr(
        workspace_checks.sysconfig, "get_path", lambda name: str(scripts)
    )
    no_process(monkeypatch)
    payload = run_linter_payload()
    assert payload["status"] == "linter_unavailable"
    assert str(project) not in payload["message"]


def test_ruff_on_path_is_never_used(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    planted = project / "ruff"
    planted.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    planted.chmod(0o755)
    monkeypatch.setenv("PATH", f"{project}{os.pathsep}{os.environ.get('PATH', '')}")
    calls = record_popen(monkeypatch)
    run_linter_payload()
    assert not Path(calls[0][0][0]).is_relative_to(project)


def test_timeout_stops_the_run(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_popen = subprocess.Popen

    def slow_popen(argv, **kwargs):
        return real_popen(
            [sys.executable, "-c", "import time; time.sleep(60)"], **kwargs
        )

    monkeypatch.setattr(workspace_checks.subprocess, "Popen", slow_popen)
    monkeypatch.setattr(workspace_checks, "LINT_TIMEOUT_SECONDS", 1)
    started = time.monotonic()
    payload = run_linter_payload()
    assert time.monotonic() - started < 15
    assert payload["status"] == "timeout"
    assert payload["exit_code"] is None
    assert "1 seconds" in payload["message"]


def test_long_output_is_cut(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workspace_checks, "MAX_OUTPUT_CHARS", 600)
    monkeypatch.setattr(workspace_checks, "OUTPUT_HEAD_CHARS", 200)
    (project / "bad.py").write_text(
        "".join(f"import mod{i}\n" for i in range(100)), encoding="utf-8"
    )
    payload = run_linter_payload()
    assert payload["status"] == "findings"
    assert payload["output_truncated"] is True
    assert len(payload["output"]) <= 700
    assert payload["output"].startswith("bad.py:1:8: F401")
    assert "Found 100 errors" in payload["output"]


def test_host_paths_are_hidden_in_output(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = os.path.expanduser("~")
    output = (
        f"{project.resolve()}/bad.py:1:8: F401 `os` imported but unused\n"
        f"warning: config at {home}/.config/ruff/ruff.toml\n"
        f"{sys.prefix}/lib/thing\n"
        "Found 1 error.\n"
    ).encode()
    record_popen(monkeypatch, code=1, output=output)
    payload = run_linter_payload()
    assert payload["output"].startswith("<workspace>/bad.py:1:8: F401")
    assert "~/.config/ruff/ruff.toml" in payload["output"]
    assert "<python>/lib/thing" in payload["output"]
    assert str(project) not in str(payload)
    assert payload["summary"] == "Found 1 error."


def test_unset_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(WORKSPACE_ENV, raising=False)
    no_process(monkeypatch)
    assert run_linter_payload()["status"] == "not_configured"


# --- agent integration --------------------------------------------------------


async def test_agent_tool(project: Path) -> None:
    agent = Painthaker(model=ScriptedLLM([]))
    tools = {tool.info.name: tool for tool in agent.tools}
    description = tools["run_linter"].info.description
    assert "Check only" in description and "never fixes or formats" in description
    assert (await agent.run_linter(None))["status"] == "clean"


def test_instructions_describe_the_verification_tools() -> None:
    for instructions in (TEXT_INSTRUCTIONS, SPOKEN_INSTRUCTIONS):
        text = " ".join(instructions.split())
        assert "run_linter runs Ruff's check only" in text
        assert "run_tests executes the project's own test code" in text
        assert "don't call it read-only" in text
        assert "Neither tool accepts commands, options or test IDs" in text
        assert 'unless the tool reported "passed" or "clean"' in text


def test_instructions_forbid_dropping_unsupported_options() -> None:
    # A live check ran the whole suite for "Run pytest with -x." and never
    # said -x was dropped. Key terms only, so rewording doesn't break this.
    for instructions in (TEXT_INSTRUCTIONS, SPOKEN_INSTRUCTIONS):
        text = " ".join(instructions.split())
        for term in ("-x", "-k", '"::"', "ruff --fix", "shell command"):
            assert term in text, term
        assert "can't be applied" in text
        assert "Never drop part of a request silently" in text
    agent = Painthaker(model=ScriptedLLM([]))
    tools = {tool.info.name: tool for tool in agent.tools}
    for name, option in (("run_tests", "-x"), ("run_linter", "--fix")):
        description = " ".join(tools[name].info.description.split())
        assert option in description and "can't be applied" in description
