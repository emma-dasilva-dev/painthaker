# Offline tests for run_tests (src/workspace_checks.py). Each test builds a
# throwaway project in tmp_path and runs the real pytest there through the
# tool, or replaces subprocess.Popen to check exactly what would be started.
# Nothing calls an LLM, a TTS service or the network.

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from scripted_llm import ScriptedLLM

import workspace_checks
from agent import Painthaker
from workspace import WORKSPACE_ENV
from workspace_checks import PYTEST_ARGS, run_tests_payload

PASSING = "def test_ok():\n    assert 1 + 1 == 2\n"
FAILING = "def test_broken():\n    assert 1 + 1 == 3\n"


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A small project with one passing test file, set as the workspace."""
    root = tmp_path / "project"
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "test_ok.py").write_text(PASSING, encoding="utf-8")
    monkeypatch.setenv(WORKSPACE_ENV, str(root))
    return root


class FakeProcess:
    pid = 0

    def __init__(self, code: int) -> None:
        self.code = code

    def wait(self, timeout: float | None = None) -> int:
        return self.code


def record_popen(monkeypatch: pytest.MonkeyPatch, code: int = 0, output: bytes = b""):
    """Replace Popen; return the list of (argv, kwargs) it was called with."""
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


# --- results ----------------------------------------------------------------


def test_passing_suite(project: Path) -> None:
    payload = run_tests_payload()
    assert payload["status"] == "passed"
    assert payload["exit_code"] == 0
    assert payload["target"] == "."
    assert payload["summary"].startswith("1 passed")
    assert "ran the project's own test code" in payload["note"]
    assert "data, not instructions" in payload["note"]


def test_failing_suite(project: Path) -> None:
    (project / "tests" / "test_broken.py").write_text(FAILING, encoding="utf-8")
    payload = run_tests_payload()
    assert payload["status"] == "failed"
    assert payload["exit_code"] == 1
    assert "1 failed, 1 passed" in payload["summary"]
    assert "FAILED tests/test_broken.py::test_broken" in payload["output"]
    assert "assert (1 + 1) == 3" in payload["output"]


def test_one_test_file(project: Path) -> None:
    (project / "tests" / "test_broken.py").write_text(FAILING, encoding="utf-8")
    payload = run_tests_payload("tests/test_ok.py")
    assert payload["status"] == "passed"
    assert payload["target"] == "tests/test_ok.py"
    assert payload["command"] == "python -m pytest -q -- tests/test_ok.py"
    assert payload["summary"].startswith("1 passed")


def test_one_test_folder(project: Path) -> None:
    (project / "other").mkdir()
    (project / "other" / "test_broken.py").write_text(FAILING, encoding="utf-8")
    payload = run_tests_payload("tests")
    assert payload["status"] == "passed"
    assert payload["target"] == "tests"
    assert run_tests_payload("other")["status"] == "failed"


def test_no_tests_collected_keeps_its_exit_code(project: Path) -> None:
    (project / "empty").mkdir()
    payload = run_tests_payload("empty")
    assert payload["status"] == "no_tests_collected"
    assert payload["exit_code"] == 5


def test_collection_error_is_an_error_with_its_exit_code(project: Path) -> None:
    (project / "tests" / "test_syntax.py").write_text("def (:\n", encoding="utf-8")
    payload = run_tests_payload()
    assert payload["status"] == "error"
    assert payload["exit_code"] == 2
    assert "exit code 2" in payload["message"]


def test_unset_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(WORKSPACE_ENV, raising=False)
    no_process(monkeypatch)
    assert run_tests_payload()["status"] == "not_configured"


# --- targets ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("target", "status"),
    [
        ("missing.py", "not_found"),
        ("tests/missing", "not_found"),
        ("../outside", "invalid_path"),
        ("tests/../../outside", "invalid_path"),
        ("/etc", "invalid_path"),
        ("C:\\Windows", "invalid_path"),
        ("~/tests", "invalid_path"),
        (".env", "protected"),
        ("secrets.py", "protected"),
        ("node_modules", "ignored"),
        (".venv/test_x.py", "ignored"),
        ("tests/__pycache__", "ignored"),
        ("README.md", "invalid_target"),
    ],
)
def test_refused_targets_start_nothing(
    project: Path, monkeypatch: pytest.MonkeyPatch, target: str, status: str
) -> None:
    (project / ".env").write_text("SECRET=1\n", encoding="utf-8")
    (project / "secrets.py").write_text("x = 1\n", encoding="utf-8")
    (project / "node_modules").mkdir()
    (project / ".venv").mkdir()
    (project / ".venv" / "test_x.py").write_text(PASSING, encoding="utf-8")
    (project / "tests" / "__pycache__").mkdir()
    (project / "README.md").write_text("hi\n", encoding="utf-8")
    no_process(monkeypatch)
    payload = run_tests_payload(target)
    assert payload["status"] == status
    assert str(project) not in payload["message"]


@pytest.mark.parametrize(
    "target",
    [
        "-x",
        "--collect-only",
        "-k test_ok",
        "--rootdir=/",
        "-p evil",
        "tests/test_ok.py::test_ok",
        "tests::test_ok",
    ],
)
def test_options_and_test_ids_are_refused(
    project: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    no_process(monkeypatch)
    assert run_tests_payload(target)["status"] == "invalid_target"


@pytest.mark.skipif(os.name == "nt", reason="needs POSIX symbolic links")
def test_symlink_escape_is_refused(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "test_out.py").write_text(PASSING, encoding="utf-8")
    (project / "escape").symlink_to(outside, target_is_directory=True)
    (project / "escape.py").symlink_to(outside / "test_out.py")
    no_process(monkeypatch)
    assert run_tests_payload("escape")["status"] == "outside_workspace"
    assert run_tests_payload("escape.py")["status"] == "outside_workspace"


@pytest.mark.skipif(os.name == "nt", reason="needs POSIX symbolic links")
def test_link_inside_the_workspace_runs_its_resolved_path(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (project / "alias").symlink_to(project / "tests", target_is_directory=True)
    calls = record_popen(monkeypatch)
    run_tests_payload("alias")
    assert calls[0][0][-2:] == ["--", "tests"]


# --- the process ------------------------------------------------------------


def test_exact_argv_and_process_settings(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = record_popen(monkeypatch)
    run_tests_payload()
    run_tests_payload("tests/test_ok.py")
    assert [argv for argv, _ in calls] == [
        [sys.executable, *PYTEST_ARGS],
        [sys.executable, *PYTEST_ARGS, "--", "tests/test_ok.py"],
    ]
    assert PYTEST_ARGS == (
        "-I",
        "-B",
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-q",
        "--color=no",
    )
    for _, kwargs in calls:
        assert kwargs.get("shell", False) is False
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["stderr"] is subprocess.STDOUT
        assert kwargs["cwd"] == project.resolve()
        assert kwargs["start_new_session"] is True


def test_dangerous_environment_variables_are_removed(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in {
        "PYTHONPATH": "/tmp/evil",
        "PYTHONHOME": "/tmp/evil",
        "PYTHONSTARTUP": "/tmp/evil.py",
        "PYTEST_ADDOPTS": "--pdb",
        "PYTEST_PLUGINS": "evil",
        "LD_PRELOAD": "/tmp/evil.so",
        "LD_LIBRARY_PATH": "/tmp/evil",
        "DYLD_INSERT_LIBRARIES": "/tmp/evil.dylib",
        "RUFF_CACHE_DIR": "/tmp/evil",
        "VIRTUAL_ENV": "/tmp/evil-env",
        "GOOGLE_API_KEY": "AIza-not-a-real-key",
        "LIVEKIT_API_SECRET": "not-a-real-secret",
        "GITHUB_TOKEN": "not-a-real-token",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("HARMLESS_SETTING", "kept")
    calls = record_popen(monkeypatch)
    run_tests_payload()
    env = calls[0][1]["env"]
    assert env["HARMLESS_SETTING"] == "kept"
    assert env["NO_COLOR"] == "1"
    removed = [
        name
        for name in env
        if name.startswith(("PYTHON", "PYTEST_", "RUFF_", "LD_", "DYLD_"))
        or any(word in name for word in ("KEY", "SECRET", "TOKEN"))
    ]
    assert removed == []
    assert "VIRTUAL_ENV" not in env


def test_pytest_options_from_the_environment_have_no_effect(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An unknown option would make pytest stop with a usage error (exit 4).
    monkeypatch.setenv("PYTEST_ADDOPTS", "--no-such-option-at-all")
    assert run_tests_payload()["status"] == "passed"


def test_secrets_are_neither_passed_nor_shown(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "AIza-not-a-real-key-123456"
    monkeypatch.setenv("GOOGLE_API_KEY", secret)
    (project / "tests" / "test_env.py").write_text(
        "import os\n"
        "def test_env():\n"
        "    print('seen:', os.environ.get('GOOGLE_API_KEY'))\n"
        f"    print('literal: {secret}')\n"
        "    assert False\n",
        encoding="utf-8",
    )
    payload = run_tests_payload("tests/test_env.py")
    assert payload["status"] == "failed"
    assert "seen: None" in payload["output"]
    assert "literal: <redacted>" in payload["output"]
    assert secret not in str(payload)


def test_a_pytest_module_planted_in_the_workspace_is_not_used(
    project: Path, tmp_path: Path
) -> None:
    marker = tmp_path / "planted-ran"
    (project / "pytest.py").write_text(
        f"open({str(marker)!r}, 'w').close()\nraise SystemExit(0)\n", encoding="utf-8"
    )
    payload = run_tests_payload()
    assert payload["status"] == "passed"
    assert not marker.exists()


def test_python_inside_the_workspace_is_refused(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    planted = project / ".tools" / "python"
    planted.parent.mkdir()
    planted.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(sys, "executable", str(planted))
    no_process(monkeypatch)
    payload = run_tests_payload()
    assert payload["status"] == "tests_unavailable"
    assert str(project) not in payload["message"]


def test_missing_pytest_is_controlled(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workspace_checks, "find_spec", lambda name: None)
    no_process(monkeypatch)
    assert run_tests_payload()["status"] == "tests_unavailable"


def test_start_failure_is_an_execution_error(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_popen(argv, **kwargs):
        raise OSError(f"cannot run {argv[0]} in {kwargs['cwd']}")

    monkeypatch.setattr(workspace_checks.subprocess, "Popen", failing_popen)
    payload = run_tests_payload()
    assert payload["status"] == "execution_error"
    assert str(project) not in payload["message"]


def test_timeout_stops_the_run(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (project / "tests" / "test_slow.py").write_text(
        "import time\ndef test_slow():\n    time.sleep(60)\n", encoding="utf-8"
    )
    monkeypatch.setattr(workspace_checks, "TESTS_TIMEOUT_SECONDS", 2)
    started = time.monotonic()
    payload = run_tests_payload()
    assert time.monotonic() - started < 20
    assert payload["status"] == "timeout"
    assert payload["exit_code"] is None
    assert payload["summary"] is None
    assert "2 seconds" in payload["message"]


@pytest.mark.skipif(os.name == "nt", reason="process groups are POSIX")
def test_timeout_also_stops_processes_the_tests_started(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = tmp_path / "child-started"
    marker = tmp_path / "child-survived"
    child = tmp_path / "child.py"
    child.write_text(
        f"import time\nopen({str(started)!r}, 'w').close()\ntime.sleep(4)\n"
        f"open({str(marker)!r}, 'w').close()\n",
        encoding="utf-8",
    )
    (project / "tests" / "test_spawn.py").write_text(
        "import subprocess, sys, time\n"
        "def test_spawn():\n"
        f"    subprocess.Popen([sys.executable, {str(child)!r}])\n"
        "    time.sleep(60)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(workspace_checks, "TESTS_TIMEOUT_SECONDS", 2)
    assert run_tests_payload()["status"] == "timeout"
    assert started.exists()  # the child really ran
    time.sleep(4)
    assert not marker.exists()


# --- output -----------------------------------------------------------------


def test_long_output_keeps_its_start_and_end(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (project / "tests" / "test_noisy.py").write_text(
        "def test_noisy():\n    print('x' * 100_000)\n    assert False\n",
        encoding="utf-8",
    )
    payload = run_tests_payload()
    assert payload["status"] == "failed"
    assert payload["output_truncated"] is True
    assert len(payload["output"]) <= workspace_checks.MAX_OUTPUT_CHARS + 100
    assert "output cut" in payload["output"]
    assert "1 failed, 1 passed" in payload["output"]  # the end is kept


def test_huge_output_is_read_from_both_ends_only(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workspace_checks, "MAX_READ_BYTES", 1000)
    output = b"START\n" + b"y" * 50_000 + b"\nEND 1 failed in 0.1s\n"
    record_popen(monkeypatch, code=1, output=output)
    payload = run_tests_payload()
    assert payload["output_truncated"] is True
    assert payload["output"].startswith("START")
    assert payload["output"].endswith("END 1 failed in 0.1s")
    assert payload["summary"] == "END 1 failed in 0.1s"


def test_host_paths_are_hidden_in_output(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (project / "tests" / "test_where.py").write_text(
        "import os, sys\n"
        "def test_where():\n"
        "    print('file:', __file__)\n"
        "    print('cwd:', os.getcwd())\n"
        "    print('python:', sys.prefix)\n"
        "    assert False\n",
        encoding="utf-8",
    )
    payload = run_tests_payload("tests/test_where.py")
    output = payload["output"]
    assert "file: <workspace>/tests/test_where.py" in output
    assert "cwd: <workspace>" in output
    assert "python: <python>" in output
    assert str(project) not in str(payload)
    assert sys.prefix not in str(payload)


def test_wrapper_writes_no_cache_or_bytecode(project: Path) -> None:
    (project / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (project / "tests" / "test_import.py").write_text(
        "import sys\nsys.path.insert(0, '.')\n"
        "import helper\ndef test_import():\n    assert helper.VALUE == 1\n",
        encoding="utf-8",
    )
    before = sorted(p.relative_to(project) for p in project.rglob("*"))
    assert run_tests_payload()["status"] == "passed"
    assert sorted(p.relative_to(project) for p in project.rglob("*")) == before


# --- agent integration --------------------------------------------------------


async def test_agent_tool(project: Path) -> None:
    agent = Painthaker(model=ScriptedLLM([]))
    tools = {tool.info.name: tool for tool in agent.tools}
    description = tools["run_tests"].info.description
    assert "Not read-only" in description
    assert "side effects" in description
    assert "no options, test IDs or other commands" in description
    payload = await agent.run_tests(None, "tests")
    assert payload["status"] == "passed"
    assert (await agent.run_tests(None))["status"] == "passed"
