# Offline tests for git_status (src/workspace_git.py). Each test builds a
# throwaway repository in tmp_path; the helper below runs git there to set up
# the scenario. Nothing calls an LLM, a TTS service or the network.

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from scripted_llm import ScriptedLLM

import workspace_git
from agent import Painthaker
from workspace import WORKSPACE_ENV
from workspace_git import (
    GIT_OPTIONS,
    PREFIX_COMMAND,
    STATUS_COMMAND,
    git_status_payload,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

SECRET = "TOP-SECRET-VALUE-42"


def _env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def git(repo: Path, *args: str) -> str:
    """Set up a test repository (never used on a real one)."""
    done = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        env=_env(),
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A repository with one commit on main, set as the workspace."""
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    (root / "app.py").write_text("print('hi')\n", encoding="utf-8")
    (root / "lib.py").write_text("x = 1\n", encoding="utf-8")
    git(root, "add", "app.py", "lib.py")
    git(root, "commit", "-q", "-m", "init")
    monkeypatch.setenv(WORKSPACE_ENV, str(root))
    return root


def test_clean_repository(repo: Path) -> None:
    payload = git_status_payload()
    assert payload["status"] == "ok"
    assert payload["branch"] == "main"
    assert payload["detached"] is False
    assert payload["clean"] is True
    assert payload["staged"] == payload["unstaged"] == payload["untracked"] == []
    assert payload["commit"] == git(repo, "rev-parse", "HEAD").strip()[:12]
    assert payload["upstream"] is None
    assert "ahead" not in payload
    assert payload["workspace_is_repository_root"] is True
    assert "nothing was fetched" in payload["note"]


def test_modified_file_is_unstaged(repo: Path) -> None:
    (repo / "app.py").write_text("print('changed')\n", encoding="utf-8")
    payload = git_status_payload()
    assert payload["unstaged"] == [{"path": "app.py", "change": "modified"}]
    assert payload["staged"] == []
    assert payload["clean"] is False


def test_staged_files(repo: Path) -> None:
    (repo / "new.py").write_text("y = 2\n", encoding="utf-8")
    (repo / "app.py").write_text("print('changed')\n", encoding="utf-8")
    git(repo, "add", "new.py", "app.py")
    (repo / "app.py").write_text("print('changed again')\n", encoding="utf-8")
    git(repo, "rm", "-q", "lib.py")
    payload = git_status_payload()
    assert payload["staged"] == [
        {"path": "app.py", "change": "modified"},
        {"path": "lib.py", "change": "deleted"},
        {"path": "new.py", "change": "added"},
    ]
    assert payload["unstaged"] == [{"path": "app.py", "change": "modified"}]


def test_staged_rename_reports_where_it_came_from(repo: Path) -> None:
    git(repo, "mv", "lib.py", "core.py")
    payload = git_status_payload()
    assert payload["staged"] == [
        {"path": "core.py", "change": "renamed", "from": "lib.py"}
    ]


def test_untracked_files_and_folders(repo: Path) -> None:
    (repo / "notes.txt").write_text("todo\n", encoding="utf-8")
    (repo / "newdir").mkdir()
    (repo / "newdir" / "a.py").write_text("", encoding="utf-8")
    payload = git_status_payload()
    assert payload["untracked"] == ["newdir/", "notes.txt"]


def test_branch_name(repo: Path) -> None:
    git(repo, "switch", "-q", "-c", "feature/x")
    assert git_status_payload()["branch"] == "feature/x"


def test_detached_head(repo: Path) -> None:
    git(repo, "checkout", "-q", "--detach")
    payload = git_status_payload()
    assert payload["detached"] is True
    assert payload["branch"] is None
    assert payload["commit"]


def test_repository_without_commits(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    monkeypatch.setenv(WORKSPACE_ENV, str(root))
    payload = git_status_payload()
    assert payload["status"] == "ok"
    assert payload["branch"] == "main"
    assert payload["commit"] is None


def test_ahead_and_behind_against_a_local_upstream(repo: Path) -> None:
    # The upstream is a local branch: no remote exists and nothing is fetched.
    git(repo, "branch", "-q", "base")
    git(repo, "branch", "-q", "--set-upstream-to=base")
    (repo / "app.py").write_text("print('v2')\n", encoding="utf-8")
    git(repo, "commit", "-q", "-am", "ahead")
    payload = git_status_payload()
    assert payload["upstream"] == "base"
    assert (payload["ahead"], payload["behind"]) == (1, 0)


def test_non_git_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.setenv(WORKSPACE_ENV, str(plain))
    payload = git_status_payload()
    assert payload["status"] == "not_a_git_repository"
    assert str(tmp_path) not in payload["message"]


def test_unset_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(WORKSPACE_ENV, raising=False)
    assert git_status_payload()["status"] == "not_configured"


def test_workspace_inside_a_larger_repository(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sub = repo / "sub"
    (sub / "pkg").mkdir(parents=True)
    (sub / "pkg" / "mod.py").write_text("a = 1\n", encoding="utf-8")
    git(repo, "add", "sub")
    git(repo, "commit", "-q", "-m", "sub")
    (sub / "pkg" / "mod.py").write_text("a = 2\n", encoding="utf-8")
    (sub / "fresh.py").write_text("", encoding="utf-8")
    (repo / "app.py").write_text("print('outside the workspace')\n", "utf-8")
    (repo / "outside_new.txt").write_text("", encoding="utf-8")
    monkeypatch.setenv(WORKSPACE_ENV, str(sub))
    payload = git_status_payload()
    # Paths relative to the workspace; changes elsewhere in the repo left out.
    assert payload["unstaged"] == [{"path": "pkg/mod.py", "change": "modified"}]
    assert payload["untracked"] == ["fresh.py"]
    assert payload["workspace_is_repository_root"] is False
    assert "app.py" not in str(payload) and "outside_new" not in str(payload)


def test_repository_nested_inside_the_workspace(repo: Path) -> None:
    inner = repo / "inner"
    inner.mkdir()
    git(inner, "init", "-q", "-b", "main")
    (inner / "inner_file.py").write_text("", encoding="utf-8")
    payload = git_status_payload()
    # The outer repository's view: the nested repository is one untracked
    # folder, and its own files aren't listed.
    assert payload["untracked"] == ["inner/"]
    assert "inner_file" not in str(payload)


def test_protected_and_ignored_names_are_hidden(repo: Path) -> None:
    (repo / ".env.local").write_text(f"KEY={SECRET}\n", encoding="utf-8")
    (repo / "build").mkdir()
    (repo / "build" / "out.js").write_text("", encoding="utf-8")
    (repo / "visible.txt").write_text("", encoding="utf-8")
    payload = git_status_payload()
    assert payload["untracked"] == ["visible.txt"]
    assert payload["hidden_protected_or_ignored"] == 2
    assert ".env" not in str(payload) and SECRET not in str(payload)
    assert payload["clean"] is False


def test_entry_lists_are_capped(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workspace_git, "MAX_GIT_ENTRIES", 2)
    for i in range(4):
        (repo / f"u{i}.txt").write_text("", encoding="utf-8")
    payload = git_status_payload()
    assert payload["untracked"] == ["u0.txt", "u1.txt"]
    assert payload["truncated"] is True
    assert "first 2" in payload["message"]


def test_results_never_reveal_the_host_path(repo: Path, tmp_path: Path) -> None:
    (repo / "app.py").write_text("changed\n", encoding="utf-8")
    (repo / "new.txt").write_text("", encoding="utf-8")
    assert str(tmp_path) not in str(git_status_payload())


def test_only_the_two_fixed_read_only_commands_run(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    real_run = subprocess.run

    def recording_run(args, **kwargs):
        calls.append((list(args), kwargs))
        return real_run(args, **kwargs)

    monkeypatch.setattr(workspace_git.subprocess, "run", recording_run)
    assert git_status_payload()["status"] == "ok"
    assert [args[1:] for args, _ in calls] == [
        [*GIT_OPTIONS, *PREFIX_COMMAND],
        [*GIT_OPTIONS, *STATUS_COMMAND],
    ]
    for args, kwargs in calls:
        assert Path(args[0]).name in {"git", "git.exe"}
        assert kwargs.get("shell", False) is False
        assert kwargs["cwd"] == repo.resolve()
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["timeout"] > 0
        assert {k for k in kwargs["env"] if k.startswith("GIT_")} == {
            "GIT_TERMINAL_PROMPT",
            "GIT_OPTIONAL_LOCKS",
        }
        mutating = {"add", "commit", "checkout", "switch", "reset", "clean", "stash"}
        network = {"push", "pull", "fetch", "remote", "clone"}
        assert not (set(args) & (mutating | network))


def test_status_does_not_write_to_the_repository(repo: Path) -> None:
    # A touched file makes a normal `git status` refresh and rewrite the index;
    # with --no-optional-locks nothing under .git may change.
    os.utime(repo / "app.py", ns=(1, 1))

    def snapshot() -> dict[str, tuple[int, int]]:
        git_dir = repo / ".git"
        return {
            str(p): (p.stat().st_mtime_ns, p.stat().st_size)
            for p in git_dir.rglob("*")
            if p.is_file()
        }

    before = snapshot()
    index_bytes = (repo / ".git" / "index").read_bytes()
    git_status_payload()
    git_status_payload()
    assert snapshot() == before
    assert (repo / ".git" / "index").read_bytes() == index_bytes


@pytest.mark.skipif(os.name == "nt", reason="uses a POSIX shell script as hook")
def test_repository_config_cannot_start_an_fsmonitor_hook(
    repo: Path, tmp_path: Path
) -> None:
    marker = tmp_path / "hook-ran"
    hook = tmp_path / "hook.sh"
    hook.write_text(f"#!/bin/sh\ntouch '{marker}'\n", encoding="utf-8")
    hook.chmod(0o755)
    git(repo, "config", "core.fsmonitor", str(hook))
    git(repo, "status")  # precondition: a plain status does run the hook
    if not marker.exists():
        pytest.skip("this git version doesn't run the fsmonitor hook for status")
    marker.unlink()
    assert git_status_payload()["status"] == "ok"
    assert not marker.exists()


def test_git_environment_variables_cannot_redirect_git(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    git(other, "init", "-q", "-b", "elsewhere")
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    assert git_status_payload()["branch"] == "main"


def test_git_program_inside_the_workspace_is_refused(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    planted = repo / "git"
    planted.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(workspace_git.shutil, "which", lambda name: str(planted))
    monkeypatch.setattr(
        workspace_git.subprocess,
        "run",
        lambda *a, **k: pytest.fail("no process may be started"),
    )
    payload = git_status_payload()
    assert payload["status"] == "git_unavailable"
    assert str(repo) not in payload["message"]


def test_missing_git_is_controlled(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workspace_git.shutil, "which", lambda name: None)
    assert git_status_payload()["status"] == "git_unavailable"


def test_git_errors_never_echo_git_output(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_run(args, **kwargs):
        return subprocess.CompletedProcess(
            args, 1, b"", b"fatal: something broke at /home/someone/private/repo"
        )

    monkeypatch.setattr(workspace_git.subprocess, "run", failing_run)
    payload = git_status_payload()
    assert payload["status"] == "git_error"
    assert "/home/someone" not in payload["message"]


def test_timeout_is_controlled(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def slow_run(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(workspace_git.subprocess, "run", slow_run)
    payload = git_status_payload()
    assert payload["status"] == "git_error"
    assert "seconds" in payload["message"]


async def test_agent_tool_returns_the_payload(repo: Path) -> None:
    agent = Painthaker(model=ScriptedLLM([]))
    tools = {tool.info.name: tool for tool in agent.tools}
    description = tools["git_status"].info.description
    assert "Read-only" in description and "never contacts a remote" in description
    payload = await agent.git_status(None)
    assert payload["branch"] == "main"
