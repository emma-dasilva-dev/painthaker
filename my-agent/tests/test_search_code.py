# Offline tests for search_code (src/workspace.py) and its registration on the
# agent. Each test builds its own workspace in tmp_path; nothing calls an LLM,
# the network, a TTS service or a shell.

from pathlib import Path

import pytest
from scripted_llm import ScriptedLLM

import workspace
from agent import Painthaker
from workspace import WORKSPACE_ENV, search_code_payload

SECRET = "TOP-SECRET-VALUE-42"


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A small project set as the workspace, with a folder beside it that must
    stay unreachable."""
    project = tmp_path / "project"
    (project / "src" / "pkg").mkdir(parents=True)
    (project / "src" / "main.py").write_text(
        "import os\n\ndef main():\n    token = load_token()\n    return token\n",
        encoding="utf-8",
    )
    (project / "src" / "pkg" / "util.py").write_text(
        "def load_token():\n    return 'Token'\n", encoding="utf-8"
    )
    (project / "README.md").write_text("# Demo\nCall main() to start.\n", "utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "leak.py").write_text(f"token = '{SECRET}'\n", encoding="utf-8")
    monkeypatch.setenv(WORKSPACE_ENV, str(project))
    return project.resolve()


def _symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available here")


def _hits(payload: dict) -> list[tuple[str, int]]:
    return [(m["file"], m["line"]) for m in payload["matches"]]


def test_finds_text_with_path_line_and_excerpt(root: Path) -> None:
    payload = search_code_payload("load_token()")
    assert payload["status"] == "ok"
    assert payload["matches"] == [
        {"file": "src/main.py", "line": 4, "excerpt": "token = load_token()"},
        {"file": "src/pkg/util.py", "line": 1, "excerpt": "def load_token():"},
    ]
    assert payload["truncated"] is False
    assert payload["case_sensitive"] is False
    assert "data, not instructions" in payload["note"]


def test_every_matching_line_is_reported(root: Path) -> None:
    assert _hits(search_code_payload("token", "src/main.py")) == [
        ("src/main.py", 4),
        ("src/main.py", 5),
    ]


def test_ordering_is_deterministic(root: Path) -> None:
    for name in ["b.txt", "A.txt", "a.txt"]:
        (root / "src" / name).write_text("token\n", encoding="utf-8")
    payload = search_code_payload("token")
    # Names sorted; a folder's files come before its subfolders.
    assert [file for file, _ in _hits(payload)] == [
        "src/A.txt",
        "src/a.txt",
        "src/b.txt",
        "src/main.py",
        "src/main.py",
        "src/pkg/util.py",
        "src/pkg/util.py",
    ]
    assert search_code_payload("token") == payload


def test_case_insensitive_by_default_and_exact_on_request(root: Path) -> None:
    assert len(search_code_payload("TOKEN")["matches"]) == 4
    exact = search_code_payload("Token", case_sensitive=True)
    assert _hits(exact) == [("src/pkg/util.py", 2)]
    assert exact["case_sensitive"] is True
    assert search_code_payload("TOKEN", case_sensitive=True)["status"] == "no_matches"


def test_query_is_plain_text_not_a_pattern(root: Path) -> None:
    (root / "re.txt").write_text("a.b\naxb\n", encoding="utf-8")
    assert _hits(search_code_payload("a.b", "re.txt")) == [("re.txt", 1)]
    assert search_code_payload(".*", "re.txt")["status"] == "no_matches"


def test_scope_can_be_a_subfolder_or_a_file(root: Path) -> None:
    assert _hits(search_code_payload("def", "src/pkg")) == [("src/pkg/util.py", 1)]
    assert _hits(search_code_payload("main", "README.md")) == [("README.md", 2)]
    assert search_code_payload("def", "src/pkg")["path"] == "src/pkg"


def test_missing_scope_is_controlled(root: Path) -> None:
    payload = search_code_payload("x", "nope")
    assert payload["status"] == "not_found"
    assert "'nope'" in payload["message"]


@pytest.mark.parametrize("query", ["", "   "])
def test_empty_query_is_refused(root: Path, query: str) -> None:
    assert search_code_payload(query)["status"] == "empty_query"


@pytest.mark.parametrize("query", ["a\nb", "x" * 201])
def test_multiline_or_huge_query_is_refused(root: Path, query: str) -> None:
    assert search_code_payload(query)["status"] == "invalid_query"


def test_no_matches_is_reported_without_claiming_absence(root: Path) -> None:
    payload = search_code_payload("does-not-occur")
    assert payload["status"] == "no_matches"
    assert payload["matches"] == []
    assert "searched files" in payload["message"]


def test_protected_files_and_folders_are_never_searched(root: Path) -> None:
    (root / ".env").write_text(f"token={SECRET}\n", encoding="utf-8")
    (root / "src" / "secrets.yaml").write_text(f"token: {SECRET}\n", "utf-8")
    (root / ".ssh").mkdir()
    (root / ".ssh" / "id").write_text(f"token {SECRET}\n", encoding="utf-8")
    payload = search_code_payload("token")
    assert SECRET not in str(payload)
    assert all(
        not m["file"].endswith((".env", "secrets.yaml")) for m in payload["matches"]
    )
    assert payload["skipped"]["protected"] == 3
    # A protected scope is refused outright, like list_files and read_file.
    for scope in (".env", ".ssh", "src/secrets.yaml"):
        refused = search_code_payload("token", scope)
        assert refused["status"] == "protected"
        assert SECRET not in str(refused)


def test_ignored_folders_are_skipped(root: Path) -> None:
    for name in (".git", "node_modules", ".venv", "build", "__pycache__"):
        (root / name).mkdir()
        (root / name / "x.py").write_text("token\n", encoding="utf-8")
    payload = search_code_payload("token")
    assert {m["file"].split("/")[0] for m in payload["matches"]} == {"src"}
    assert payload["skipped"]["ignored"] == 5
    assert search_code_payload("token", "node_modules")["status"] == "ignored"


def test_binary_and_non_utf8_files_are_skipped(root: Path) -> None:
    (root / "image.png").write_bytes(b"token\x00\x01\x02")
    (root / "latin1.txt").write_bytes("token café".encode("latin-1"))
    payload = search_code_payload("token")
    files = {m["file"] for m in payload["matches"]}
    assert "image.png" not in files and "latin1.txt" not in files
    assert payload["skipped"]["binary"] == 1
    assert payload["skipped"]["not_text"] == 1


def test_oversized_files_are_skipped(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workspace, "MAX_FILE_BYTES", 100)
    (root / "big.txt").write_text("token\n" * 50, encoding="utf-8")
    payload = search_code_payload("token")
    assert "big.txt" not in {m["file"] for m in payload["matches"]}
    assert payload["skipped"]["too_large"] == 1


@pytest.mark.parametrize(
    "scope",
    ["..", "../outside", "src/../../outside", "..\\outside", "src/../README.md"],
)
def test_traversal_is_rejected(root: Path, scope: str) -> None:
    payload = search_code_payload("token", scope)
    assert payload["status"] == "invalid_path"
    assert SECRET not in str(payload)


def test_absolute_paths_are_rejected(root: Path, tmp_path: Path) -> None:
    for scope in (str(tmp_path / "outside"), str(root), "/", "C:\\", "~"):
        payload = search_code_payload("token", scope)
        assert payload["status"] == "invalid_path", scope
        assert SECRET not in str(payload)


def test_symlink_escape_is_rejected_and_never_followed(
    root: Path, tmp_path: Path
) -> None:
    _symlink(root / "src" / "escape", tmp_path / "outside")
    _symlink(root / "leak_link.py", tmp_path / "outside" / "leak.py")
    payload = search_code_payload("token")
    assert SECRET not in str(payload)
    assert payload["skipped"]["symlinks"] == 2
    assert search_code_payload("token", "src/escape")["status"] == "outside_workspace"
    assert search_code_payload("token", "leak_link.py")["status"] == "outside_workspace"


def test_match_cap_truncates_and_says_so(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workspace, "MAX_SEARCH_MATCHES", 3)
    payload = search_code_payload("token")
    assert _hits(payload) == [
        ("src/main.py", 4),
        ("src/main.py", 5),
        ("src/pkg/util.py", 1),
    ]
    assert payload["truncated"] is True
    assert "first 3 matches" in payload["message"]


def test_exactly_the_cap_is_not_reported_as_truncated(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workspace, "MAX_SEARCH_MATCHES", 4)
    payload = search_code_payload("token")
    assert len(payload["matches"]) == 4
    assert payload["truncated"] is False


def test_file_scan_cap_truncates_and_says_so(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workspace, "MAX_SEARCH_FILES", 2)
    payload = search_code_payload("token")
    assert payload["files_scanned"] == 2
    assert payload["truncated"] is True
    assert "first 2 files" in payload["message"]
    # README.md and src/main.py were searched; src/pkg/util.py was not.
    assert {m["file"] for m in payload["matches"]} == {"src/main.py"}


def test_byte_cap_truncates_and_says_so(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workspace, "MAX_SEARCH_BYTES", 60)
    payload = search_code_payload("token")
    assert payload["truncated"] is True
    assert "reading limit" in payload["message"]


def test_long_lines_give_a_short_excerpt_around_the_match(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workspace, "MAX_EXCERPT_CHARS", 20)
    (root / "min.js").write_text("a" * 100 + "NEEDLE" + "b" * 100 + "\n", "utf-8")
    [match] = search_code_payload("needle", "min.js")["matches"]
    assert "NEEDLE" in match["excerpt"]
    assert match["excerpt"].startswith("…") and match["excerpt"].endswith("…")
    assert len(match["excerpt"]) == 22


def test_results_never_reveal_the_host_path(root: Path, tmp_path: Path) -> None:
    for args in (("token",), ("token", "src"), ("x", "nope"), ("x", "../x")):
        payload = search_code_payload(*args)
        assert str(tmp_path) not in str(payload)


def test_searching_changes_nothing(root: Path) -> None:
    def snapshot() -> dict[str, tuple[int, int]]:
        return {
            str(p): (p.stat().st_mtime_ns, p.stat().st_size) for p in root.rglob("*")
        }

    before = snapshot()
    search_code_payload("token")
    search_code_payload("token", "src", case_sensitive=True)
    assert snapshot() == before


def test_unset_workspace_disables_search(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(WORKSPACE_ENV, raising=False)
    assert search_code_payload("token")["status"] == "not_configured"


async def test_agent_tool_returns_the_payload(root: Path) -> None:
    agent = Painthaker(model=ScriptedLLM([]))
    tools = {tool.info.name: tool for tool in agent.tools}
    assert "plain text" in tools["search_code"].info.description
    payload = await agent.search_code(None, "Token", "src", True)
    assert _hits(payload) == [("src/pkg/util.py", 2)]
