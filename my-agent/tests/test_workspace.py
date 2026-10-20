# Offline tests for src/workspace.py (read-only workspace tools) and their
# registration on the agent. Each test builds its own workspace in tmp_path;
# nothing calls an LLM, the network, a TTS service or a shell.

import ast
import os
from pathlib import Path

import pytest
from scripted_llm import ScriptedLLM

import workspace
from agent import Painthaker
from workspace import (
    WORKSPACE_ENV,
    WorkspaceError,
    is_protected,
    list_files_payload,
    read_file_payload,
    resolve_in_workspace,
    workspace_root,
)

SECRET = "TOP-SECRET-VALUE-42"


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A small project set as the workspace, with a folder beside it that must
    stay unreachable."""
    project = tmp_path / "project"
    (project / "src" / "pkg").mkdir(parents=True)
    (project / "src" / "main.py").write_text(
        "def main():\n    return 1\n", encoding="utf-8"
    )
    (project / "src" / "pkg" / "deep.py").write_text("x = 1\n", encoding="utf-8")
    (project / "README.md").write_text("# Demo\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.txt").write_text(SECRET, encoding="utf-8")
    monkeypatch.setenv(WORKSPACE_ENV, str(project))
    return project.resolve()


def _symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available here")


def _status(path: str) -> str:
    with pytest.raises(WorkspaceError) as caught:
        resolve_in_workspace(workspace_root(), path)
    return caught.value.status


# --- workspace root ---------------------------------------------------------


def test_unset_workspace_disables_the_tools_with_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(WORKSPACE_ENV, raising=False)
    for payload in (list_files_payload("."), read_file_payload("README.md")):
        assert payload["status"] == "not_configured"
        assert WORKSPACE_ENV in payload["message"]


@pytest.mark.parametrize("kind", ["missing", "relative", "file"])
def test_unusable_workspace_setting_is_refused(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a-file").write_text("x", encoding="utf-8")
    value = {
        "missing": str(tmp_path / "nope"),
        "relative": "some/relative/dir",
        "file": str(tmp_path / "a-file"),
    }[kind]
    monkeypatch.setenv(WORKSPACE_ENV, value)
    payload = list_files_payload(".")
    assert payload["status"] == "workspace_unavailable"
    assert str(tmp_path) not in payload["message"]


def test_filesystem_root_is_not_accepted_as_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(WORKSPACE_ENV, tmp_path.anchor)
    assert list_files_payload(".")["status"] == "workspace_unavailable"


def test_workspace_root_is_canonical_even_through_a_link(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = tmp_path / "project-link"
    _symlink(link, root)
    monkeypatch.setenv(WORKSPACE_ENV, str(link))
    assert workspace_root() == root
    assert read_file_payload("README.md")["status"] == "ok"


# --- path safety ------------------------------------------------------------


def test_normal_path_resolves_inside_workspace(root: Path) -> None:
    real, shown = resolve_in_workspace(root, "src/main.py")
    assert real == root / "src" / "main.py"
    assert shown == "src/main.py"


@pytest.mark.parametrize("path", [".", "", "./", "  .  "])
def test_workspace_root_itself_works(root: Path, path: str) -> None:
    assert resolve_in_workspace(root, path) == (root, ".")
    assert list_files_payload(path)["status"] == "ok"


def test_backslashes_are_read_as_separators(root: Path) -> None:
    assert resolve_in_workspace(root, "src\\main.py")[1] == "src/main.py"


@pytest.mark.parametrize(
    "path",
    [
        "..",
        "../outside/private.txt",
        "src/../../outside/private.txt",
        "src/pkg/../../../outside",
        "src/../README.md",  # stays inside, but '..' is refused outright
        "..\\outside\\private.txt",
        "src\\..\\..\\outside",
    ],
)
def test_traversal_is_rejected(root: Path, path: str) -> None:
    assert _status(path) == "invalid_path"
    assert read_file_payload(path)["status"] == "invalid_path"


def test_absolute_paths_are_rejected_inside_or_outside(
    root: Path, tmp_path: Path
) -> None:
    paths = [
        "/etc/passwd",
        str(tmp_path / "outside" / "private.txt"),
        str(root / "README.md"),  # absolute even though it is inside
        "C:\\Windows\\win.ini",
        "c:/Users",
        "C:relative",
        "\\\\server\\share\\file",
        "//server/share",
        "~/.ssh/id_rsa",
        "~",
    ]
    for path in paths:
        assert _status(path) == "invalid_path", path
        payload = read_file_payload(path)
        assert payload["status"] == "invalid_path"
        assert SECRET not in str(payload)


def test_nul_character_is_rejected(root: Path) -> None:
    assert _status("README.md\x00.txt") == "invalid_path"


def test_symlinked_folder_pointing_outside_is_rejected(
    root: Path, tmp_path: Path
) -> None:
    _symlink(root / "escape", tmp_path / "outside")
    assert _status("escape") == "outside_workspace"
    assert _status("escape/private.txt") == "outside_workspace"
    for payload in (
        list_files_payload("escape"),
        read_file_payload("escape/private.txt"),
    ):
        assert payload["status"] == "outside_workspace"
        assert SECRET not in str(payload)
        assert str(tmp_path) not in payload["message"]


def test_symlinked_file_pointing_outside_is_rejected(
    root: Path, tmp_path: Path
) -> None:
    _symlink(root / "src" / "looks_fine.py", tmp_path / "outside" / "private.txt")
    payload = read_file_payload("src/looks_fine.py")
    assert payload["status"] == "outside_workspace"
    assert SECRET not in str(payload)


def test_symlink_inside_workspace_is_followed(root: Path) -> None:
    _symlink(root / "main_link.py", root / "src" / "main.py")
    payload = read_file_payload("main_link.py")
    assert payload["status"] == "ok"
    assert "def main():" in payload["content"]


def test_symlink_to_a_protected_file_is_rejected(root: Path) -> None:
    (root / ".env").write_text(f"API_KEY={SECRET}\n", encoding="utf-8")
    _symlink(root / "notes.txt", root / ".env")
    payload = read_file_payload("notes.txt")
    assert payload["status"] == "protected"
    assert SECRET not in str(payload)


def test_dangling_and_looping_symlinks_are_controlled(root: Path) -> None:
    _symlink(root / "dangling", root / "missing-target")
    assert read_file_payload("dangling")["status"] == "not_found"
    loop_a, loop_b = root / "loop_a", root / "loop_b"
    _symlink(loop_a, loop_b)
    _symlink(loop_b, loop_a)
    assert read_file_payload("loop_a")["status"] in {"unreadable", "not_found"}


def test_errors_never_reveal_the_host_path(root: Path, tmp_path: Path) -> None:
    (root / "secrets.json").write_text(SECRET, encoding="utf-8")
    requests = [
        ("read", "missing.txt"),
        ("read", "src"),
        ("read", "secrets.json"),
        ("read", "../x"),
        ("list", "README.md"),
        ("list", "missing"),
    ]
    for tool, path in requests:
        payload = (read_file_payload if tool == "read" else list_files_payload)(path)
        assert payload["status"] != "ok"
        assert str(tmp_path) not in str(payload)
        assert str(root) not in str(payload)
        assert SECRET not in str(payload)


# --- sensitive files --------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        ".env",
        ".env.local",
        ".env.production",
        "server.pem",
        "private.key",
        "id_rsa",
        "id_ed25519",
        "credentials.json",
        "credentials.yml",
        "secrets.yaml",
        "secrets.toml",
        # Case and spellings a Windows filesystem (or /mnt/c) opens as these.
        ".ENV",
        "ID_RSA",
        "Secrets.Json",
    ],
)
def test_protected_files_are_never_read(root: Path, name: str) -> None:
    (root / name).write_text(f"{SECRET}\n", encoding="utf-8")
    payload = read_file_payload(name)
    assert payload["status"] == "protected"
    assert SECRET not in str(payload)
    listing = list_files_payload(".")
    assert name not in [entry["name"] for entry in listing["entries"]]
    assert listing["hidden"]["protected"] >= 1


@pytest.mark.parametrize(
    "name", [".env.", ".env ", ".env::$DATA", "id_rsa.", "secrets.json:stream"]
)
def test_windows_aliases_of_protected_names_are_refused(root: Path, name: str) -> None:
    # Refused by name before the filesystem is consulted.
    assert _status(name) == "protected"


def test_files_inside_protected_folders_are_refused(root: Path) -> None:
    (root / ".ssh").mkdir()
    (root / ".ssh" / "config").write_text(SECRET, encoding="utf-8")
    (root / "config").mkdir()
    (root / "config" / ".env").write_text(SECRET, encoding="utf-8")
    for path in (".ssh", ".ssh/config", "config/.env"):
        payload = read_file_payload(path)
        assert payload["status"] == "protected", path
        assert SECRET not in str(payload)
    assert list_files_payload(".ssh")["status"] == "protected"


@pytest.mark.parametrize(
    "name", ["id_rsa.pub", "environment.py", "keyboard.md", "my_secrets_doc.md"]
)
def test_similar_but_harmless_names_stay_readable(root: Path, name: str) -> None:
    assert not is_protected(name)
    (root / name).write_text("fine\n", encoding="utf-8")
    assert read_file_payload(name)["status"] == "ok"


# --- list_files -------------------------------------------------------------


def test_lists_files_and_folders_with_their_types(root: Path) -> None:
    payload = list_files_payload(".")
    assert payload["status"] == "ok"
    assert payload["path"] == "."
    assert payload["entries"] == [
        {"name": "src", "type": "dir"},
        {"name": "README.md", "type": "file", "size_bytes": 7},
    ]
    assert payload["truncated"] is False


def test_lists_a_subfolder_without_recursing(root: Path) -> None:
    payload = list_files_payload("src")
    names = [entry["name"] for entry in payload["entries"]]
    assert payload["path"] == "src"
    assert names == ["pkg", "main.py"]
    assert "deep.py" not in str(payload)


def test_ignored_folders_are_omitted(root: Path) -> None:
    for name in workspace.IGNORED_DIRS:
        folder = root / name.replace("*", "demo")
        folder.mkdir()
        (folder / "inside.txt").write_text(SECRET, encoding="utf-8")
    payload = list_files_payload(".")
    assert [entry["name"] for entry in payload["entries"]] == ["src", "README.md"]
    assert payload["hidden"] == {"ignored": len(workspace.IGNORED_DIRS)}
    for path in (".git", ".git/inside.txt", "node_modules/inside.txt", "build"):
        assert read_file_payload(path)["status"] == "ignored"
        assert list_files_payload(path)["status"] == "ignored"


def test_ordering_is_deterministic_folders_first(root: Path) -> None:
    for name in ["b.txt", "A.txt", "c.txt", "_x.txt"]:
        (root / "src" / "pkg" / name).write_text("", encoding="utf-8")
    for name in ["zeta", "Alpha"]:
        (root / "src" / "pkg" / name).mkdir()
    payload = list_files_payload("src/pkg")
    names = [entry["name"] for entry in payload["entries"]]
    assert names == ["Alpha", "zeta", "_x.txt", "A.txt", "b.txt", "c.txt", "deep.py"]
    assert list_files_payload("src/pkg") == payload


def test_long_listings_are_capped_and_say_so(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workspace, "MAX_LIST_ENTRIES", 3)
    for i in range(5):
        (root / f"f{i}.txt").write_text("", encoding="utf-8")
    payload = list_files_payload(".")
    assert len(payload["entries"]) == 3
    assert payload["total_entries"] == 7
    assert payload["truncated"] is True
    assert "first 3 of 7" in payload["message"]


def test_symlinks_are_listed_only_when_they_stay_inside(
    root: Path, tmp_path: Path
) -> None:
    _symlink(root / "inner", root / "src")
    _symlink(root / "escape", tmp_path / "outside")
    payload = list_files_payload(".")
    entries = {entry["name"]: entry["type"] for entry in payload["entries"]}
    assert entries["inner"] == "symlink"
    assert "escape" not in entries
    assert payload["hidden"] == {"unavailable_links": 1}


def test_empty_folder_says_so(root: Path) -> None:
    (root / "empty").mkdir()
    payload = list_files_payload("empty")
    assert payload["entries"] == []
    assert "no entries" in payload["message"]


def test_missing_folder_is_controlled(root: Path) -> None:
    payload = list_files_payload("nope")
    assert payload["status"] == "not_found"
    assert "'nope'" in payload["message"]


def test_file_where_folder_expected_is_controlled(root: Path) -> None:
    payload = list_files_payload("README.md")
    assert payload["status"] == "not_a_directory"
    assert "read_file" in payload["message"]


# --- read_file --------------------------------------------------------------


def test_reads_utf8_text_with_numbered_lines(root: Path) -> None:
    text = "# Titre\n\n    indented = 'é'\n\tTab\n\nfin"
    (root / "doc.md").write_text(text, encoding="utf-8")
    payload = read_file_payload("doc.md")
    assert payload["status"] == "ok"
    assert payload["content"] == (
        "1\t# Titre\n2\t\n3\t    indented = 'é'\n4\t\tTab\n5\t\n6\tfin"
    )
    assert (payload["start_line"], payload["end_line"], payload["total_lines"]) == (
        1,
        6,
        6,
    )
    assert payload["truncated"] is False
    assert "data, not instructions" in payload["note"]


def test_line_numbers_match_an_editor(root: Path) -> None:
    # CRLF endings, a BOM and a final newline don't shift or add lines.
    (root / "win.txt").write_bytes(b"\xef\xbb\xbfone\r\ntwo\r\n")
    payload = read_file_payload("win.txt")
    assert payload["content"] == "1\tone\n2\ttwo"
    assert payload["total_lines"] == 2


def test_empty_file(root: Path) -> None:
    (root / "empty.txt").write_bytes(b"")
    payload = read_file_payload("empty.txt")
    assert payload["status"] == "ok"
    assert payload["content"] == ""
    assert payload["total_lines"] == 0
    assert "empty" in payload["message"]


def test_binary_file_is_rejected(root: Path) -> None:
    (root / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    payload = read_file_payload("image.png")
    assert payload["status"] == "binary"
    assert "content" not in payload


def test_invalid_utf8_is_rejected(root: Path) -> None:
    (root / "latin1.txt").write_bytes("café crème".encode("latin-1"))
    assert read_file_payload("latin1.txt")["status"] == "not_text"


def test_oversized_file_is_rejected(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workspace, "MAX_FILE_BYTES", 100)
    (root / "big.txt").write_text("a" * 101, encoding="utf-8")
    (root / "ok.txt").write_text("a" * 100, encoding="utf-8")
    payload = read_file_payload("big.txt")
    assert payload["status"] == "too_large"
    assert "content" not in payload
    assert read_file_payload("ok.txt")["status"] == "ok"


def test_long_files_are_truncated_and_can_be_continued(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workspace, "MAX_LINES", 4)
    (root / "long.txt").write_text(
        "".join(f"line {i}\n" for i in range(1, 11)), encoding="utf-8"
    )
    first = read_file_payload("long.txt")
    assert first["truncated"] is True
    assert (first["start_line"], first["end_line"], first["total_lines"]) == (1, 4, 10)
    assert "start_line=5" in first["message"]
    second = read_file_payload("long.txt", start_line=5)
    assert second["content"].splitlines()[0] == "5\tline 5"
    last = read_file_payload("long.txt", start_line=9)
    assert last["content"] == "9\tline 9\n10\tline 10"
    assert last["truncated"] is False
    assert "message" not in last


def test_output_is_capped_by_characters(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Each 20-character line costs 23 characters numbered (with its newline).
    monkeypatch.setattr(workspace, "MAX_OUTPUT_CHARS", 46)
    (root / "wide.txt").write_text(
        "x" * 20 + "\n" + "y" * 20 + "\n" + "z\n", encoding="utf-8"
    )
    payload = read_file_payload("wide.txt")
    assert len(payload["content"]) <= 46
    assert payload["end_line"] == 2
    assert payload["truncated"] is True


def test_very_long_lines_are_cut_and_reported(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workspace, "MAX_LINE_CHARS", 10)
    (root / "min.js").write_text("a" * 30 + "\nok\n", encoding="utf-8")
    payload = read_file_payload("min.js")
    assert payload["content"] == "1\t" + "a" * 10 + "\n2\tok"
    assert "cut at 10" in payload["message"]


@pytest.mark.parametrize("start_line", [0, -1, 4])
def test_out_of_range_start_line_is_controlled(root: Path, start_line: int) -> None:
    payload = read_file_payload("src/main.py", start_line=start_line)
    assert payload["status"] == "invalid_range"


def test_missing_file_is_controlled(root: Path) -> None:
    assert read_file_payload("src/nope.py")["status"] == "not_found"


def test_directory_where_file_expected_is_controlled(root: Path) -> None:
    payload = read_file_payload("src")
    assert payload["status"] == "not_a_file"
    assert "list_files" in payload["message"]


def test_unsafe_and_protected_paths_are_rejected_by_read_file(root: Path) -> None:
    (root / ".env.local").write_text(SECRET, encoding="utf-8")
    assert read_file_payload("../outside/private.txt")["status"] == "invalid_path"
    assert read_file_payload(".env.local")["status"] == "protected"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs named pipes")
def test_named_pipe_is_refused_without_blocking(root: Path) -> None:
    os.mkfifo(root / "pipe")
    assert read_file_payload("pipe")["status"] == "not_a_file"


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="needs O_NOFOLLOW")
def test_file_swapped_for_a_link_after_the_check_is_not_followed(
    root: Path, tmp_path: Path
) -> None:
    # Simulates the race: the checked path is a symlink by the time it's opened.
    _symlink(root / "swapped.txt", tmp_path / "outside" / "private.txt")
    with pytest.raises(WorkspaceError) as caught:
        workspace._read_bytes(root / "swapped.txt", root, "swapped.txt")
    assert caught.value.status == "unreadable"


def test_opened_file_outside_workspace_is_refused(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulates a parent folder swapped after the check: the descriptor's real
    # location is what counts.
    monkeypatch.setattr(
        workspace, "_opened_path", lambda fd, path: tmp_path / "outside" / "x"
    )
    payload = read_file_payload("README.md")
    assert payload["status"] == "outside_workspace"


def test_reading_and_listing_change_nothing(root: Path) -> None:
    def snapshot() -> dict[str, tuple[int, int]]:
        return {
            str(p): (p.stat().st_mtime_ns, p.stat().st_size) for p in root.rglob("*")
        }

    before = snapshot()
    list_files_payload(".")
    list_files_payload("src")
    read_file_payload("src/main.py")
    read_file_payload("missing.txt")
    assert snapshot() == before


# --- agent integration ------------------------------------------------------


def test_tools_are_registered_on_painthaker() -> None:
    agent = Painthaker(model=ScriptedLLM([]))
    tools = {tool.info.name: tool for tool in agent.tools}
    # The full tool set: no write, delete, shell or command tool exists.
    assert set(tools) == {
        "inspect_code",
        "search_notes",
        "list_files",
        "read_file",
        "search_code",
        "git_status",
    }
    for name in ("list_files", "read_file", "search_code"):
        description = tools[name].info.description
        assert "workspace" in description
        assert "Read-only" in description
        assert "relative to the workspace root" in description


async def test_tools_return_payloads_when_called_directly(root: Path) -> None:
    agent = Painthaker(model=ScriptedLLM([]))
    listing = await agent.list_files(None, ".")
    assert listing["status"] == "ok"
    content = await agent.read_file(None, "src/main.py", 2)
    assert content["content"] == "2\t    return 1"


def test_workspace_module_imports_no_process_or_network_modules() -> None:
    source = Path(workspace.__file__).read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert imported == {"os", "re", "stat", "collections.abc", "fnmatch", "pathlib"}
    for call in ("system(", "popen(", "subprocess", "unlink(", "remove(", "rename("):
        assert call not in source
