"""Read-only Git status of the workspace (the git_status tool).

Runs exactly two fixed, local git commands, with nothing taken from the model:

    git rev-parse --show-prefix
    git status --porcelain=v2 --branch -z --untracked-files=normal -- .

Both run in the resolved workspace root, without a shell, with a timeout, no
stdin, and these protections:
- `--no-optional-locks`, so `status` doesn't refresh and rewrite the index;
- `-c core.fsmonitor=false`, so a repository's config can't make git start a
  hook program;
- every GIT_* variable removed from the environment (GIT_DIR, GIT_WORK_TREE,
  GIT_CONFIG_PARAMETERS, ...), so nothing can point git at another repository
  or inject settings, and GIT_TERMINAL_PROMPT=0;
- the git program found on PATH beforehand, and refused if it lies inside the
  workspace (a planted "git" there is never run).
Neither command contacts a remote: ahead/behind compares with the upstream as
last fetched. Paths are reported relative to the workspace, the `.` pathspec
leaves out changes elsewhere in an enclosing repository, protected and ignored
names are hidden as in list_files, and git's own error text (which can contain
host paths) is never returned.
"""

import os
import shutil
import subprocess
from pathlib import Path

from workspace import WorkspaceError, is_ignored, is_protected, workspace_root

GIT_TIMEOUT_SECONDS = 10
MAX_GIT_ENTRIES = 200  # per list (staged, unstaged, untracked, conflicted)

GIT_OPTIONS = ("--no-optional-locks", "-c", "core.fsmonitor=false")
PREFIX_COMMAND = ("rev-parse", "--show-prefix")
STATUS_COMMAND = (
    "status",
    "--porcelain=v2",
    "--branch",
    "-z",
    "--untracked-files=normal",
    "--",
    ".",
)

_CHANGES = {
    "M": "modified",
    "T": "type_changed",
    "A": "added",
    "D": "deleted",
    "R": "renamed",
    "C": "copied",
}

STATUS_NOTE = (
    "A read-only snapshot of the local repository, limited to the workspace. "
    "Ahead/behind compare with the upstream as last fetched: nothing was "
    "fetched, so the remote may have changed since."
)


def _git_executable(root: Path) -> str:
    found = shutil.which("git")
    if found is None:
        raise WorkspaceError("git_unavailable", "Git is not installed or not on PATH.")
    program = Path(found).resolve()
    if program.is_relative_to(root):
        raise WorkspaceError(
            "git_unavailable",
            "The git program found on PATH is inside the workspace, so it isn't run.",
        )
    return str(program)


def _git_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    # English messages, so "not a git repository" can be recognized.
    env.update(GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0", LC_ALL="C")
    return env


def _run(git: str, root: Path, command: tuple[str, ...]) -> bytes:
    try:
        done = subprocess.run(
            [git, *GIT_OPTIONS, *command],
            cwd=root,
            env=_git_env(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise WorkspaceError(
            "git_error", f"Git did not answer within {GIT_TIMEOUT_SECONDS} seconds."
        ) from None
    except OSError:
        raise WorkspaceError("git_unavailable", "Git could not be started.") from None
    if done.returncode != 0:
        # Only recognized cases are described; the raw text may name host paths.
        error = done.stderr.decode("utf-8", "replace").lower()
        if "not a git repository" in error:
            raise WorkspaceError(
                "not_a_git_repository",
                "The workspace is not inside a Git repository.",
            )
        if "dubious ownership" in error:
            raise WorkspaceError(
                "git_error",
                "Git refused the repository because another user owns it "
                "(safe.directory). Painthaker doesn't change Git settings.",
            )
        raise WorkspaceError("git_error", f"Git failed (exit code {done.returncode}).")
    return done.stdout


def _parse(output: bytes) -> tuple[dict[str, str], list[tuple[str, str, str, str]]]:
    """Branch headers, and (kind, XY, path, original path) per entry, from
    `git status --porcelain=v2 --branch -z`. Paths are repository-relative."""
    fields = output.decode("utf-8", "replace").split("\0")
    headers: dict[str, str] = {}
    entries: list[tuple[str, str, str, str]] = []
    index = 0
    while index < len(fields):
        field = fields[index]
        index += 1
        if field.startswith("# "):
            key, _, value = field[2:].partition(" ")
            headers[key] = value
        elif field.startswith("1 "):  # 1 XY sub mH mI mW hH hI path
            parts = field.split(" ", 8)
            entries.append(("change", parts[1], parts[8], ""))
        elif field.startswith("2 "):  # 2 XY sub mH mI mW hH hI score path, orig
            parts = field.split(" ", 9)
            entries.append(("change", parts[1], parts[9], fields[index]))
            index += 1
        elif field.startswith("u "):  # u XY sub m1 m2 m3 mW h1 h2 h3 path
            parts = field.split(" ", 10)
            entries.append(("conflict", parts[1], parts[10], ""))
        elif field.startswith("? "):
            entries.append(("untracked", "", field[2:], ""))
    return headers, entries


def _in_workspace(path: str, prefix: str) -> str | None:
    """The workspace-relative form of a repository path, or None if it is
    outside the workspace."""
    if not path or not path.startswith(prefix):
        return None
    return path[len(prefix) :] or None


def _hidden(path: str) -> bool:
    return any(is_protected(p) or is_ignored(p) for p in path.rstrip("/").split("/"))


def git_status() -> dict[str, object]:
    """Branch and changed files of the repository holding the workspace."""
    root = workspace_root()
    git = _git_executable(root)
    prefix = _run(git, root, PREFIX_COMMAND).decode("utf-8", "replace").strip()
    headers, entries = _parse(_run(git, root, STATUS_COMMAND))
    lists: dict[str, list] = {
        "staged": [],
        "unstaged": [],
        "untracked": [],
        "conflicted": [],
    }
    hidden = 0
    for kind, xy, repo_path, repo_origin in entries:
        path = _in_workspace(repo_path, prefix)
        if path is None:
            continue
        if _hidden(path):
            hidden += 1
            continue
        if kind == "untracked":
            lists["untracked"].append(path)
            continue
        if kind == "conflict":
            lists["conflicted"].append(path)
            continue
        origin = _in_workspace(repo_origin, prefix)
        for side, code in (("staged", xy[0]), ("unstaged", xy[1])):
            if code == ".":
                continue
            entry: dict[str, str] = {"path": path, "change": _CHANGES.get(code, code)}
            if code in "RC" and origin and not _hidden(origin):
                entry["from"] = origin
            lists[side].append(entry)

    head = headers.get("branch.head")
    oid = headers.get("branch.oid", "(initial)")
    detached = head == "(detached)"
    payload: dict[str, object] = {
        "status": "ok",
        "branch": None if detached else head,
        "detached": detached,
        "commit": None if oid == "(initial)" else oid[:12],
        "upstream": headers.get("branch.upstream"),
        "workspace_is_repository_root": prefix == "",
    }
    if "branch.ab" in headers:
        ahead, behind = headers["branch.ab"].split()
        payload["ahead"] = int(ahead.lstrip("+"))
        payload["behind"] = int(behind.lstrip("-"))
    truncated = False
    for name, items in lists.items():
        items.sort(key=lambda item: item if isinstance(item, str) else item["path"])
        truncated = truncated or len(items) > MAX_GIT_ENTRIES
        payload[name] = items[:MAX_GIT_ENTRIES]
    payload["clean"] = not hidden and not any(lists.values())
    payload["truncated"] = truncated
    if truncated:
        payload["message"] = (
            f"Some lists were cut to their first {MAX_GIT_ENTRIES} entries."
        )
    if hidden:
        payload["hidden_protected_or_ignored"] = hidden
    payload["note"] = STATUS_NOTE
    return payload


def git_status_payload() -> dict[str, object]:
    """What the git_status tool returns to the model."""
    try:
        return git_status()
    except WorkspaceError as error:
        return {"status": error.status, "message": error.message}
