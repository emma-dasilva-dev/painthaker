# Offline tests for src/notes.py (local notes search). No LLM or network calls.
# tests/fixtures/notes holds small synthetic notes with invented facts, plus a
# hidden folder, a hidden file and an unsupported .pdf that must be ignored.

import os
import shutil
from pathlib import Path

import pytest

import notes
from notes import NOTES_DIR_ENV, query_terms, search, search_notes_payload

FIXTURES = Path(__file__).parent / "fixtures" / "notes"


@pytest.fixture
def notes_dir(tmp_path: Path) -> Path:
    """A writable copy of the fixtures."""
    root = tmp_path / "notes"
    shutil.copytree(FIXTURES, root)
    return root


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").split("\n")


def _assert_line_ranges_are_exact(root: Path, result) -> None:
    for excerpt in result.excerpts:
        lines = _lines(root / excerpt.source)
        assert excerpt.text == "\n".join(
            lines[excerpt.start_line - 1 : excerpt.end_line]
        )


def test_finds_relevant_passage_with_exact_file_and_lines() -> None:
    result = search("Quel canal Wi-Fi utilise le routeur du labo Baobab ?", FIXTURES)
    assert result.terms == ["canal", "wi", "fi", "utilise", "routeur", "labo", "baobab"]
    best = result.excerpts[0]
    assert best.source == "reseau.md"
    assert "Le routeur principal utilise le canal Wi-Fi 11." in best.text
    assert best.start_line <= 4 <= best.end_line  # the fact is on line 4
    _assert_line_ranges_are_exact(FIXTURES, result)
    assert result.incomplete == []


def test_accents_are_ignored_for_matching_and_kept_in_excerpts() -> None:
    result = search("securite", FIXTURES)
    [excerpt] = result.excerpts
    assert excerpt.source == "reseau.md"
    assert "## Sécurité" in excerpt.text
    result = search("sauvegardes verifiees Aicha", FIXTURES)
    assert result.excerpts[0].source == "projets/serveur.txt"
    assert "vérifiées chaque lundi par Aïcha." in result.excerpts[0].text
    _assert_line_ranges_are_exact(FIXTURES, result)


def test_multiline_passage_keeps_its_lines_together() -> None:
    result = search("sauvegardes nocturnes lundi", FIXTURES)
    excerpt = result.excerpts[0]
    assert (excerpt.source, excerpt.start_line, excerpt.end_line) == (
        "projets/serveur.txt",
        3,
        7,
    )
    assert excerpt.text.split("\n")[2:4] == [
        "Les sauvegardes nocturnes démarrent à 02:30",
        "et sont vérifiées chaque lundi par Aïcha.",
    ]


def test_no_matches_returns_nothing_and_no_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(NOTES_DIR_ENV, str(FIXTURES))
    payload = search_notes_payload("adresse IP de l'imprimante")
    assert payload["status"] == "no_matches"
    assert payload["excerpts"] == []


def test_hidden_and_unsupported_files_are_never_searched() -> None:
    result = search("routeur Baobab canal", FIXTURES)
    sources = {e.source for e in result.excerpts}
    assert sources == {"reseau.md"}
    assert all(
        "canal 6" not in e.text and "canal 13" not in e.text for e in result.excerpts
    )


def test_edits_and_deletions_show_up_on_the_next_search(notes_dir: Path) -> None:
    assert "canal Wi-Fi 11" in search("canal routeur", notes_dir).excerpts[0].text
    (notes_dir / "reseau.md").write_text(
        "Notes réseau\nLe routeur a été passé sur le canal 36.\n", encoding="utf-8"
    )
    [excerpt] = search("canal routeur", notes_dir).excerpts
    assert (excerpt.start_line, excerpt.end_line) == (1, 3)
    assert "canal 36" in excerpt.text and "canal Wi-Fi 11" not in excerpt.text

    (notes_dir / "reseau.md").unlink()
    assert search("canal routeur", notes_dir).excerpts == []
    assert search("sauvegardes lundi", notes_dir).excerpts  # other files still found


def test_symlinks_cannot_escape_the_notes_folder(
    notes_dir: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text(
        "routeur Baobab secret: hunter2\n", encoding="utf-8"
    )
    (notes_dir / "lien.md").symlink_to(outside / "secret.md")
    (notes_dir / "dossier").symlink_to(outside, target_is_directory=True)

    result = search("routeur Baobab secret", notes_dir)
    assert all("hunter2" not in e.text for e in result.excerpts)
    assert {e.source for e in result.excerpts} == {"reseau.md"}
    assert "2 symbolic link(s) skipped" in result.incomplete


def test_relative_and_dotted_folder_paths_resolve_to_the_same_files(
    notes_dir: Path,
) -> None:
    dotted = notes_dir / "projets" / ".." / "."
    result = search("canal routeur", dotted)
    assert {e.source for e in result.excerpts} == {"reseau.md"}


def test_size_limits_are_reported(
    notes_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (notes_dir / "gros.md").write_text("routeur canal\n" * 100, encoding="utf-8")
    monkeypatch.setattr(notes, "MAX_FILE_BYTES", 400)
    result = search("routeur canal", notes_dir)
    assert "gros.md: larger than the 400 byte limit, skipped" in result.incomplete
    assert {e.source for e in result.excerpts} == {"reseau.md"}

    monkeypatch.setattr(notes, "MAX_FILES", 1)
    result = search("routeur canal", notes_dir)
    assert any("only the first 1 of" in reason for reason in result.incomplete)


def test_output_is_bounded_and_the_cut_is_reported(
    notes_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for n in range(10):
        (notes_dir / f"routeur-{n}.md").write_text(
            f"Le routeur {n} est sur le canal {n}.\n", encoding="utf-8"
        )
    result = search("routeur canal", notes_dir)
    assert len(result.excerpts) == notes.MAX_EXCERPTS
    assert any("lower-ranked excerpt(s) not shown" in r for r in result.incomplete)

    monkeypatch.setattr(notes, "MAX_OUTPUT_CHARS", 80)
    result = search("routeur canal", notes_dir)
    assert sum(len(e.text) for e in result.excerpts) <= 80
    assert "output size limit reached; some excerpts omitted" in result.incomplete


def test_unreadable_files_are_reported(notes_dir: Path) -> None:
    (notes_dir / "latin1.txt").write_bytes("routeur canal défaut".encode("latin-1"))
    result = search("routeur canal", notes_dir)
    assert "latin1.txt: not valid UTF-8, skipped" in result.incomplete


def test_not_configured_and_missing_folder_explain_what_to_do(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(NOTES_DIR_ENV, raising=False)
    payload = search_notes_payload("routeur")
    assert payload["status"] == "not_configured"
    assert NOTES_DIR_ENV in payload["message"] and ".env.local" in payload["message"]

    monkeypatch.setenv(NOTES_DIR_ENV, str(tmp_path / "absent"))
    assert search_notes_payload("routeur")["status"] == "folder_not_found"


def test_payload_carries_sources_and_the_untrusted_content_notice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(NOTES_DIR_ENV, str(FIXTURES))
    payload = search_notes_payload("canal routeur Baobab")
    assert payload["status"] == "ok" and payload["search_complete"] is True
    first = payload["excerpts"][0]
    assert first["source"] == f"reseau.md:{first['start_line']}-{first['end_line']}"
    assert "not instructions" in payload["note"]


def test_question_words_and_filler_are_not_search_terms() -> None:
    assert query_terms("Que disent mes notes sur le routeur ?") == ["disent", "routeur"]
    assert query_terms("What do my notes say about the router?") == ["say", "router"]
    assert query_terms("?!") == []


def test_fixture_files_exist_as_expected() -> None:
    # Guard: the hidden and unsupported fixtures must be present for the tests
    # above to mean anything.
    for name in (".brouillons/cache.md", ".cache-routeur.md", "export.pdf"):
        assert (FIXTURES / name).is_file(), name
    assert not os.path.islink(FIXTURES / "reseau.md")


def test_a_single_matching_word_still_returns_the_passage() -> None:
    # Live regression: the model searched "température d'impression Tisserande"
    # style queries where only one word appears in the note; a fixed "at least
    # two words" filter dropped the only relevant passage.
    result = search("ordinateurs inventaire matériel", FIXTURES)
    assert [e.source for e in result.excerpts] == ["reseau.md"]
    assert "8 ordinateurs" in result.excerpts[0].text


def test_weaker_matches_are_dropped_when_much_better_ones_exist(
    notes_dir: Path,
) -> None:
    (notes_dir / "bruit.md").write_text(
        "Un mot sur le canal de Suez.\n", encoding="utf-8"
    )
    result = search("canal Wi-Fi routeur Baobab", notes_dir)
    assert {e.source for e in result.excerpts} == {"reseau.md"}  # "canal" alone: noise
