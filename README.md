# Painthaker

Painthaker is an AI learning companion for developers and cybersecurity learners. It explains security concepts, reviews code, and separates confirmed vulnerabilities from patterns that only look risky. It answers in French by default and in English when you write in English or ask for it.

It is built with [LiveKit Agents](https://docs.livekit.io/agents/) (Python) and uses Google Gemini (`gemini-3.5-flash-lite`) as its model. The application lives in [`my-agent/`](my-agent/), which started from LiveKit's Python agent starter; [`my-agent/README.md`](my-agent/README.md) has the full details.

## Status

**Working today (text chat in a terminal, Ubuntu / WSL):**

- **Chat:** a local terminal chat (`my-agent/scripts/chat.sh`) that accepts pasted messages and multiline code with indentation intact. Enter sends, Alt+Enter adds a new line.
- **Teaching style:** French by default with English on request, concise answers in Markdown with code blocks, harmless attack demonstrations, and fixes that suit the task.
- **`inspect_code` tool:** a rule-based scan for risky patterns (`eval`/`exec`, `shell=True`, `innerHTML`, hardcoded secrets, SQL built from strings). The reply keeps *detected patterns*, *confirmed vulnerabilities* and *possible concerns* apart.
- **Current date and time:** taken from the machine's clock in your timezone (default `Africa/Porto-Novo`, set with `PAINTHAKER_TIMEZONE`).
- **Saved conversations** in a local SQLite file. Each start opens a new chat, and you can go back to an earlier one:
  - `/list` shows saved conversations, `/resume <id>` reopens one, `/new` starts another, `/delete <id>` removes one after confirmation, and `/help` shows the commands;
  - failed or interrupted replies are never saved;
  - the model sees the most recent exchanges within a size budget.
- **Tests:** offline tests that need no API key and run in CI, plus opt-in live tests against Gemini.

- **Notes search:** ask about your own `.md`/`.txt` notes ("D'après mes notes, …"). Painthaker searches the folder set in `PAINTHAKER_NOTES_DIR` with a simple keyword search and cites `file:start-end`. When the retrieved passages don't answer the question, it says it couldn't find the answer in them, not that your notes lack it. See [Notes search](#notes-search).

**Not built yet (planned):** voice conversations (the STT/TTS pipeline from the starter is configured but not in use), long-term memory of preferences, meaning-based retrieval (embeddings), more tools, and a web frontend. Nothing is deployed.

## Setup (Ubuntu / WSL)

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.11, which uv can install. The [LiveKit CLI](https://docs.livekit.io/intro/basics/cli/) is optional; you only need it for `lk agent debugger` and simulations.

```bash
git clone https://github.com/emma-dasilva-dev/painthaker.git
cd painthaker/my-agent

# One-time: create the Linux environment outside the repository.
UV_PROJECT_ENVIRONMENT="$HOME/.virtualenvs/painthaker" uv sync --locked --python 3.11

cp .env.example .env.local   # then fill it in (see Credentials)
```

The environment lives outside the repository on purpose. If the project folder is shared with Windows, its `.venv` belongs to Windows, and running plain `uv run` from Linux would replace it.

## Launch

From any directory:

```bash
path/to/painthaker/my-agent/scripts/chat.sh
```

The launcher does the following:
- uses `~/.virtualenvs/painthaker`, or `PAINTHAKER_VENV` if set;
- checks without installing anything that the environment matches `uv.lock`;
- if the environment is missing or out of date, prints the exact setup command instead of starting.

## Credentials

Put these in `my-agent/.env.local`. Git ignores that file; never commit it.

| Variable | Needed for |
|---|---|
| `GOOGLE_API_KEY` | Every model reply in the chat, and the live tests |
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | LiveKit CLI features such as simulations, and running the agent against LiveKit Cloud |

The offline tests and CI need no credentials.

## Notes search

1. Put UTF-8 `.md` or `.txt` notes in a folder outside the repository, for example `~/notes`.
2. Add `PAINTHAKER_NOTES_DIR=/home/you/notes` to `my-agent/.env.local`, then restart the chat.
3. Ask, for example: *« D'après mes notes, quel canal Wi-Fi utilise le routeur du labo ? »*

Painthaker calls the read-only `search_notes` tool. The reply cites sources such as `reseau.md:3-7`, keeps what your notes say separate from general explanations, and says when it couldn't find the answer in the retrieved passages. It never claims your notes don't contain something, since a keyword search can miss differently worded notes, and it mentions when a search was incomplete. If the variable isn't set, ordinary chat works normally and the tool explains how to enable it.

**How it searches:** plain keyword matching, with accents and case ignored and common words skipped. Excerpts include a few lines of context, are ranked by how many different query words they contain, and are read fresh on every search, so edits and deletions apply immediately. It has no synonyms or meaning-based matching, so a question phrased very differently from your notes can miss them.

**Boundaries:**
- Only non-hidden `.md`/`.txt` files inside the folder are read. Symbolic links are skipped, and files are opened so nothing outside the folder can be reached.
- Limits: 512 KB per file, 1 000 files and about 6 000 characters of excerpts per search. Anything skipped (too large, not UTF-8, a link) or cut is reported as an incomplete search.
- Note contents are treated as data, never as instructions, and are never logged.

**Privacy:** retrieved excerpts are sent to the model provider (Google Gemini) to write the answer. They are also saved in the conversation history file, as part of the tool result. Keep the notes folder outside the repository; a `notes/` folder inside it is ignored by Git.

## Conversation history

- **Storage:** conversations are saved to `~/.local/share/painthaker/history.sqlite3` (or `$XDG_DATA_HOME/painthaker/…`; override with `PAINTHAKER_HISTORY_DB`). The file is readable only by your user, but it is **not encrypted**.
- **What is saved:** only your messages, tool calls with their results, and replies. Instructions, the date and API keys are never stored.
- **Deleting:** `/delete` removes a conversation from the database file. It isn't secure erasure: journals, filesystems, SSDs and backups may keep copies.

See [`my-agent/README.md`](my-agent/README.md#conversation-history) for the context limits and failure handling.

## Tests

From `my-agent/`, with the Linux environment selected (`export UV_PROJECT_ENVIRONMENT="$HOME/.virtualenvs/painthaker"`):

```bash
uv run --no-sync ruff format --check && uv run --no-sync ruff check
uv run --no-sync pytest                                    # offline only: no API key, no network calls to models
uv run --no-sync pytest -m live tests/test_agent.py        # live: calls Gemini, uses quota
uv run --no-sync pytest -m live tests/test_response_quality.py
```

Tests marked `live` are deselected unless you pass `-m live`. The [CI workflow](.github/workflows/ci.yml) runs formatting, lint and the offline tests on every pull request.

## Current limitations

- **Platform:** Ubuntu / WSL only. On native Windows the agent crashes at startup (`0xc0000005`, not diagnosed).
- **Model behavior:** the prompt and checks make mistakes less likely, but don't prevent them. Each fix rests on a small number of live checks. Absolute claims, invented details and language slips can still happen.
- **LiveKit console:** `lk agent console --text` (lk 2.18.8) drops pasted text; use `scripts/chat.sh`.
- **Gemini free tier:** about 15 requests per minute. Provider errors (429, 504) print a traceback; the turn is discarded, not saved.
- **Resuming after tool calls:** Gemini's thought signatures aren't persisted, so resumed tool history is sent without them. Gemini accepted this in testing.
- **Context budget:** counted in characters, not tokens, so the token count is approximate.
- **Not run yet:** the LiveKit Cloud simulations in `my-agent/scenarios.yaml`.

[`my-agent/KNOWN_ISSUES.md`](my-agent/KNOWN_ISSUES.md) has the details.

## License

MIT; see [`my-agent/LICENSE`](my-agent/LICENSE) (inherited from the LiveKit starter).
