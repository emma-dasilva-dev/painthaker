<a href="https://livekit.io/">
  <img src="./.github/assets/livekit-mark.png" alt="LiveKit logo" width="100" height="100">
</a>

# LiveKit Agents starter for Python

A starter project for building voice AI apps with [LiveKit Agents for Python](https://github.com/livekit/agents) and [LiveKit Cloud](https://cloud.livekit.io/).

The starter includes:

- A simple [voice AI assistant](https://docs.livekit.io/agents/start/voice-ai/) to extend and customize.
- A voice pipeline built on [LiveKit Inference](https://docs.livekit.io/agents/models/inference/), which gives you access to [models](https://docs.livekit.io/agents/models/) from top labs with no extra configuration:
  - The default LLM is Gemma 4 31B, an open-weight model [hosted by LiveKit](https://docs.livekit.io/agents/models/llm/livekit/) and tuned for voice AI.
  - The default TTS is [Fish Audio S2.1 Pro](https://docs.livekit.io/agents/models/tts/fishaudio/), an expressive and cost-effective voice.
  - More than 50 other models are available from OpenAI, Cartesia, Deepgram, and other providers.
  - [Realtime models](https://docs.livekit.io/agents/models/realtime/) and many others are available through the [plugin ecosystem](https://docs.livekit.io/agents/models/#plugins).
- [Expressive mode](https://docs.livekit.io/agents/models/tts/expressive/), on by default, so your agent's voice carries emotion and pacing that fit the conversation.
- [Keyterms](https://docs.livekit.io/agents/models/stt/keyterms/), on by default, so speech recognition gets your names, brands, and jargon right, including names it picks up during the conversation, like a caller's.
- [LiveKit Turn Detector](https://docs.livekit.io/agents/logic/turns/turn-detector/), which knows when the user has finished speaking, in 14 languages.
- [Adaptive interruption handling](https://docs.livekit.io/agents/logic/turns/adaptive-interruption-handling/), which tells a real interruption from an "uh-huh" or background noise, so your agent doesn't stop talking when it shouldn't.
- [Background voice cancellation](https://docs.livekit.io/transport/media/noise-cancellation/).
- Session transcripts, traces, and recordings from LiveKit [Agent Observability](https://docs.livekit.io/testing/observability/).
- [Simulations](https://docs.livekit.io/testing/simulations/) that test full conversations with your agent, run in CI on every merge to `main`.
- A `Dockerfile` for [deploying to LiveKit Cloud](https://docs.livekit.io/deploy/agents/).

The starter works with any [custom web or mobile frontend](https://docs.livekit.io/frontends/) or with [telephony](https://docs.livekit.io/telephony/).

## Using coding agents

This project works with coding agents like [Claude Code](https://claude.com/product/claude-code), [Cursor](https://www.cursor.com/), and [Codex](https://openai.com/codex/).

LiveKit offers both a CLI and an [MCP server](https://docs.livekit.io/reference/developer-tools/docs-mcp/) for browsing and searching its documentation. Search returns short excerpts, so fetch the full page to read the details:

```console
lk docs search "testing my agent"
lk docs get-page /testing/unit-tests
```

The project also includes an [`AGENTS.md`](AGENTS.md) file and LiveKit's [agent skills](https://docs.livekit.io/intro/coding-agents/#agent-skills), so your coding agent follows LiveKit's best practices for workflows, handoffs, and testing, and tries its changes with the [agent debugger](https://docs.livekit.io/testing/debugger/). See the [coding agents guide](https://docs.livekit.io/intro/coding-agents/) for more details, including MCP server setup and how to update the skill.

## Dev setup

Install the [LiveKit CLI](https://docs.livekit.io/intro/basics/cli/), version 2.18.8 or later:

- **macOS:** `brew install livekit-cli`
- **Linux:** `curl -sSL https://get.livekit.io/cli | bash`
- **Windows:** `winget install LiveKit.LiveKitCLI`

Check your version with `lk --version`. To update an existing install, see [Update the CLI](https://docs.livekit.io/reference/developer-tools/livekit-cli/#updates).

Then create a project from this template. The CLI clones the template and configures your environment:

```console
lk cloud auth
lk agent init my-agent --template agent-starter-python
```

<details>
<summary>Set up the project manually</summary>

Clone the repository and install dependencies into a virtual environment with [uv](https://docs.astral.sh/uv/):

```console
git clone https://github.com/livekit-examples/agent-starter-python.git
cd agent-starter-python
uv sync
```

Sign up for [LiveKit Cloud](https://cloud.livekit.io/), then copy `.env.example` to `.env.local` and fill it in. To have the CLI write your project's URL and API keys into the file instead, run:

```console
lk cloud auth
lk app env --write --destination .env.local
```

</details>

## Painthaker on Ubuntu WSL

Painthaker runs from Ubuntu WSL (verified 2026-10-07 with uv 0.12.21, `lk` 2.18.8 and Python 3.11.16). On native Windows, the agent currently crashes at startup with `exit status 0xc0000005`; that hasn't been diagnosed.

Linux uses its own environment outside the repository, `~/.virtualenvs/painthaker`. The project's `my-agent/.venv` is for Windows; don't run plain `uv run` or `uv sync` from Linux in this folder. Without `UV_PROJECT_ENVIRONMENT`, uv would replace `.venv` with an empty Linux one.

### Start the chat

From any directory, in any new terminal:

```console
path/to/painthaker/my-agent/scripts/chat.sh
```

The script selects the Linux environment itself, so you don't need any exports. It checks that the environment matches `uv.lock` without installing anything, then starts `src/chat.py`. It stops with the exact setup command if the environment is missing or out of date. To use another environment, set `PAINTHAKER_VENV`.

| Key | Action |
|---|---|
| Ctrl+V or Ctrl+Shift+V (VS Code on Windows, including WSL terminals) | Paste into the input. Nothing is sent yet, and newlines and indentation are kept |
| Enter | Send the whole input as one message |
| Alt+Enter (or Ctrl+J) | New line while typing |
| Ctrl+C | Clear the current input, or interrupt a reply in progress (that turn isn't saved) |
| Ctrl+D | Quit (pending turns are saved first) |

### Conversation history

Each start opens a **new** conversation; an old one is never loaded automatically. The start-up line shows the most recent conversation and the command to resume it.

| Command | Action |
|---|---|
| `/new` | Start a new conversation |
| `/list` | Saved conversations: short ID, last update, number of exchanges, title |
| `/resume <id>` | Resume a conversation. Use the ID from `/list` (at least 4 characters). The last two exchanges are shown |
| `/delete <id>` | Delete a conversation permanently, after you type `oui` to confirm. Anything else cancels |
| `/help` | Show the commands and keys |

Commands are handled locally and never sent to the model. A single line that starts with one of these commands is always treated as that command. If its argument is malformed, as in `/resume ID`, `/resume the Baobab chat` or `/new please`, Painthaker shows how to use it and sends nothing. Multiline text, paths such as `/etc/passwd …`, and a line starting with an unknown `/word` followed by more text are sent as normal messages.

The model is told how this history works: chats are saved locally, a new chat starts empty, and older chats are loaded only with `/resume <id>`. When you ask about an earlier conversation that isn't loaded, it should point you to `/list` and `/resume` rather than deny that history exists or guess what was said.

How saving works:

- A conversation is saved after its first completed exchange. Each later exchange (your message, any tool calls with their results, and the reply) is saved as one unit.
- A reply that fails or that you interrupt with Ctrl+C is **not** saved. It is also removed from what the model sees, so you can simply resend the message.
- If saving fails, Painthaker says so in red. It retries on your next message and when you quit, and never claims a turn was saved when it wasn't. Each exchange gets an ID when it completes, and the database accepts that ID only once. A retry after a save that went through but was interrupted before confirmation therefore never stores the exchange twice. Databases from earlier versions are upgraded automatically on first start, keeping all saved messages.
- If unsaved exchanges remain, `/new` and `/resume` ask before switching, and you must type `oui` to abandon them. Anything else keeps you in the current conversation. `/delete` on the open conversation says how many unsaved exchanges would be lost too. Quitting with unsaved exchanges prints a red warning.
- The title comes from your first message, generated locally with no model call.
- A resumed conversation keeps its language (French or English). It uses the current instructions and today's date, so dates are never replayed from the past.
- Instructions, the date note and API keys are never stored.
- Tool calls are saved with their results and any provider metadata the agent framework puts in the call's `extra` field. The Gemini plugin, however, keeps its *thought signatures* in memory rather than in the chat history, so a resumed conversation sends earlier tool calls without them. Gemini accepted that in testing (2026-10-07, `gemini-3.5-flash-lite`). Google's current documentation doesn't say whether signatures on earlier turns are validated.

**Context limits:** the model sees the current instructions plus the most recent exchanges, within two limits:

- at most **20 exchanges**;
- at most **60 000 characters** of conversation. The budget counts **characters, not tokens**. It includes message text, tool arguments and tool results, but not the instructions or the language and date notes. At roughly 4 characters per token that's about 15 000 tokens, more for code or French. Set it with `PAINTHAKER_CONTEXT_CHARS`, a positive integer.

Older exchanges are dropped as whole exchanges, newest kept first, so a tool call is never separated from its result. The exchange being answered is always sent, even if a large tool result pushes it over the budget. A **single message longer than the budget is refused** with an explanation and isn't sent or saved; it is never silently cut. Split it into parts, or raise the budget. The full transcript stays in the database, and nothing is summarized automatically.

**Storage:** a SQLite file at `~/.local/share/painthaker/history.sqlite3` (or `$XDG_DATA_HOME/painthaker/…`). Set `PAINTHAKER_HISTORY_DB` to use another file. A new file is created readable only by you (`0600`) before anything is written to it, even inside an existing folder. The permissions of an existing database file or folder are never changed. The file is **not encrypted**. Anyone with access to your Linux account can read your chats, including any code or secrets you pasted. Database files are ignored by Git.

**Deletion:** `/delete` removes the conversation and all its messages from the database in one transaction. The deletion is permanent as far as Painthaker is concerned. Painthaker turns on SQLite's `secure_delete` setting, which [overwrites deleted content with zeros](https://www.sqlite.org/pragma.html#pragma_secure_delete) *inside the database file*. This is **not secure erasure**:

- During each write, SQLite copies the pages it changes into a temporary rollback journal (`history.sqlite3-journal`). It then deletes that file without overwriting it, so fragments can stay in free disk space.
- The filesystem, the WSL virtual disk, SSD wear-levelling, snapshots and backups (including Windows backups of the WSL disk) can keep older copies that SQLite can't reach.

To remove everything Painthaker stored, quit it and delete the database file. Even then, the same storage limitations apply. Don't paste secrets you can't afford to have on disk.

If the database can't be read (corrupt, not a Painthaker file, or written by a newer version), Painthaker stops with an explanation and leaves the file untouched. It never replaces or deletes it. Move the file aside, or point `PAINTHAKER_HISTORY_DB` elsewhere.

### First-time setup, or after `uv.lock` changes

```console
cd path/to/painthaker/my-agent
UV_PROJECT_ENVIRONMENT="$HOME/.virtualenvs/painthaker" uv sync --locked --python 3.11
```

`.env.local` holds `GOOGLE_API_KEY`, needed for every model reply, and `LIVEKIT_URL`, `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` for LiveKit CLI and Cloud features. It is ignored by Git; never commit it. The offline tests and CI need none of these.

Painthaker reads the current date and time from the machine's clock in your timezone. The default is `Africa/Porto-Novo`; to use another, set `PAINTHAKER_TIMEZONE` to an IANA name such as `Europe/Paris`.

### Advanced: tests, debugger, `lk` console

These commands need the Linux environment selected in the current shell first:

```console
cd path/to/painthaker/my-agent
export UV_PROJECT_ENVIRONMENT="$HOME/.virtualenvs/painthaker"
export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT" PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"

uv run --no-sync ruff format && uv run --no-sync ruff check
uv run --no-sync pytest                                          # offline only (live tests deselected)
uv run --no-sync pytest -m live tests/test_agent.py              # live: calls Gemini
uv run --no-sync pytest -m live tests/test_response_quality.py   # live: run separately
# The Gemini free tier allows 15 requests/min, so wait ~1 min between live runs.

lk agent debugger start                      # drive the agent turn by turn
lk agent debugger say "Bonjour !"
lk agent debugger stop
```

`lk agent console --text` also works, but it can't take pasted text: lk 2.18.8 drops pastes, and its input is one line capped at 1000 characters. Use `scripts/chat.sh` instead. Open issues are tracked in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Run the agent

The `lk agent console`, `lk agent dev`, and `lk agent debugger` commands run your agent on your own machine. To talk to it in your terminal:

```console
lk agent console
```

To connect it to LiveKit Cloud so a frontend, a phone call, or the [Agent Console](https://docs.livekit.io/testing/agent-console/) can reach it:

```console
lk agent dev
```

To let a coding agent or a script test it one text turn at a time, use the [agent debugger](https://docs.livekit.io/testing/debugger/). Each turn prints the agent's reply along with the tool calls and handoffs behind it:

```console
lk agent debugger start
lk agent debugger say "Hi, what can you do?"
lk agent debugger stop
```

In production, run the agent directly:

```console
uv run src/agent.py start
```

## Frontends and telephony

Pair the agent with a prebuilt frontend starter, or add telephony:

| Platform | Link | Description |
|----------|----------|-------------|
| **Web** | [`livekit-examples/agent-starter-react`](https://github.com/livekit-examples/agent-starter-react) | Web voice AI assistant with React & Next.js |
| **iOS/macOS** | [`livekit-examples/agent-starter-swift`](https://github.com/livekit-examples/agent-starter-swift) | Native iOS, macOS, and visionOS voice AI assistant |
| **Flutter** | [`livekit-examples/agent-starter-flutter`](https://github.com/livekit-examples/agent-starter-flutter) | Cross-platform voice AI assistant app |
| **React Native** | [`livekit-examples/voice-assistant-react-native`](https://github.com/livekit-examples/voice-assistant-react-native) | Native mobile app with React Native & Expo |
| **Android** | [`livekit-examples/agent-starter-android`](https://github.com/livekit-examples/agent-starter-android) | Native Android app with Kotlin & Jetpack Compose |
| **Web Embed** | [`livekit-examples/agent-starter-embed`](https://github.com/livekit-examples/agent-starter-embed) | Voice AI widget for any website |
| **Telephony** | [Documentation](https://docs.livekit.io/telephony/) | Add inbound or outbound calling to your agent |

For more options, see the [frontend guide](https://docs.livekit.io/frontends/).

## Testing and debugging

Simulations run full multi-turn conversations between a simulated user and your agent on LiveKit Cloud, then judge each transcript. The scenarios live in [`scenarios.yaml`](scenarios.yaml). Run them locally with the CLI:

```console
lk agent simulate text --scenarios scenarios.yaml
```

The `Simulations` workflow in [`.github/workflows/simulations.yml`](.github/workflows/simulations.yml) runs the same file on every merge to `main`, and on demand from the Actions tab. It doesn't run on every pull request push because each run uses real inference. See the [simulations guide](https://docs.livekit.io/testing/simulations/) for how to write scenarios and read results.

To check a change turn by turn without a live session, use the [agent debugger](https://docs.livekit.io/testing/debugger/) shown in [Run the agent](#run-the-agent).

To debug a running agent, open it in the [Agent Console](https://docs.livekit.io/testing/agent-console/). It shows events, tool calls, and model timing as you talk to the agent. To stream logs from a deployed agent, run `lk agent logs`.

## Using this template for your own project

After you create your own project from this template:

- **Commit `uv.lock`.** The template doesn't track it, but your project should, for reproducible builds. If you deploy to LiveKit Cloud, commit `livekit.toml` too.
- **Add repository secrets.** Add `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` as [repository secrets](https://docs.github.com/en/actions/how-tos/writing-workflows/choosing-what-your-workflow-does/using-secrets-in-github-actions) so the simulations can run in CI.

## Deploying to production

To deploy the agent to LiveKit Cloud or another environment with the included `Dockerfile`, see the [deployment guide](https://docs.livekit.io/deploy/agents/).

## Self-hosted LiveKit

You can self-host LiveKit instead of using LiveKit Cloud. See the [self-hosting guide](https://docs.livekit.io/transport/self-hosting/local/). If you self-host, use [model plugins](https://docs.livekit.io/agents/models/#plugins) instead of LiveKit Inference, and remove the [LiveKit Cloud noise cancellation](https://docs.livekit.io/transport/media/noise-cancellation/) plugin.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
