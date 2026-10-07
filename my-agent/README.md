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

## Painthaker: verified setup on Ubuntu WSL

Painthaker is developed and run from Ubuntu WSL. This was verified on 2026-10-07 with uv 0.12.21, `lk` 2.18.8 and Python 3.11.16. On native Windows, the agent currently crashes at startup with `exit status 0xc0000005`; that hasn't been diagnosed.

The project folder lives on the Windows drive, and `my-agent/.venv` is a **Windows** virtual environment (Python 3.12, `Scripts/`). Don't reuse or overwrite it from Linux. Linux uses a separate environment outside the repository, selected explicitly with `UV_PROJECT_ENVIRONMENT`:

```console
cd /mnt/c/Users/HP/Documents/Development/painthaker/my-agent
export UV_PROJECT_ENVIRONMENT="$HOME/.virtualenvs/painthaker"

# First time only, or after uv.lock changes:
uv sync --locked --python 3.11
```

To make `lk` start the agent with that environment, put it on `PATH` in the same shell:

```console
export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT" PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"

lk agent console --text                      # chat with Painthaker in the terminal (no pasting, see below)
lk agent debugger start                      # or drive it turn by turn
lk agent debugger say "Bonjour !"
lk agent debugger stop
```

These variables only last for the current shell. Run the exports in every new terminal.

### Pasting messages and code

`lk agent console` 2.18.8 drops pasted text, and its input is a single line capped at 1000 characters. To paste messages or multiline code, use the project's own text chat. It runs the same agent locally in text mode:

```console
uv run --no-sync python src/chat.py
```

| Key | Action |
|---|---|
| Ctrl+V or Ctrl+Shift+V (VS Code on Windows, including WSL terminals) | Paste into the input. Nothing is sent yet, and newlines and indentation are kept |
| Enter | Send the whole input as one message |
| Alt+Enter (or Ctrl+J) | New line while typing |
| Ctrl+C | Clear the current input |
| Ctrl+D | Quit |

Checks:

```console
uv run --no-sync ruff format && uv run --no-sync ruff check
uv run --no-sync pytest tests/test_language.py tests/test_response_checks.py tests/test_chat_input.py   # offline, no LLM calls
uv run --no-sync pytest tests/test_agent.py              # live: calls Gemini
uv run --no-sync pytest tests/test_response_quality.py   # live: run separately
# The Gemini free tier allows 15 requests/min, so wait ~1 min between live runs.
```

`.env.local` must define `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` and `GOOGLE_API_KEY`. Painthaker reads the current date and time from the machine's clock in your timezone. The default is `Africa/Porto-Novo`; to use another, set `PAINTHAKER_TIMEZONE` to an IANA name such as `Europe/Paris`. It is ignored by Git; never commit it. Open issues are tracked in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

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
