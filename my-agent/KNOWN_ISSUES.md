# Known issues

Observed but not fixed yet. For each behavior issue, add a scenario to `scenarios.yaml` that reproduces it before changing the prompt (see AGENTS.md).

## Agent behavior

Addressed on 2026-10-07 and covered by `tests/test_response_quality.py`, `tests/test_language.py` and `scenarios.yaml`: threats defined as external only, "Threat + Vulnerability = Risk" taught as a formula, raw LaTeX, overly long answers with generic praise, destructive attack examples, `cat`-based fixes, and treating `../` filtering as a file-access boundary. The reply language is now chosen in code (`llm_node` in `src/agent.py`) after replies drifted into French, including partway through an English one.

Also addressed: the agent had no date context and answered "2025" when asked the year. It now gets the date from the application clock on every LLM call (timezone: `PAINTHAKER_TIMEZONE`, default `Africa/Porto-Novo`). Ordinary questions are no longer steered back to cybersecurity.

Still to watch:

- **The date comes from the machine clock.** If the system clock is wrong, Painthaker's date is wrong too.

- **Destructive examples are prevented by the prompt only.** The cliché `; rm -rf /` appeared in 2 of 5 runs of the file-reading review until the prompt named it explicitly; then 0 of 8 code-review runs. A reply could still use a different destructive example.
- **Language detection is a word-list heuristic.** It handles French and English only. A message with too few common words (e.g. "ok") keeps the previous language. Mixed-language messages go to whichever language has more cue words.
- **"Concise" is checked with a 160-word limit in the tests**, on two simple questions only.
- **Answers about code it hasn't seen (addressed, not guaranteed).** In a brand-new conversation (2026-10-07), "Dans la fonction que tu viens de vérifier, quel argument de subprocess.run faut-il retirer ?" got a confident "Il faut retirer l'argument shell=True…". Now:
  - a prompt rule says to answer only from code in the conversation and to ask for it again otherwise;
  - code adds a context note when it knows something is missing: a failed or interrupted message, or exchanges left out by trimming or resuming.

  The same question now gets a request to paste the code. Covered by offline tests on the notes and live tests for the failed, resumed, trimmed and fresh cases. Remaining gaps:
  - Code can't detect every missing reference. In a fresh process the failure flag from an earlier run is gone, and only the prompt rule applies there.
  - The model could still answer from a vague reference ("ce code") when something *similar* is in context.
  - Each live case was checked once.

- **Absolute claims and the saved history (addressed, not guaranteed).** In real use (2026-10-07), the agent said MAC addresses are permanent and "never change", and in a new chat it said Painthaker had no persistent history.
  - **MAC addresses:** Android 10+ randomizes the Wi-Fi MAC address by default (source.android.com), and Apple devices use a private, optionally rotating, address per network. The prompt now has a general rule against absolute words where exceptions matter; it doesn't list individual facts.
  - **History:** the terminal chat gives the agent a note on how saved history works (local saves, new chats start empty, `/list` and `/resume <id>`).
  - **Checks:** each case passed one live check, with different wording from the original question. Other absolute claims can still slip through.

## Conversation history

- **Gemini thought signatures aren't persisted.** The Gemini plugin keeps them in memory on the LLM object (`_thought_signatures`), not in the chat items, so they're lost on restart, on `/resume`, and when a failed turn rebuilds the session. Gemini accepted resumed tool history without them in a live test (2026-10-07). Google's docs don't say whether signatures on earlier turns are validated, so this could change.
- **Provider errors print a traceback.** When Gemini fails (for example `504 DEADLINE_EXCEEDED`, seen 2026-10-07 after 4 attempts), the SDK logs a traceback to the terminal before Painthaker's own "not saved" message. The turn is correctly discarded, but the output is noisy.
- **One chat window per conversation.** Two windows writing to the same conversation keep the database consistent, but each window only sees its own exchanges.

## Runtime

- **`agent_name` deprecation warning at startup:** "agent_name is set in code; move it to livekit.toml ([agent] name). The agent_name parameter will be removed in a future release." It comes from `@server.rtc_session(agent_name="my-agent")` in `src/agent.py`.
- **Event-loop stall warnings at startup.** For example, "event loop blocked for 170ms importing unittest.mock" and "314ms importing anyio._backends._asyncio". The warning suggests importing these modules during process prewarm.
- **`lk agent console --text` can't take pastes.** In lk 2.18.8, the console's `Update` handles key events only. Bracketed pastes (`tea.PasteMsg`) are dropped, and the single-line input would turn newlines into spaces and cut text at 1000 characters anyway. Work around it with `src/chat.py` (see README). Reported to LiveKit through `lk docs submit-feedback`.
- **Native Windows crash.** `lk agent console --text` exits with `0xc0000005` ("Agent exited with no output"). Not diagnosed; use Ubuntu WSL (see README).

## Testing

- **Simulations haven't been run yet.** `lk agent simulate text --scenarios scenarios.yaml` uses LiveKit Cloud simulation usage. With Gemini's free tier (15 requests/min for `gemini-3.5-flash-lite`), parallel scenarios are likely to fail with 429 errors.
- **CI runs offline checks only.** The root workflow `.github/workflows/ci.yml` runs formatting, lint and the offline tests with no secrets. The starter's workflows in `my-agent/.github/workflows/` (ruff, simulations, template checks, version tagging) are intentionally inactive: GitHub only runs workflows from the repository root. Live tests and simulations stay manual because they need API keys and use quota.
