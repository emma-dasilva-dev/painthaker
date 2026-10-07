# Known issues

Observed but not fixed yet. For each behavior issue, add a scenario to `scenarios.yaml` that reproduces it before changing the prompt (see AGENTS.md).

## Agent behavior

Addressed on 2026-10-07 and covered by `tests/test_response_quality.py`, `tests/test_language.py` and `scenarios.yaml`: threats defined as external only, "Threat + Vulnerability = Risk" taught as a formula, raw LaTeX, overly long answers with generic praise, destructive attack examples, `cat`-based fixes, and treating `../` filtering as a file-access boundary. The reply language is now chosen in code (`llm_node` in `src/agent.py`) after replies drifted into French, including partway through an English one.

Still to watch:

- **Destructive examples are prevented by the prompt only.** The cliché `; rm -rf /` appeared in 2 of 5 runs of the file-reading review until the prompt named it explicitly; then 0 of 8 code-review runs. A reply could still use a different destructive example.
- **Language detection is a word-list heuristic.** It handles French and English only. A message with too few common words (e.g. "ok") keeps the previous language. Mixed-language messages go to whichever language has more cue words.
- **"Concise" is checked with a 160-word limit in the tests**, on two simple questions only.

## Runtime

- **`agent_name` deprecation warning at startup:** "agent_name is set in code; move it to livekit.toml ([agent] name). The agent_name parameter will be removed in a future release." It comes from `@server.rtc_session(agent_name="my-agent")` in `src/agent.py`.
- **Event-loop stall warnings at startup.** For example, "event loop blocked for 170ms importing unittest.mock" and "314ms importing anyio._backends._asyncio". The warning suggests importing these modules during process prewarm.
- **`lk agent console --text` can't take pastes.** In lk 2.18.8, the console's `Update` handles key events only. Bracketed pastes (`tea.PasteMsg`) are dropped, and the single-line input would turn newlines into spaces and cut text at 1000 characters anyway. Work around it with `src/chat.py` (see README). Reported to LiveKit through `lk docs submit-feedback`.
- **Native Windows crash.** `lk agent console --text` exits with `0xc0000005` ("Agent exited with no output"). Not diagnosed; use Ubuntu WSL (see README).

## Testing

- **Simulations haven't been run yet.** `lk agent simulate text --scenarios scenarios.yaml` uses LiveKit Cloud simulation usage. With Gemini's free tier (15 requests/min for `gemini-3.5-flash-lite`), parallel scenarios are likely to fail with 429 errors.
- **The CI workflows don't run yet.** They live in `my-agent/.github/workflows/`, but GitHub only runs workflows from the repository root's `.github/workflows/`. The simulations workflow would also need a `GOOGLE_API_KEY` repository secret.
