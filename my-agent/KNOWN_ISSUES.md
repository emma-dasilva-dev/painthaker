# Known issues

Observed but not fixed yet. For each behavior issue, add a scenario to `scenarios.yaml` that reproduces it before changing the prompt (see AGENTS.md).

## Agent behavior (seen in `lk agent console --text`, 2026-10-07)

- **Threats are defined as external only.** The agent described threats as coming only from outside. Threats can also be internal: insiders, mistakes, misconfiguration.
- **"Threat + Vulnerability = Risk" presented as a formula.** This is a simplification. Risk is usually explained as a combination of likelihood and impact, given a threat and a vulnerability. It shouldn't be taught as literal arithmetic.
- **French word in an English reply.** An English response contained "risque". The "never mix languages" rule isn't always followed.
- **Too verbose for simple questions.** Answers to a simple conceptual question were much longer than needed. The "concise by default" rule isn't working well enough.

## Runtime

- **`agent_name` deprecation warning at startup:** "agent_name is set in code; move it to livekit.toml ([agent] name). The agent_name parameter will be removed in a future release." It comes from `@server.rtc_session(agent_name="my-agent")` in `src/agent.py`.
- **Event-loop stall warnings at startup.** For example, "event loop blocked for 170ms importing unittest.mock" and "314ms importing anyio._backends._asyncio". The warning suggests importing these modules during process prewarm.
- **`lk agent console --text` can't take pastes.** In lk 2.18.8, the console's `Update` handles key events only. Bracketed pastes (`tea.PasteMsg`) are dropped, and the single-line input would turn newlines into spaces and cut text at 1000 characters anyway. Work around it with `src/chat.py` (see README). Reported to LiveKit through `lk docs submit-feedback`.
- **Native Windows crash.** `lk agent console --text` exits with `0xc0000005` ("Agent exited with no output"). Not diagnosed; use Ubuntu WSL (see README).

## Testing

- **Simulations haven't been run yet.** `lk agent simulate text --scenarios scenarios.yaml` uses LiveKit Cloud simulation usage. With Gemini's free tier (15 requests/min for `gemini-3.5-flash-lite`), parallel scenarios are likely to fail with 429 errors.
- **The CI workflows don't run yet.** They live in `my-agent/.github/workflows/`, but GitHub only runs workflows from the repository root's `.github/workflows/`. The simulations workflow would also need a `GOOGLE_API_KEY` repository secret.
