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
  - **Regression in English (2026-10-07, after `/prefs`).** A manual check with `/prefs language en` got "MAC addresses never change" again. Asked directly ("Is my MAC address a permanent ID?"), the agent was right 3 of 3 times. A plain definition ("What is a MAC address?") slipped the claim in as an aside or analogy ("a permanent fingerprint, unlike IP addresses") 3 of 3 times with the default beginner/concise note, and 2 of 3 times without it. So the rule only applied when permanence was the question, and the new note made it worse. The prompt now says the rule covers side remarks and analogies, the beginner note says analogies must stay true, and the concise note keeps "the main exception to a general statement". After the change, 1 of 9 English definition runs still said "permanent digital fingerprint" (that reply did mention randomization later), and 1 hedged with "usually" without naming the exception. Covered by `test_definition_does_not_slip_in_an_absolute_claim_in_english` (live), which can therefore fail occasionally.

## Conversation history

- **Gemini thought signatures aren't persisted.** The Gemini plugin keeps them in memory on the LLM object (`_thought_signatures`), not in the chat items, so they're lost on restart, on `/resume`, and when a failed turn rebuilds the session. Gemini accepted resumed tool history without them in a live test (2026-10-07). Google's docs don't say whether signatures on earlier turns are validated, so this could change.
- **Provider errors print a traceback.** When Gemini fails (for example `504 DEADLINE_EXCEEDED`, seen 2026-10-07 after 4 attempts), the SDK logs a traceback to the terminal before Painthaker's own "not saved" message. The turn is correctly discarded, but the output is noisy.
- **One chat window per conversation.** Two windows writing to the same conversation keep the database consistent, but each window only sees its own exchanges.

## Runtime

- **`agent_name` deprecation warning at startup:** "agent_name is set in code; move it to livekit.toml ([agent] name). The agent_name parameter will be removed in a future release." It comes from `@server.rtc_session(agent_name="my-agent")` in `src/agent.py`.
- **Event-loop stall warnings at startup.** For example, "event loop blocked for 170ms importing unittest.mock" and "314ms importing anyio._backends._asyncio". The warning suggests importing these modules during process prewarm.
- **`lk agent console --text` can't take pastes.** In lk 2.18.8, the console's `Update` handles key events only. Bracketed pastes (`tea.PasteMsg`) are dropped, and the single-line input would turn newlines into spaces and cut text at 1000 characters anyway. Work around it with `src/chat.py` (see README). Reported to LiveKit through `lk docs submit-feedback`.
- **Native Windows crash.** `lk agent console --text` exits with `0xc0000005` ("Agent exited with no output"). Not diagnosed; use Ubuntu WSL (see README).

## Preferences

- **Level and length depend on the prompt.** The language preference is applied in code. `level` and `length` are a per-call note to the model, so how closely replies follow them, and whether a one-off "explique en détail" is honored, depends on the model. Offline tests verify the note reaches the model, not the reply's style. No live Gemini check has been run for preferences yet.
- **Language overrides recognize explicit requests only.** With `language fr`, an English message gets a French reply unless it says something like "in English" or "en anglais". Other phrasings aren't detected as requests.
- **Applies everywhere.** Preferences are global, not per conversation.

## Voice

Prepared on 2026-10-08 and covered by offline tests (`tests/test_voice.py`). A debugger text check the same day confirmed that text replies still use Markdown.

**First browser voice test (2026-10-08): manual observations reported by the user, not automated or measured results.** Agent Console, `lk agent dev` in WSL, preemptive generation off.

- **Interruptions:** the agent stops and follows the new request.
- **Follow-up questions:** context is kept.
- **Responsiveness:** no noticeable delay.
- **Recognition:** generally understood.
- **English speech:** sounds good.
- **French speech: clear, but with a very unnatural accent.** Likely cause, from provider documentation and not yet tested: the voice is Fish Audio "Sawyer" (`fa4c9eb3…`), an American English voice. Fish voices are cross-lingual and keep their native accent in every language. No `language` is set on the TTS.
- **Occasional breaks: undiagnosed.** It isn't known yet whether they come from premature end-of-turn detection or from gaps in audio playback. Interruption and endpointing settings stay unchanged until that is established, for example from Agent Console's Events and Audio panes.

**Bilingual voice routing (implemented 2026-10-19, not yet tested by voice).** `Painthaker.llm_node` now sets the TTS voice and `language` from the same `conversation_language()` result that picks the reply language, on every spoken LLM call. Text-only output leaves the TTS untouched. The session starts with Lucas and `fr`.

Implemented:

- French → Lucas (`f69bca09…`), `language="fr"`
- English → Sawyer (`fa4c9eb3…`), `language="en"`

Covered by offline tests only (`tests/test_voice.py`, no TTS calls). These check the mapping, the switches and that text output doesn't change the voice, not how it sounds. The change reaches the reply being generated because the framework (livekit-agents 1.8.3) opens that reply's TTS stream after `llm_node` starts, and the stream copies the options when it opens. That ordering comes from the framework's code, not its documentation.

Still to verify manually:

- actual voice switching in a live browser session
- French quality with Lucas in full conversations
- English quality with Sawyer after switching mid-conversation
- switching French → English → French in the same session
- occasional audio breaks

Still not checked by voice: whether Gemini follows every spoken rule (no Markdown, no code read aloud, no offers to send code elsewhere), and STT accuracy on technical terms.

- **Spoken rules depend on the prompt.** Code picks the rule set; it doesn't check the reply. A reply could still contain Markdown or a code block, which the TTS would read out.
- **Typed messages in a voice session get spoken rules.** The choice follows the output (audio on means the reply is spoken), not how the user wrote the message. Exact code is best asked for in the terminal chat.
- **The session note mentions `/list` and `/resume <id>`.** Said aloud, these may sound odd.
- **Preferences are read once, at session start.** A `/prefs` change in the terminal chat applies to the next voice session.
- **The browser test needs LiveKit Cloud.** STT and TTS run through LiveKit Inference and use the project's monthly credits, a hard cap on the free plan. The turn detector and adaptive interruptions have free local-development allowances (7,500 and 40,000 requests a month), after which the turn detector falls back to a smaller local model.

## Notes search

- **Keyword matching only.** A question worded differently from the notes ("impression" vs "imprime") finds only what the shared words reach; there are no synonyms or embeddings. A live check (2026-10-07) first exposed a fixed two-word threshold that dropped the only matching passage. The threshold is now relative, with a regression test.
- **Prompt-injection resistance relies on the prompt.** Excerpts are labelled as data, and the instructions say never to follow requests inside notes, but that isn't a guarantee.
- **Excerpts leave the machine and persist.** They go to the model provider, and they're saved in the conversation history file as part of the tool result.
- **Checked live once per case:** a supported question with a citation, and an unsupported one.

## Workspace files

- **First version: bounded and non-recursive.** `list_files` shows one folder at a time (200 entries) and `read_file` returns 300 lines or 12 000 characters per call, so exploring a large project takes many calls. Each call's result also counts toward the conversation's character budget. There is no edit tool.
- **`search_code` is plain substring matching.** No regular expressions, word boundaries or multi-line matches. Case-insensitive matching uses `str.lower()`, so a few Unicode characters whose lowercase form has a different length (such as `İ`) can put the excerpt slightly off the match. There's no per-file match limit, so one file with many matches can use up the 100-match budget. The search also stops at 2 000 files or 16 MB, after which the result is marked incomplete. Files are searched in walk order, not by relevance.
- **`git_status` trusts the `git` found on PATH**, as long as it isn't inside the workspace. It reads the user's global Git configuration. Only `core.fsmonitor` is overridden; other settings stay in effect, and none of them makes `status` write or contact a network. Submodules are reported as git reports them. A repository owned by another user is refused by git's `safe.directory` check, and Painthaker doesn't change that setting.
- **`git_status` reads outside the workspace when the workspace is a subfolder of a repository.** Git reads the enclosing repository's `.git` folder to answer, but only changes inside the workspace are listed.
- **The block list is name-based.** A secret stored under an ordinary name (`config.yaml`, `settings.py`) is readable. `.env.example` is blocked along with every other `.env.*` file. The ignored folders (`dist`, `build`, …) are refused at any depth, including a source package that happens to be called `build`.
- **Not checked on native Windows.** The tests ran in WSL (Linux). On native Windows, the checks after opening a file fall back to `realpath`, since there's no `/proc/self/fd`, and `O_NOFOLLOW` doesn't exist there. 8.3 short names (`SECRET~1.JSO`) aren't recognized as protected. The agent doesn't start on native Windows anyway (see [Runtime](#runtime)).
- **Paths that differ only in case.** On a case-insensitive mount such as `/mnt/c`, a symbolic link whose target spells the workspace path in different case is treated as leading outside and refused. This errs on the safe side.
- **Prompt-injection resistance relies on the prompt.** File contents are labelled as data, and the instructions say never to follow requests inside them, but that isn't a guarantee.
- **Not tried with the live model yet.** `list_files` and `read_file` are covered by offline tests and by a scripted session. `search_code` and `git_status` are covered by offline tests that call the tools directly, against real temporary repositories for `git_status`. No Gemini session or simulation scenario has exercised any of them.

## Verification tools

- **`run_tests` runs the project's code.** Its tests and `conftest.py` files can modify or delete files, make network requests, start processes, or print secrets they read from files. Painthaker removes secret-looking environment variables and redacts their values if they appear in the output, and hides the workspace, Python-environment and home paths. A secret stored in a file or under an ordinary variable name, and other host paths, are passed on as printed. The wrapper writes no pytest cache or bytecode, but it can't stop the tests from writing.
- **Painthaker's own Python environment runs the tests.** A project whose dependencies aren't installed there fails with import errors, and nothing is installed. A workspace whose project has its own virtualenv gets no special handling. If Painthaker itself runs from a virtualenv inside the workspace, both tools refuse to start.
- **The project's own configuration applies.** pytest reads the project's `pyproject.toml`, `pytest.ini` or `conftest.py` (`addopts`, plugins, markers), possibly from a parent folder of the workspace. Installed pytest plugins load as usual. Ruff reads the project's configuration, except that fixing is always off.
- **Bounded, not sandboxed.** Runs are stopped after 120 seconds (tests) or 30 seconds (Ruff), and the whole process group is killed. A test process that leaves that group (for example by calling `setsid`) can survive the timeout. Output is captured in a temporary file outside the workspace, which a very noisy run can make large until the timeout. There's no CPU, memory or network limit.
- **Paths are checked once.** The target is resolved and checked before pytest starts; a folder swapped for a link after that check isn't re-checked.
- **Not tried with the live model yet.** Both tools are covered by offline tests that run the real pytest and Ruff on temporary projects. No Gemini session has called them.
- **POSIX only.** The process-group kill and the tests that use links and shell scripts assume Linux or macOS; the agent doesn't start on native Windows anyway.

## Testing

- **Simulations haven't been run yet.** `lk agent simulate text --scenarios scenarios.yaml` uses LiveKit Cloud simulation usage. With Gemini's free tier (15 requests/min for `gemini-3.5-flash-lite`), parallel scenarios are likely to fail with 429 errors.
- **CI runs offline checks only.** The root workflow `.github/workflows/ci.yml` runs formatting, lint and the offline tests with no secrets. The starter's workflows in `my-agent/.github/workflows/` (ruff, simulations, template checks, version tagging) are intentionally inactive: GitHub only runs workflows from the repository root. Live tests and simulations stay manual because they need API keys and use quota.
