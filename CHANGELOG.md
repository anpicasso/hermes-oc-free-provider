# Changelog

Notable changes to `oc-free-provider`.

## Unreleased

- **Upstream HTTP status reaches Hermes.** `CompletionRefused.status_code` now
  carries the 4xx/5xx the official runtime reports (companion `completion`
  outcome, OpenCode `step_failed.error.status`, `response_unverified` outcome),
  so a 429 free-usage limit is classified as a rate limit and 5xx as a server
  error. Refusals without a reported status carry `None`; nothing is inferred
  from message text. Adapted from douglasanpa `8a4fc07`.
- **Synchronous create runs inside the call.** `chat.completions.create()`
  performs the request before returning when no event loop is running, so
  Hermes' interrupt and stale-call watchdog covers it. Previously the request
  ran on first attribute access, after the watched worker had returned, and an
  interrupt waited for the whole response. Awaiting from an event loop is
  unchanged (off-loop thread). Adapted from pasei `9841760`.
- **Profile copies stay isolated.** Package-relative sibling imports when Hermes
  loads a profile's copy, so two profiles in one process no longer share bare
  `oc_*`/`hermes_oc_client` modules. Adapted from pasei `a05dd17`.
- **Catalog context window.** `get_model_context_length` returns the native
  catalog `limit.context` from this profile's cached discovery rows, without
  starting OpenCode; a cold cache defers to Hermes' fallback. Adapted from pasei
  `e31b61d` (context only; `max_tokens` remains unsupported).
- **Runtime environment.** The runtime child keeps the same platform allowlist
  as the version probe, including Windows loader variables. Native Windows
  remains unverified. Related to douglasanpa `e9cfe32`.

## 4.2.1 — 2026-11-16

- **Companion isolation hardening.** Each runtime admits only one live session.
  Concurrent `open` requests are serialized; capacity, duplicate/closed ids,
  and native-session existence are checked before tool-registry mutation.
  Rejected admission cannot replace the live session or reset its generation.
  Close→open and retained finish→advance remain supported.
- **Pinned profile identity.** The per-request context records the originating
  profile home. Lazy completion refuses missing or mismatched profile identity
  before accessing a runtime or affinity database, rather than rebinding silently.
- **Verification.** All 126 tests pass with the official OpenCode 2.3.0 binary
  in an isolated home; 12 admission/profile regressions were independently
  rerun successfully. Plugin validation and doctor pass with the existing
  private-core-import warning. This patch was tested with local fixtures, not
  a new hosted CLI acceptance run; the real CLI continuity checks above the
  transport were completed for 4.2.0.
- **Scope.** These fixes address reproduced RPC admission/tool-schema cross-talk
  and harden lazy profile binding. They do not establish absence of all leaks;
  the unconfirmed late lifecycle-event race remains outside this patch.
  Native Windows/macOS execution remains unverified.

## 4.2.0 — 2026-11-16

- **Session/cache affinity.** Scoped conversations retain one exact
  official-generated native OpenCode session and a durable private workspace
  per active profile/scope/model lane. The mapping lives in
  `<profile>/plugin-data/oc-free-provider/data.db` through the documented
  `plugins.plugin_storage.plugin_db` helper; Hermes' main session database is
  untouched. A missing scope or profile stays ephemeral and never shares an
  anonymous bucket.
- **Lane identity uses both raw ids.** Normal/compression/`/branch`/gateway
  lanes use `cache_scope_id`; `sub_<sha>` delegate scopes add the physical
  `session_id` so siblings never merge (delegate compression resets affinity).
  `/btw` and same-model review share the parent's lane and serialize; a `sub_`
  scope without a physical id stays ephemeral.
- **Companion protocol 5.** `open` can adopt an exact existing native id;
  `advance` accepts generation N+1 after the new explicit `finish` boundary,
  carries the current Hermes tool registry, and returns the event cursor
  boundary. `finish` rejects parked executor results, interrupts/wait-settles a
  live prompt and keeps the native session. Late prompt/wait rejection
  callbacks are generation-guarded so they cannot mutate the next generation.
- **Recovery and isolation.** Native `GET /api/session/<id>` corroborates the
  stored id and workspace before reuse. Only a real 404 resets affinity; fresh
  allocation replaces the mapping and sets `Completion.affinity_reset`.
  Per-scope cross-process ownership is held for the runtime lifetime (including
  idle) and released only after the server stops; same-scope in-process calls
  get a bounded busy refusal. Unsafe-finish recovery allocates a fresh session
  with exactly one runtime lease.
- **Bounded retention.** Idle process eviction keeps durable native data and
  the mapping. `RuntimePool.MAX_OWNERS` bounds live lanes and
  `oc_affinity.prune()` removes inactive owned scope directories plus their rows
  under the ownership lock (default 14 days), re-reading eligibility under that
  lock. Mapping rows contain no transcript or credentials; native data remains
  sensitive.
- **Verification.** The full offline suite (121 tests) passes with the official
  2.3.0 binary and an isolated `HERMES_HOME`; a bounded hosted two-turn probe
  confirmed one native id and cache-read usage, while the ephemeral control also
  showed provider-side cache hits, so no causal speedup is claimed.
- **Real CLI acceptance.** An isolated Hermes profile retained the exact native
  session id across gateway shutdown and resumed conversation. Two real
  delegated siblings received distinct native ids. All 11 acceptance assertions
  passed; the production gateway was not restarted.

## 4.1.0 — 2026-11-16

Native OpenCode provider, read-only catalog policy.

- **Runtime replaced.** The direct-Bun transport (`oc_transport.js`) and the
  Bun requirement are removed. The provider starts a private, unmodified
  official `opencode serve` process (>= 2.3.0) and loads a bundled companion
  through public plugin hooks. There is no fork, proxy service, core patch or
  provider-side tool execution. The provider does not claim vendor approval.
- **Catalog.** The model-facing native set is exactly `read`, translated to
  Hermes `read_file`. Every other native descriptor (write, edit, patch, shell,
  glob/grep, skill, webfetch/websearch, question, subagent, todowrite,
  `execute`, browser/opencode/console families) is removed by identity before
  the request. All genuine Hermes schemas in the session are advertised with
  their full descriptions and parameters; a Hermes name colliding with a
  native registry id gets the reversible wire alias `hermes_<name>`.
- **One mapper.** `oc_translate` implements only the bounded `read` adapter
  (`offset` 0/1 = line 1, positive limit capped, negative offset refused). The
  former write/edit/shell/glob/grep/skill/webfetch/websearch/question
  translators and their inverses are gone.
- **`read_file` required.** Planning refuses before inference when Hermes does
  not expose it, instead of advertising a compatibility read that cannot run.
- **Binary onboarding.** The provider resolves a validated official `opencode`
  >= 2.3.0 from the Hermes process PATH, then the active profile's
  `plugins.entries.oc-free-provider.settings.binary_path`. `auth_handler` for
  `hermes auth add opencode-free` and `model_setup_flow` for `hermes model`
  share one interactive helper that validates before saving and leaves the
  configuration untouched on cancel/invalid input. No environment-variable
  binary setting. Core's resolved command is honored as a validated hint, and
  same-process onboarding refreshes the profile's launch command.
- **Runtime portability.** The exact-artifact hash allowlist and blanket
  Linux-only refusal are replaced by a version >= 2.3.0 gate plus the companion
  handshake. Launch stays an argv list (`shell=False`); process-group/session
  startup, ownership checks and teardown are portable, with `/proc` used only
  when present. Linux startup/cancellation/cleanup remain the verified target;
  other platforms are not claimed.
- **Companion protocol 4.** The completion gate requires terminal `[DONE]`, a
  supported consistent `finish_reason` and valid `prompt_tokens` /
  `completion_tokens` (zero valid). Missing usage, malformed usage, conflicting
  or unsupported finishes, errors and truncation refuse before any tool is
  handed to Hermes. Optional cache/reasoning breakdowns are preserved.
- **Native events** own text, reasoning and tool names/arguments/call IDs.
  Opaque per-part metadata is preserved through Hermes history in one
  `reasoning_details` carrier (`opencode-free.native_assistant`), position-keyed
  so sparse metadata does not shift on replay. Step-level `cost`/`rawFinish`
  are not replayed.
- **Unsolicited native calls.** Exposed `read` and registered Hermes tools
  dispatch; native `execute` and every other unadvertised native tool are
  refused with their own event and no side effect. `decide` still cannot
  authorize JavaScript.
- **Request controls.** `stream=True`, `temperature`, `top_p`, token limits,
  `stop`, `response_format` and `reasoning_effort` are refused at create time
  (including through the lazy facade); iterating a non-stream completion raises
  the same clear refusal. True streaming and generation-control parity remain
  deferred in issue #11; unsupported non-default values are never silently
  ignored.
- **Auxiliary.** `resolve_aux_model` returns `""` because auxiliary requests
  send controls this transport cannot map yet. The provider advertises no cheap
  model instead of pretending those controls are honored.
- **Disclosure.** The unmodified binary composes its own request preamble
  (working directory, git status, OS details) and follows its own outbound HTTP
  defaults. The provider runs it in an isolated workspace with a small
  environment allowlist and a fresh per-process `OPENCODE_PASSWORD`, and refuses
  redirects on its own loopback RPC; it does not intercept upstream traffic.
- **Verification.** Offline suite against the installed official OpenCode 2.3.0
  binary with an isolated `HERMES_HOME`; bounded hosted text plus read/tool-result
  continuation with the same binary. Fixture output is not hosted evidence.
- **Relay compatibility retained.** Local traceparent metadata is accepted
  without forwarding headers; direct and lazy completions serialize for
  middleware.

## 4.0.0 — 2026-11-13

- Follow OpenCode's then-current free-tier gate with a Bun transport
  (`oc_transport.js`), the session header, `<env>` preamble and `stream: true`
  tool-call shape. Model discovery used Python's standard-library HTTPS client.
- Use the Console endpoints `https://opencode.ai/inference/openai/v1` for
  inference and `/inference/v1/models` for discovery.
- Keep the free-model catalog live: refresh the relay list at most every 10
  minutes, models.dev metadata every 6 hours, persist the last successful
  discovery for restart fallback, and pin the provider picker floor to the
  discovered catalog.
- Replay `delegate_task` history completely, including multi-task batches and
  follow-up task actions.
- Replay unmappable Hermes `patch` history (for example V4A multi-file patch
  mode) under its native name instead of poisoning the conversation.
- Tolerate `null` `tool_calls` deltas in Chat Completions streams.
- Require Hermes >= 0.23.1.

## 3.0.0 — 2026-09-21

- Rename the Hermes plugin identity to `oc-free-provider` while keeping the
  repository URL stable.
- Remove the OpenCode CLI, Node.js, npm and `npx` dependency.
- Discover live free models over HTTPS by intersecting the OpenCode Zen
  `/v1/models` response with models.dev.
- Translate compatible OpenCode and Hermes tool schemas, calls, forced choices
  and replayed history while Hermes executes tools locally.
- Expand the compatibility and terms notice around third-party access, pricing,
  retention and acceptable use.

## 0.2.1 — 2026-09-20

- Initial public release line. See the Git history and tagged release for the
  original notes.
