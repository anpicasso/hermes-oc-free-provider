# Companion RPC contract

Status: contract for `companion/index.ts` and the sibling runtime in 4.1.0.
Companion id `hermes-oc-native`, protocol `4`. The official OpenCode binary is unchanged.
The companion is loaded into the official `opencode serve` process and uses
only public plugin hooks and RPC; it never starts a process of its own. A
mismatched protocol, companion id or binary version is refused before any model
request.

Supported binary: official OpenCode >= 2.3.0 (resolved from the Hermes process
PATH or the profile's `binary_path` setting; no env override).

Configure the plugin `package` as the **companion directory** (`.../companion`),
not `index.ts`: OpenCode v2.3.0 rejects a file plugin path. Production passes
no plugin options. Fixture registration lives only in
`tests/fixtures/companion/` and is never part of the packaged companion.

## Transport and authentication

The companion registers authenticated loopback JSON RPC under id
`hermes-oc-native`:

```text
POST /api/rpc/hermes-oc-native/<method>?location[directory]=<workspace>
Content-Type: application/json
Authorization: Basic base64("opencode:<OPENCODE_PASSWORD>")

{"input": {...}}
```

- The runtime generates a fresh random `OPENCODE_PASSWORD` per OpenCode
  process and passes it only in the child environment. The server listens on
  `127.0.0.1` with an ephemeral port.
- Every method is request/response JSON. Redirects are refused by the client.
  `hello` and `catalog` need no session; `open` creates one; `advance`, `read`,
  `results`/`settle`, `decide`, `state` and `close` require an existing session.
- Methods: `hello`, `catalog`, `open`, `advance`, `read`, `results`, `decide`,
  `state`, `close`. `settle` is accepted as a one-revision alias of `results`;
  new code uses `results`. Nothing else is an alias.

Startup readiness is the runtime's stdio contract: `opencode serve` prints one
JSON line containing its loopback URL, the runtime validates that URL, then
calls `hello`. The companion does not expose a separate readiness endpoint.

## `hello`

```json
{}
```

Returns:

```json
{
  "protocol": 4,
  "companion": "hermes-oc-native",
  "binaryVersion": "2.3.0",
  "location": {"directory": "<cwd>", "project": "<canonical project>"},
  "sessions": 0,
  "closedSessions": 0,
  "plugins": ["<loaded plugin id>"],
  "tools": [{"id": "read", "name": "read", "definition": "<sha256>", "forwarded": true, "hermes": false}]
}
```

`definition` is a structural SHA256 of `name`, `description`, `input` and
`output`; `outputDefinition` hashes the output schema alone. Every registered
tool must report `forwarded: true`; the companion replaces native executors and
never lets one run. `hermes: true` marks a tool registered from a Hermes schema
(Python reserves native ids and supplies reversible aliases); `hermes: false`
marks a native OpenCode tool. `hello` does not authorize inference or create a
session. It may be called repeatedly and re-reconciles the tool registry each
time.

## `catalog`

```json
{}
```

Returns non-secret native discovery metadata:

```json
{
  "models": [{
    "id": "big-pickle",
    "providerID": "opencode",
    "name": "Big Pickle",
    "package": "@opencode/ai/providers/openai-compatible",
    "providerPackage": "@opencode/ai/providers/openai-compatible",
    "capabilities": {"tools": true, "input": ["text"], "output": ["text"]},
    "cost": [{"input": 0, "output": 0, "cache": {"read": 0, "write": 0}}],
    "limit": {"context": 200000, "output": 32000},
    "status": "active",
    "enabled": true
  }],
  "providers": [{"id": "opencode", "name": "OpenCode", "package": "@opencode/ai/providers/openai-compatible"}]
}
```

- Source is the live public API: `ctx.model.list()` plus `ctx.provider.list()`.
  The companion does not read files, environment, auth storage or connection
  records.
- Keep every returned model, including unsupported transports. Preserve
  `cost` exactly, including all tiers; never infer, zero or normalize a price.
  Keep `limit`, `capabilities`, `status`, `type`, `family`, `compatibility` and
  `time` as reported. `variants` is reduced to each variant's non-secret `id`.
- `providerPackage` is the owning provider's package; `package` is the
  model-level override when present.
- Never serialize `settings`, `headers`, `body`, API keys, base URLs,
  `connection`, `auth` or any credential material.
- A discovery failure is an RPC error. An empty catalog is a valid empty list.
  No fixture, cache or default substitutes for it.

The runtime applies free-only policy separately in `oc_catalog.py`; the
companion asserts no eligibility.

## `open`

```json
{
  "providerID": "opencode",
  "modelID": "big-pickle",
  "hermesTools": [
    {
      "name": "write_file",
      "description": "…",
      "parameters": {"type": "object"},
      "codemode": false
    }
  ]
}
```

Returns `{"sessionID": "<id>", "location": "<cwd>", "model": "opencode/big-pickle"}`.

- `providerID`/`modelID` are used verbatim. The runtime selects them from a
  validated catalog; the companion does not alias or repair ids.
- `hermesTools` is optional. Names are Python-owned; Python reserves every
  current native registry id and emits a reversible alias (`hermes_<name>`) for
  a collision, so a name that still collides is refused before the session is
  created. `parameters` must be an object; no fixture catalog is used.
  Omitted/empty means no Hermes tools, and registrations from an earlier
  generation are removed when no live session blocks the change.
- An identical definition is idempotent. A changed definition or removal is
  applied only between generations: while any live session advertises the
  previous registry, `open` refuses with `invalid_state` naming the affected
  tools. Pure additions are allowed alongside a live session.
- A late native tool that takes a registered Hermes name is refused at the next
  reconciliation; native names always win.
- No session is created and no model request occurs when registration fails.
- Up to 16 live sessions per runtime. The provider normally uses one
  short-lived session per request, but tool registration is runtime-wide.

## `advance`

```json
{
  "sessionID": "<id>",
  "generation": 1,
  "snapshot": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
  "systemParts": ["HERMES SYSTEM TEXT"],
  "userText": "hi"
}
```

Returns `{"generation": 1, "authorized": true, "session": {...state...}}`.

- `generation` must be exactly the previous generation + 1. Any other value is
  `invalid_state`.
- `snapshot` is the canonical Hermes transcript in native message shape. It
  replaces the native transcript for the one authorized request. An empty
  snapshot is refused; the companion never falls back to OpenCode's retained
  history. Maximum 512 messages / 4 MiB.
- `systemParts` are appended to OpenCode's genuine system parts, never
  substituted. Maximum 1 MiB.
- `userText` is the prompt anchor for the first advance. It is not injected as
  a sentinel; the snapshot is the content.
- Each `advance` installs exactly one generation permit. A second permit cannot
  queue behind an unconsumed one. The `context` hook consumes the permit for
  the next primary request; a request without a permit is refused before
  vendor traffic. Exactly one primary response may bind to that permit.
- Generation 1 is the normal request. Later advances on the same session are
  refused in the current stock flow; the runtime opens a fresh session for each
  model request. The code retains a release boundary for protocol
  compatibility, but native JavaScript approval is blocked.
- Per-generation message/call/primary state is reset on every accepted advance.
- `advance` on a closed session is `invalid_state`.

## `read`

```json
{"sessionID": "<id>", "cursor": 12}
```

Returns:

```json
{"cursor": 14, "events": [{"seq": 13, "type": "authorized", "…": "…"}], "state": {"…": "…"}}
```

- `seq` is strictly increasing per session from 1. A reader must receive every
  sequence exactly once and in order. If the bounded event log drops an event,
  the runtime sees a gap and fails; it never treats a partial stream as history.
- `read` waits up to about 900 ms for new events, then returns an empty list
  rather than outliving the caller's HTTP window. It is safe to poll.
- On a closed session it returns immediately with `state.closed == true`.

## `results` / `settle`

```json
{
  "sessionID": "<id>",
  "results": [
    {"callID": "call_1#1", "content": "human readable", "output": {"example": true}, "isError": false}
  ]
}
```

Returns `{"settled": 1, "session": {...state...}}`.

- Settles nested `tools.*` calls made by one approved `execute` program. It
  never advances a generation and never authorizes inference. With native
  `execute` hard-refused there is normally no approved program or nested call.
- The batch is atomic: validate every entry (non-empty call id, known pending
  call, no duplicate in the batch, no previously settled id) before settling
  any. An invalid entry returns `invalid_state` and leaves all pending calls
  untouched.
- `output` is optional JSON. `isError: true` rejects the pending executor;
  the program decides whether to continue.
- Unknown or already-settled child ids are refused. Maximum 64 results per
  batch. No replay.

## `decide`

```json
{"sessionID": "<id>", "callID": "call_exec_1", "allow": true, "reason": "optional"}
```

For an open session, always returns an `invalid_state` RPC error with the
native-JavaScript unsupported message; a closed session returns the closed
error. `execute.before` refuses every program before an interpreter or executor
runs, so there is no parked gate for `decide` to release. An explicit
`allow: true` cannot authorize JavaScript.

## `state`

```json
{"sessionID": "<id>"}
```

Returns `generation`, `running`, `pending`, `gates`, `after`, `decisions`,
`executing`, `started`, `closed`, `promptError`, `snapshotMessages`,
`activeRequest` and `outcome`. Diagnostic only; not an authorization.

## `close`

```json
{"sessionID": "<id>"}
```

- Marks the session closed before teardown that can block.
- Rejects parked gates and pending executors, clears decisions, interrupts and
  removes the native session.
- Drops snapshot, system parts, calls, pending/gate maps and events, then
  deletes the live session. A bounded tombstone (128 ids) keeps
  `close`/`state`/`read` idempotent without retaining a transcript.
- Idempotent. A late result, decision, advance or nested call after close is
  refused. Closing does not start inference.

## Events and upstream SSE

Upstream OpenCode consumes the provider's Chat Completions SSE. The companion's
`http.response` hook observes the primary response through a pass-through
`TransformStream`: the stock driver sees the original bytes unchanged, while
the parser derives a bounded completion outcome (`[DONE]`, finish reason,
usage, error, truncation, call framing). Text is counted; tool-call arguments
are bounded and parsed only for framing validation, not exposed. Non-primary
response kinds are excluded and logged as `observer_excluded`.

Hermes never reads vendor SSE directly. The companion appends per-session
events and the runtime polls them through `read` (a long-poll of about 900 ms).
Events have the shape `{"seq": <int>, "type": "<type>", ...data}`.

Production event types include:

- `opened {model}`, `advanced {generation}`,
  `snapshot_applied {generation, messages, systemParts, nativeSystemParts}`;
- `authorized {generation, requestSeq}`, `unauthorized {message}`,
  `tool_guard {name, reason}`;
- `completion {generation, requestSeq, status, ok, done, finish, usage, error,
  calls, textChars, bytes, events}`, `correlation_error {reason}`,
  `observer_excluded {kind}`;
- `step_started`, `step_streamed {generation, assistantMessageID, primary,
  calls, completionValidated, completion?}`, `response_unverified`,
  `step_ended`, `step_failed`, `tool_failed`, `text`;
- `nested_call`, `nested_settled`;
- `execute_refused {callID, generation, reason}` — the model-facing refusal;
- `execute_unexpected {callID, generation, status}` — defence in depth if a
  program somehow reaches `execute.after`;
- `catalog_filtered {removed}` — native descriptors removed from the
  model-facing request;
- `native_refused {callID, generation, name, reason}` — an unadvertised native
  tool outside `read`/`execute` was emitted;
- `reasoning {assistantMessageID, text, native}` and
  `text {assistantMessageID, text, native}` from OpenCode's own ending events;
- `prompt_settled`, `prompt_error`, `event_stream_error`, `closed`.

A `completion` is `ok: true` only for a complete, well-framed SSE response with
terminal `[DONE]`, a supported terminal finish reason (`stop`, `tool_calls`,
`length`), valid `prompt_tokens`/`completion_tokens` (zero counts are valid),
consistent call metadata and no error event. Missing, malformed or inconsistent
usage is a rejection, never zero and never `null`-as-success.

## Tool registry rules

1. Every registered native tool has its executor replaced. A native definition
   (`name`, `description`, `input`, `output`) is never edited.
2. Hermes tools are additive entries registered through the public editor with
   their actual `name`, `description` and `parameters`.
3. Native registry ids are reserved; Python supplies a reversible alias (for
   example `hermes_patch`) instead of colliding. The companion refuses any
   remaining collision before traffic and never silently renames.
4. One catalog policy runs in the `context` hook: keep native `read` (only while
   `read_file` is registered) and every genuine Hermes entry; delete every
   other native descriptor before the request. Removal is by identity, never a
   name filter, so a Hermes tool is never dropped because OpenCode shares its
   name. Removed names are reported as `catalog_filtered`.
5. Reconciliation runs at setup, `hello`, `advance` and the `context` hook. It
   wraps late native tools, verifies every native signature is unchanged and
   verifies every registered Hermes definition still matches.
6. A tool advertised to the model without a wrapped executor is refused before
   inference. A call to any tool outside the exposed set is refused in
   `execute.before`, whether or not the model saw it.

## Native JavaScript refusal

- `execute.before` checks every call. The exposed native `read` and registered
  Hermes tools may dispatch through their held executor. Native `execute`
  emits `execute_refused`; any other unexposed native emits `native_refused`.
  Both throw before an interpreter or executor runs.
- `execute.after` is defence in depth: if an `execute` completion is ever
  observed, it emits `execute_unexpected` and throws. No JavaScript result is
  parked for a later advance.
- `decide` never authorizes a program and never creates a gate.
- Code Mode's `fetch`/`webfetch` permission is denied because Hermes owns
  network access.
- `execute_code` itself is only an additive Hermes tool: the companion
  registers the exact schema `open` supplies and never invents one.

## Bounds

- Event log: 512 events per session; older events are dropped. That is a hard
  event gap for the runtime, never silent truncation.
- SSE parser: 256 KiB per line/SSE event (not per transport chunk), 64 tool
  calls, 1 MiB total tool-call arguments, 100,000 SSE events. Violations set
  `ok: false`.
- Snapshot: 512 messages / 4 MiB; system parts: 1 MiB.
- Hermes tool definitions: 256 per runtime; parameters must be JSON objects.
- Pending nested executors: 64 per session; result batches: 64 entries.
- Live sessions: 16 per runtime; closed tombstones: 128 ids.
- Nested calls: 32 per program; child ids are `<parentCallID>#<n>` and are
  never reused.

## Test-only boundary

`tests/fixtures/companion/index.ts` is the only companion that registers the
deterministic fixture provider (`hermes-fixture/nonce`). It is selected by
setting `HERMES_OPENCODE_COMPANION` to that directory in fixture tests.
Production `companion/` does not read `fixtureURL`, does not register a
provider and has no `parseProbe` RPC.
