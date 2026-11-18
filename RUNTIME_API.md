# Native runtime and client API

Status: implementation contract for `oc_runtime.py`, `oc_binary.py`,
`oc_watchdog.py`, `oc_affinity.py` and `hermes_oc_client.py` in the current
source (4.2.0). The official OpenCode binary is
unmodified. Hermes owns instructions, canonical history, approvals and every
Hermes tool effect; OpenCode owns its process, model discovery, authentication
and vendor transport.

## Version and binary

- Required: official OpenCode **>= 2.3.0**. Earlier versions are refused before
  the server spawns; newer versions are accepted only if the loaded companion
  reports the expected protocol/id/version at handshake.
- Resolution: explicit `binary=` argument (tests), then a usable `opencode` on
  the Hermes process `PATH`, then
  `plugins.entries.oc-free-provider.settings.binary_path` in the active
  profile's `config.yaml`, then a validated hint from the host. There is **no
  environment-variable binary setting**, and no download/install path.
- Every candidate is validated with a bounded `<binary> --version` probe (argv
  list, `shell=False`, scrubbed environment). A stale hint or stale saved path
  never wins over a working PATH entry; it falls through to the next candidate.
- Companion protocol: `5`, companion id `hermes-oc-native`. Selection: bundled
  `companion/index.ts` next to `oc_runtime.py`; `companion=` or
  `HERMES_OPENCODE_COMPANION` override for packaging/tests. A missing or
  mismatched companion refuses before inference.
- Runtime root: `root=`/`workspace=` (or `HERMES_OPENCODE_RUNTIME_ROOT`) name
  **parent** directories only. The runtime creates a fresh `mkdtemp` child under
  that parent and writes/removes only that child. Pre-existing caller files are
  never modified or deleted. Without a parent it uses the system temp directory.

## Process and authentication

One `NativeRuntime` owns one `opencode serve` process:

```text
opencode serve --stdio --port 0 --hostname 127.0.0.1 --log-level warning
```

- Launch is an argv list with no shell. On POSIX the child is a new session
  leader; on Windows it gets a new process group. Teardown signals only a group
  whose ownership is proven, then falls back to the exact child handle.
- The child gets a fresh `secrets.token_urlsafe(32)` in `OPENCODE_PASSWORD`
  for that process only. It is not read from the parent environment, written
  to disk, or reused across runtimes.
- The child environment is an allowlist (`PATH`, locale and SSL variables)
  plus `HOME`, `XDG_*`, `TMPDIR` and `OPENCODE_PASSWORD` pointing into the
  private runtime root. API keys, `OPENCODE_CONFIG`, preload hooks and unrelated
  credentials are not inherited. The constructor's `env` mapping can add
  string overrides for packaging/tests; production leaves it empty.
- The private `opencode.json` loads the bundled companion directory as
  `plugins[].package`, disables automatic compaction and snapshots, and is
  written `0600`. The runtime workspace is private and initialized as a git
  repository if `git` is available.
- Startup readiness is **stdio only**: the first stdout line must be a JSON
  object containing the loopback `url`. The URL is rejected unless it is
  `http`, points at `127.0.0.1`/`::1`/`localhost`, has a port, and carries no
  userinfo or unexpected path. After the stdio line, `hello` retries RPC until
  `startup_timeout`.
- Teardown closes stdin, waits, then signals the owned process group (POSIX) or
  terminates the child handle (Windows). If ownership cannot be proven, only the
  exact child handle is killed; an unrelated process group is never signalled.
  `cleanup=True` removes only the runtime's `mkdtemp` root.

## `oc_runtime.py`

### `NativeRuntime`

```python
NativeRuntime(
    *,
    binary: str | Path | None = None,
    binary_hints: Iterable[str] | None = None,
    companion: str | Path | None = None,
    root: str | Path | None = None,
    workspace: str | Path | None = None,
    durable_root: bool = False,
    startup_timeout: float = 60.0,
    rpc_timeout: float = 120.0,
    execute_deadline: float = 120.0,
    cleanup: bool = True,
    env: dict | None = None,
)
```

`execute_deadline` is still accepted and a watchdog object is constructed to
prove ownership and bound teardown, but no supported path arms a JavaScript
deadline. Native `execute` is refused before any interpreter runs.

`durable_root=True` treats `root=` as the owned runtime directory itself
(creating it if absent) instead of creating a `mkdtemp` child. Retained
affinity lanes use this with `cleanup=False`: process eviction stops the server
but keeps `project/`, `data/` and the rest of the native workspace for the next
process. Callers own removal.

- `start() -> NativeRuntime`: validates the binary version, prepares the
  private root, spawns the process, validates the stdio readiness URL and
  performs `hello()`. Any failure kills/reaps the owned process group.
- `hello() -> dict`: cached handshake. Retries only transient
  `rpc.unavailable`/`rpc.internal`/`runtime.transport` until
  `startup_timeout`; validates `protocol == 5`, companion id, binary version
  >= 2.3.0, and requires every registered tool to report `forwarded: true`.
  Each tool row also reports `hermes: true` when it was registered from a Hermes
  schema rather than being a native OpenCode tool.
- `catalog(*, timeout=None) -> list[dict]`: native model records. Tries the
  companion `catalog` RPC first; falls back to official `GET /api/model`.
  Records are sanitized to `id, modelID, providerID, name, type, family,
  package, providerPackage, api, capabilities, compatibility, cost, limit,
  time, status, enabled, variants`. Provider settings, API keys, headers, base
  URLs and credentials are never returned.
- `list_models(*, timeout=None) -> list[str]`: native ids in catalog order.
- `new_session(provider_id, model_id, hermes_tools=None, *, session_id=None) -> Session`.
  With `session_id`, the companion adopts that exact existing native session
  (never minting or recreating an id) after verifying it exists.
- `get_session(session_id, *, timeout=None) -> dict | None`: official
  `GET /api/session/<id>`. Returns `None` only for a real 404; transport,
  permission or storage errors propagate. Used to corroborate a persisted id
  and its workspace before reuse.
- `rpc(method, payload=None, *, timeout=None) -> dict`: authenticated loopback
  JSON RPC.
- `get(path, *, timeout=None) -> dict`: authenticated loopback GET, only
  `/api/...`.
- `stop() -> int`: idempotent teardown as described above.
- Attributes: `alive`, `pid`, `url`, `info`, `watchdog`.

### Loopback RPC

The companion registers OpenCode plugin RPC:

```text
POST {url}/api/rpc/hermes-oc-native/{method}?location[directory]={workspace}
Content-Type: application/json
Authorization: Basic base64("opencode:<OPENCODE_PASSWORD>")

{"input": {...}}
```

- Every request uses the per-process Basic password. The server binds
  `127.0.0.1` on an ephemeral port.
- Redirects are refused. Only `/api/...` paths are accepted. Responses are
  bounded at 64 MiB; error bodies at 64 KiB. Non-JSON, non-object envelopes,
  HTTP errors and transport failures raise `RpcRefused`.
- `get("/api/model")` uses the same authentication and path restriction.

### `Session`

```python
session = runtime.new_session("opencode", "big-pickle", hermes_tools=[...])
result = session.generate(snapshot=[...], system_parts=[...], user_text="...")
```

`generate()` calls `runtime.reconcile()` (a fresh `hello` to wrap late native
tools), issues exactly one `advance` (one generation permit), waits for the
primary boundary, and applies the completion gate:

- require an observed `completion` event bound to this generation with
  `ok is True`, `done is True`, a supported terminal finish reason
  (`stop`/`tool_calls`/`length`) and no error;
- require valid `prompt_tokens` and `completion_tokens` in that completion (zero
  is valid; missing, negative, non-integer or inconsistent totals refuse);
- require `step_streamed.completionValidated is True`; expose no call from a
  truncated, errored or unverified response;
- reject `length` together with calls, duplicate/empty call ids, non-object
  arguments, `tool_calls` finish with zero calls, and conflicting finish
  reasons;
- a timeout is never success.

Returns `StepResult(text, reasoning, calls, finish, usage, generation,
assistant_message_id, session, step_usage, completion, native_parts)`.
`calls` are `ToolCall(call_id, name, arguments, native)`. `text` and `reasoning`
come from OpenCode's own ending events (never from the raw SSE body).
`native_parts` is `{"parts": [{"type", "index", "native", ...}]}` carrying the
opaque per-part metadata received from those events, position-keyed for replay.

Other methods: `read()`, `state()`, `wait_pending()`, `decide(allow, reason)`,
`settle(results)`, `finish()`, `events_until(predicate, timeout)`, `close()`.

- `advance()` moves the event cursor past the authorization boundary, so a late
  interrupt/step event from an earlier generation cannot be attributed to the
  new one.
- `finish()` is the explicit generation boundary for a retained session. The
  companion rejects parked executor results, interrupts an in-flight prompt,
  waits (bounded) for the run to leave the active set, keeps the native session
  and returns `{settled, interrupted, seq}`. `advance(generation=N+1)` is
  refused until `finish()` settles generation N. A false `settled` means the
  caller must abandon that native session and allocate a fresh official id.

- `read()` polls the companion's per-session event log with a cursor. The
  companion long-polls for about 900 ms and returns an empty batch rather than
  outliving the HTTP window.
- Event cursors must be contiguous. A dropped event is a hard `EventGap`, never
  a silent partial transcript.
- `decide()` remains a protocol method, but the current companion always
  returns `invalid_state` for it; there is no supported JavaScript approval.
- `settle()` is only for nested `tools.*` results and is unreachable without an
  approved program. `results` is canonical; `settle` is a one-revision alias.

### Canonical snapshot

`oc_translate.snapshot_from_openai(messages, plan=...)` is the production
converter; `OpenCodeClient.complete()` uses it and maps `SnapshotError` to
`SnapshotRefused`. The runtime keeps `canonical_messages(messages)` as a small
stdlib-only text fallback for direct runtime users and tests. The production
translator additionally preserves media, native call provenance and wire-alias
replay.

The runtime rejects image/audio/file parts in its fallback converter rather than
dropping them. The production translator carries representable media as native
`media`/`file` parts. In both cases a request without a non-empty user message
is refused because OpenCode needs a prompt anchor.

Generations without a logical scope (auxiliary/stateless calls) use one
short-lived native session per request and close it after the step. A scoped
request (`hermes_session_context`, emitted by the profile from
`cache_scope_id`/`session_id`) retains one native session per active
profile/scope/model lane. `finish()` closes the previous generation, then the
next request replaces the native transcript with the canonical Hermes snapshot.
OpenCode's retained transcript is never reused as Hermes history and the
companion never executes a Hermes tool.

### Retained affinity lane

- The mapping is one profile-scoped SQLite row written through the documented
  `plugins.plugin_storage.plugin_db("oc-free-provider")`: schema version, local
  scope key, exact official native id, native provider/model and last-used.
  Hermes' main session database is never opened or modified.
- The profile sends both raw ids (`session_id`, `cache_scope_id`) and its
  pinned `profile_home` in `hermes_session_context`; the client derives the
  local lane:
  - normal/compression/`/branch`/gateway: `cache_scope_id` (stable across
    compression rotation and per-response gateway ids);
  - `sub_<sha>` delegate scopes: `sub:<scope>:<session_id>`, because delegate
    siblings share one cache bucket. Delegate compression rotates the physical
    id and therefore resets delegate affinity explicitly;
  - `/btw` and same-model review share the parent's two ids by design, so they
    share one lane and are serialized (a concurrent call gets busy);
  - missing `session_id` on a `sub_` scope or both ids absent stays ephemeral,
    never a shared anonymous bucket.
  The client validates the pinned profile against the active profile before any
  runtime or affinity-database access. A marker built under one profile and
  resolved under another is refused; it is never silently rebound. The local
  key is a SHA-256 of `(profile home, lane, provider, model)`. It is never a
  wire id.
- First use calls native create without an id, then persists the exact returned
  id. Reuse verifies it with native `GET /api/session/<id>` (id and workspace).
  Only a real 404 resets affinity: allocate a fresh official id, replace the
  mapping and report `Completion.affinity_reset = True` plus a log warning.
  Transport/permission/storage errors propagate instead of being treated as
  absence.
- Native `project/`, `data/`, config and cache live under
  `<profile>/plugin-data/oc-free-provider/native/<scope-key>/` and survive idle
  process eviction (`durable_root=True`, `cleanup=False`). The root is stable,
  so the native system prefix that names the working directory does not change
  between turns.
- Per-scope POSIX/Windows advisory ownership is held for the native runtime
  lifetime, including idle: a competing process gets a bounded busy refusal
  before it can start a second server on the same durable root. The owner stops
  the server before releasing the lock on close/eviction. An in-process
  non-blocking generation lock serializes same-scope threads. SQLite
  transactions stay short and are never open during inference.
- `RuntimePool.TTL_SECONDS` (idle process eviction) and `MAX_OWNERS` bound live
  runtimes. `oc_affinity.prune()` removes inactive owned scope directories and
  their rows under the same lock (default 14 days), re-reading the row's
  `last_used` under that lock before removal; an expired scope loses affinity
  explicitly on its next use. Mapping rows contain no transcript or credentials;
  native data may contain prompts/results and is sensitive.

## `hermes_oc_client.py`

### `OpenCodeClient`

```python
OpenCodeClient(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout: float = 120.0,
    binary: str | Path | None = None,
    companion: str | Path | None = None,
    root: str | Path | None = None,
    startup_timeout: float = 60.0,
    cleanup: bool = True,
    runtime: NativeRuntime | None = None,
    runtime_owner: RuntimeOwner | None = None,
    owner_key: str | None = None,
    command: str | None = None,
    args: Any = None,
    **kwargs: Any,
)
```

`api_key`, `base_url` and `args` are accepted but ignored for profile-call
compatibility. `command` is core's resolved external-process executable: when it
is an absolute existing path it becomes a validated **hint** to the runtime
resolver (never used unvalidated, never allowed to outrank a usable PATH).
Constructor `session_id`/`profile_home` are only the legacy explicit owner key
described below. Retained affinity uses the per-request
`hermes_session_context` top-level field (consumed by the client, never sent to
the vendor) emitted by `ProviderProfile.build_api_kwargs_extras`.
Native OpenCode state owns authentication; this facade owns no HTTP transport.
`LOGICAL_BASE_URL = "opencode-native://local"` is identity metadata only.
`supports_streaming = False`.

`runtime_owner`/`owner_key` share one process owner across facade
reconstruction for one profile/session conversation. When no explicit
`owner_key` is supplied, the client derives one from the authoritative
`session_id` and profile home via `RuntimePool.key_for(profile_home,
session_id)`. Core may supply `session_id`/`profile_home` through client kwargs;
they are used only to derive that key. The key is a JSON array, so path/session
text cannot collide. Owners are reference-counted and idle-expire after
`RuntimePool.TTL_SECONDS` (300 s); a daemon reaper checks every second.
`RuntimePool.close_for_session(profile_home, session_id)` and
`RuntimePool.close(key)` are explicit teardown. A facade `close()` releases its
reference. No session identity means an isolated owner per facade.

If an owned process died, a new explicit `complete()` or catalog call may
replace it with a freshly started runtime under the same owner. Recovery is not
a retry: the failed generation is not replayed, and sessions/results from the
old runtime stay refused. A runtime passed explicitly with `runtime=` is never
restarted. A stopped runtime is not restarted automatically.

### Catalog and inference

- `catalog()` -> raw sanitized native records.
- `list_models()` -> `oc_catalog.free_model_ids(records)`.
- `list_model_details()` -> raw sanitized records; the provider profile applies
  `oc_catalog.describe_models()` for picker rows.
- Before every inference the client calls `runtime.catalog()` and
  `oc_catalog.select_model(records, requested_model)`. It uses the returned
  `providerID`/`id`; it does not normalize, alias or repair an id. `CatalogError`
  becomes `ModelRefused` before a session opens. An unsupported transport or
  package becomes `TransportRefused`.

```python
completion = client.complete(messages=[...], tools=[...])
completion = await client.acomplete(messages=[...], tools=[...])
```

`complete()` is synchronous; `acomplete()` wraps it with `asyncio.to_thread`.
Accepted request keys are `messages`, `tools`, `model`, `timeout`, `stream`
(only `False`/absent), `tool_choice` (only absent/`"auto"`) and `extra_headers`
containing only local string `traceparent` metadata. `stream=True`, non-auto
`tool_choice`, `temperature`, `top_p`, `max_tokens`, `max_completion_tokens`,
`stop`, `response_format`, `reasoning_effort`, `extra_body` and unknown controls
raise `ControlRefused` **at create time**, even through the lazy
`chat.completions.create` facade, before the runtime starts or any model request
occurs. That keeps auxiliary/Relay callers from receiving an object they cannot
iterate. Iterating a non-stream completion raises the same clear refusal.

Native `read`, `write`, `glob`, `grep`, `question` and `todowrite` are
advertised only while their Hermes target (`read_file`, `write_file`,
`search_files`, `clarify`, `todo_list`) is enabled, and that target schema is
then withheld. The client sends the kept set as `nativeTools` to `open` and
`advance`. Every other enabled Hermes tool is advertised with its full schema.
If `read_file` is not among the supplied tools, the client raises
`ControlRefused` before inference. The runtime config sets
`experimental.todo: true` so the pinned binary registers `todowrite`. Every native registry id is
reserved, so a Hermes tool with a colliding name is advertised as
`hermes_<name>` and mapped back to its Hermes identity on dispatch.

Mapped native calls whose arguments cannot be represented raise
`ControlRefused` before any effect. Absolute paths into the private runtime root
are refused; relative paths resolve against the Hermes working directory, and a
guidance system part tells the model so. Native calls outside the kept set are
never dispatched. If a model emits one,
the client returns a terminal refusal completion naming the tool, with no tool
call, no effect and no extra inference. Native `execute` keeps its dedicated
refusal message.

Result: `Completion(choices, usage, model, finish_reason, text, affinity_reset,
reasoning, tool_calls, native_tool_calls, native_usage)`. `affinity_reset` is
true only when a persisted native session was missing and a fresh official id
replaced the mapping. `usage` is a `Usage` object with
measured values; it is always present for a successful completion (zero counts
are valid). `tool_calls` are ordinary Hermes calls; `native_tool_calls` keeps
the raw native batch for diagnostics. `choices[0].message.reasoning_details`
carries the opaque native metadata entry described in
[`TRANSLATION_API.md`](TRANSLATION_API.md). The client never executes a tool;
the Hermes agent loop owns dispatch and approvals.

`resolve_aux_model` returns `""`: auxiliary requests send temperature and token
limits that this transport cannot map yet, so the provider advertises no cheap
model rather than promising controls it would reject.

`process_command` is a dynamic profile property (PATH -> active profile setting
-> bare `opencode`), not a value frozen at registration, so core's credential
gate sees external config edits and profile switches in the same process. It is
cheap (no subprocess). Runtime resolution additionally accepts core's resolved
command as a validated fallback hint, after PATH and config.

`create_lazy()` returns a value that resolves on attribute access or `await`.
`chat.completions.create()` from synchronous code runs the request before it
returns, like an OpenAI client, so Hermes' worker watchdog (interrupt and stale
kill) covers it. Called from a running event loop it returns immediately and
the request runs in a worker thread when awaited.
`stream()` raises `ControlRefused("native streaming is not advertised")`.
`close()`/`aclose()` release the owner or stop an owned runtime; idempotent.

## Error contract

All public errors inherit `OpenCodeRuntimeError` and expose `.code`:
`runtime.binary`, `runtime.startup`, `runtime.handshake`, `runtime.rpc`,
`runtime.session`, `runtime.completion`, `runtime.event_gap`,
`runtime.control`, `runtime.execute`, `runtime.snapshot`, `runtime.model`,
`runtime.transport`, `runtime.closed`. A timeout, truncation, malformed event,
unverified completion or dead process is never returned as a successful
completion. A local contract failure does not fabricate an upstream HTTP status.
Every error also exposes `.status_code`: the upstream 4xx/5xx the official
runtime reported (`completion.status`, `step_failed.error.status`, or a
`response_unverified` outcome), else `None`.

## Test-only fixture boundary

Fixture registration is a test subclass in `tests/runtime/helpers.py`
(`FixtureRuntime`), not a production constructor option. Production
`NativeRuntime`/`OpenCodeClient` build no `fixtureURL` plugin option and load no
fixture provider or fixture RPC. `tests/fixtures/companion/` adds one zero-price
`opencode/nonce` model so the strict `oc_catalog.select_model` policy stays
active in tests; there is no test-only selector bypass. Recovery tests replace
the same subclass through `RuntimeOwner(factory=...)`; production owners never
pass a factory.
