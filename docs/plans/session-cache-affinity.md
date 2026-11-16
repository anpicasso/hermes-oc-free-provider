# Session identity and cache-affinity implementation plan

> **Status:** Implemented in 4.2.0. Regression and real Hermes CLI acceptance passed; the baseline observations below document why the change was needed.

**Goal:** Keep a stable OpenCode identity and request prefix for a logical Hermes conversation without introducing a second transcript authority.

**Architecture:** Obtain logical identity through the existing per-request provider hook, reuse the plugin's bounded in-memory runtime registry, and retain one native session per active logical conversation/model. Let the official OpenCode binary allocate the native session ID. Persist a small profile-scoped mapping to that ID and preserve its native data/workspace; never synthesize a wire session ID from a hash. Hermes remains the sole canonical transcript authority; the companion replaces context rather than appending a replay.

**Tech stack:** Existing Python stdlib runtime/pool, one profile-scoped SQLite mapping opened through Hermes' documented `plugins.plugin_storage.plugin_db("oc-free-provider")`, Hermes ProviderProfile callbacks, official OpenCode >=2.3.0 and TypeScript companion. This supersedes the per-scope JSON proposal. No schema changes or direct SQL writes to Hermes' session database or OpenCode's database.

## Pre-implementation evidence and corrections

- `hermes_oc_client.py:588-597` creates and closes a native session on every completion. The pool retains a process, not a conversation.
- Constructor `session_id` support does not prove stock core supplies it. The real request seam is `ProviderProfile.build_api_kwargs_extras`; installed core `agent/transports/chat_completions.py:664-671` passes session_id and cache_scope_id and merges top-level extras.
- `companion/index.ts:1321-1325` explicitly refuses ordinary same-session advancement after generation one. Removing Python's close alone cannot implement retention.
- `companion/index.ts:1005` replaces native messages with the canonical snapshot. Generic advice to send only deltas does NOT apply to this bridge: retain full snapshot replacement and test that no old native state leaks into the request.
- `companion/index.ts:1471-1476` interrupts and removes the native session. Runtime `_prepare` creates random private project/data directories; stop deletes them (`oc_runtime.py:514-515,562,945-946`). A saved native ID alone would point at deleted state.
- Inspected OpenCode source `38bd73789a43bf043599d6196af96583228035d7`: `packages/core/src/session.ts:89-98,290-293` accepts an optional create ID and returns existing state for that ID; `packages/schema/src/session-id.ts` accepts the ses prefix. `session/model-request.ts:321-336` derives affinity headers and promptCacheKey from the session; `session/affinity.ts` chooses parent/fork/self. Confirm the installed official binary agrees before implementation.
- Original public commit `7492876cae926c8b33c76fc32ab6560c26922c12`, `tests/test_client.py:63-71`, explicitly tests `^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$` and descending timestamp shape, with a recorded gateway-rejection warning for ascending timestamps. Official `packages/schema/src/identifier.ts:16-31` generates that timestamp/counter + random suffix structure. The permissive local schema and special hash handling in promptCacheKey are NOT proof of hosted acceptance. The earlier deterministic native-ID proposal is withdrawn; the official generator owns all wire IDs.
- Native built-in instructions include the working directory and date (`instructions/builtins.ts`). A stable ID alone does not fix prefix changes caused by random runtime workspaces.
- Existing hosted publication probe reported 1772 cache-read tokens across two calls. This proves caching occurred, not optimal affinity or an improvement attributable to retention.

## Storage decision

1. Reuse Hermes session/cache-scope identity; do not add fields/tables to its SQLite database.
2. Keep active process/session handles and generation state in the existing plugin pool, partitioned by active profile, logical scope, selected provider/model and runtime configuration identity.
3. Hash a versioned unambiguous tuple of profile namespace, logical scope and provider/model ONLY for the local registry key/directory name, never for a native session ID. Prefer canonical cache_scope_id, session_id otherwise. Missing identity means isolated ephemeral work, never a shared anonymous bucket.
4. On first use, call native create WITHOUT an id, then persist the exact returned ID. On process loss, reopen the same owned native data/workspace and call native get with that exact ID. Do not recreate a missing session under an invented or previously deleted ID. If the session genuinely no longer exists, allocate a fresh official ID, atomically replace the mapping, and report that affinity reset. Transport/permission/storage errors are not evidence that a session is absent.
5. Recommended mapping store: `<profile>/plugin-data/oc-free-provider/data.db`, opened with the documented `plugins.plugin_storage.plugin_db`. One table, one row per logical scope/model: schema version, scope key, exact native session ID, provider/model and last-used timestamp. Delegates add rows, not JSON files. Use short explicit transactions and parameterized SQL; the helper owns connection setup, not transactions or runtime ownership. Re-resolve the active profile on acquisition; never reuse a connection across profiles.
6. Keep native `project/` and `data/` below owned scope directories under the same plugin data root. SQLite consolidates mappings, NOT native workspaces or native databases. Those still need bounded retention and cleanup. Idle runtime eviction stops processes without deleting retained native data. Explicit reset/retention cleanup removes inactive native data and its mapping under the ownership lock with recoverable ordering; filesystem removal and database commit are not one atomic transaction. Native data may contain prompts/results and must be treated as sensitive; mapping rows contain no transcript or credentials.
7. Validate record types/bounds and corroborate the ID/model/location through native get before use; a regex alone is not ownership validation. Retain per-scope cross-process advisory ownership locks, to be tested on POSIX/Windows; a competing process receives a bounded busy error. SQLite mapping transactions do not serialize a long-lived native generation. Never hold a database transaction over network calls or inference. Initial native creation and mapping publication are not atomic: interruption can orphan an unused native record; recover under the ownership lock without claiming exactly-once allocation. Corruption is reported, not silently repaired.

### Why not the main Hermes state.db?

- Installed core `b299329313957328f72425054fdee8c8e4bb9af3` does expose `SessionDB.patch_session_model_config`, `get_session_model_config_value` and `update_conversation_model_config_value` (`hermes_state_sessions.py:782-798`). The last atomically updates a key at the compression tip, but silently returns None when the session row is absent; its updater must be pure because retries can call it repeatedly. This makes native allocation inside that callback invalid.
- A disposable SessionDB probe verified writing a namespaced key without losing another key, plus the missing-row no-op. This proves technical feasibility, NOT a documented provider-plugin storage contract or preservation across fork/delegation/whole-model-config rewrites. No production database was opened for this probe.
- The official plugin guide instead documents `from plugins.plugin_storage import plugin_db`. Installed `plugins/plugin_storage.py:27-46` resolves the current profile and uses Hermes' shared SQLite setup (WAL where safe, fallback otherwise). A disposable probe verified table insertion and reopening through this exact helper. Unlike `ctx.state`, it requires no standalone PluginContext and is callable from a model-provider module at request time.
- Prefer that supported plugin-owned SQLite helper now: no per-session JSON and no invasive core schema change. Using main state.db remains a separate option only after proving the model-provider access/lifecycle contract, row availability, compression/fork/delegate isolation, and safe persistence against all core writers. Do not call an importable core method a guaranteed plugin API.

## Tasks and acceptance gates

### 1. Prove the identity seam (before changing production behavior)
- Files: `__init__.py`, new `tests/test_session_affinity.py`.
- Through a real isolated stock AIAgent kwargs builder, verify per-request scope reaches a synthetic profile callback and custom client. Check ordinary turn, resumed session, compression scope, child delegation and absent auxiliary scope.
- Proposed internal top-level field: `hermes_session_context`, explicitly consumed by the custom client before generation validation. Do not send it as provider extra_body or leak it into any generic fallback client.
- Keep active-profile resolution authoritative at request time. Do not bind a shared client permanently to its first caller's session.

### 2. Prove native identity recovery with the official binary
- Test-only companion/probe, temporary profile and loopback fixture; zero hosted calls needed.
- Create without an ID, read back the exact returned ID, stop and restart using the SAME owned native data/project path, and get that existing session again. Capture native request affinity headers and any supported body cache key.
- Test removal/missing-store recovery separately: a lost record gets a new official ID with an explicit affinity-reset diagnostic. Do not manually recreate a deleted ID or bypass event-log tombstones.
- Stop and revise the design if official API/schema behavior contradicts source. Do not rewrite upstream headers/body to pretend an unsupported identity works.

### 3. Retain the live session safely
- Files: `hermes_oc_client.py`, `oc_runtime.py`, `companion/index.ts`; tests `test_native_bridge.py`, `test_runtime_recovery.py`.
- Extend existing RuntimeOwner/RuntimePool rather than create a parallel registry. An entry owns runtime, native session, generation counter and in-flight state. Facade close releases a lease; it does not destroy the logical conversation immediately.
- Add an explicit finish-generation operation/state transition: stop/settle parked native tool execution without allowing another inference, discard pending native results, retain session identity, then accept the next canonical snapshot and one new authorization.
- On ordinary completed text, allow another prompt in the same session after the previous prompt has settled. Do not just delete the generation>1 guard: reset started/running state and isolate delayed callbacks with a generation token.
- Cancellation poisons only the affected generation/runtime; next call reopens retained native data, verifies the session is idle, and replaces context with the Hermes snapshot. If that state cannot be safely recovered, allocate a new official session and report affinity reset. Never replay a Hermes tool side effect as recovery.
- Same-scope overlapping calls must not share a mutable companion/tool registry. Prefer an explicit busy error for the retained lane; if compatibility requires an ephemeral fallback, use a separate runtime and document affinity behavior. Independent delegates use distinct identities/runtimes.
- Bound idle entries and total runtime count; reuse existing TTL/reaper with request leases. Process TTL cleanup preserves durable mapping/native state. A separate documented disk-retention policy removes inactive owned scope directories under lock; expired scopes lose affinity explicitly.

### 4. Preserve request-prefix stability and isolation
- Files: `oc_runtime.py`, `oc_translate.py` only if capture proves a change is necessary.
- Stable owned workspace plus deterministic schema ordering when the enabled schema set is unchanged. Preserve genuine native system instructions and opaque replay metadata.
- Full snapshot replacement remains mandatory. Compare captured outbound messages to the canonical expected transcript; no duplicate history, old failed assistant turn, fabricated tool result or native memory bleed.
- Model switch selects a separate lane. Compression may retain logical affinity but legitimately changes the cached prefix; do not promise a cache hit after compaction. Fork/reset must follow core identity semantics, not infer ancestry from content.

### 5. Verify recovery and measure caching
- Fixture regressions: same identity across two user turns and a tool continuation; isolation between profiles/delegates; facade rebuild; resumed request; model switch; missing scope; cancel/retry; concurrent same-scope calls; runtime crash; TTL expiry; no extra inference or native tool effect.
- Capture assertions: same native affinity on sequential calls/reconstruction; exact stable system/tool prefix where inputs did not change; requested canonical messages only.
- Hosted bounded A/B: same model and equivalent long prefix, record per-request prompt/cache-read/cache-write tokens and latency; compare old ephemeral-session behavior with retained behavior across repeated runs. Do not infer a causal gain from one cached response. Report zero/inconsistent cache hits honestly; server TTL/routing remain outside plugin control.
- Run the full existing suite with isolated HERMES_HOME set before imports, actual Hermes interpreter and explicit PYTHONPATH. Run validate/doctor plus actual CLI resume, delegation and steering checks. No production installation, release or gateway restart as part of this analysis.

## Commands for the implementation phase

Set `CORE` to the supported Hermes checkout and `PY` to its own environment interpreter, not system Python. Set `HERMES_OPENCODE_BINARY` only as the existing test harness override (not a product configuration variable).

```sh
HERMES_HOME="$(mktemp -d "$TMPDIR/oc-affinity-home.XXXXXX")" \
  PYTHONPATH="$CORE" "$PY" -m unittest tests.test_session_affinity -v
HERMES_HOME="$(mktemp -d "$TMPDIR/oc-affinity-suite.XXXXXX")" \
  PYTHONPATH="$CORE" "$PY" -m unittest discover -s tests -t . -q
hermes plugins validate . --json
hermes plugins doctor . --ci
```

Expected future acceptance: all new assertions and existing tests pass; actual integration test counts/results must be recorded after implementation.

### Executed feasibility checks (official binary 2.3.0, isolated local probes)

- Native HTTP API create without `id` returned `ses_e1d167a23ffedLr2OkXJyrfiI5`; get returned 200 with that exact ID. After SIGTERM and a new serve process with the same owned HOME/XDG data/workspace, get returned 200 with the same ID and creation timestamp. This proves persistence across process restart, not crash consistency during an in-flight generation.
- A separate native session received two prompts against a local SSE fixture. Both captured provider requests carried identical `x-opencode-session-id`, `x-session-affinity`, `x-session-id` and `x-opencode-session`, all equal to the official-generated session ID. `prompt_cache_key` was absent for that fixture configuration; do not require it universally or inject it manually.
- Parent read back evidence: `oc-affinity-probe/evidence.json` and `oc-affinity-probe/affinity-evidence.json` under the local scratch directory. Reproduction scripts are `probe.py` and `probe_affinity.py` there.
- No hosted inference was used in these probes. They do NOT prove cache savings, recovery of parked tool calls, production companion multi-generation safety, or non-Linux behavior. Those remain implementation acceptance gates above. No production code/configuration was changed.

## Explicitly out of scope

New hosted protocol families, streaming/controls, changing vendor instructions, upstream header spoofing, changing Hermes core/schema, retaining duplicate transcripts, guaranteeing provider-side cache hits, or modifying the released artifact during planning.

## Implementation status (2026-11-16)

Implemented end to end in the working tree; no commit, push, release, install or core edit.

- **Seam.** `__init__.py::OpenCodeFreeProfile.build_api_kwargs_extras` emits a versioned top-level `hermes_session_context` carrying both raw `session_id` and `cache_scope_id`, and nothing for auxiliary/absent scopes. The real `AIAgent` kwargs builder carries it; the auxiliary projection and `extra_body` never see it. `OpenCodeClient` consumes it before validation and never forwards it.
- **Mapping and lock.** `oc_affinity.py` uses `plugins.plugin_storage.plugin_db("oc-free-provider")` per profile, one `native_sessions` row per local scope key, short transactions, no transaction during inference. The per-scope advisory lock is held for the native runtime lifetime including idle; the owner stops the server before releasing it on close/eviction. Corrupt rows raise `AffinityRefused`; they are not repaired. `RuntimePool.MAX_OWNERS` and TTL bound live owners.
- **Native identity.** The official binary allocates every id. First use creates without an id and persists the exact returned string; reuse is corroborated through native `GET /api/session/<id>` (id and workspace). Only a real 404 allocates a replacement and sets `Completion.affinity_reset` plus a warning. Transport/permission/storage errors propagate. No invented wire ids.
- **Durable state.** `NativeRuntime(durable_root=True, cleanup=False)` reuses `<profile>/plugin-data/oc-free-provider/native/<scope-key>/` across process eviction/crash, keeping `project/`, `data/`, config and cache. Idle eviction stops only the process. The workspace path in OpenCode's own system prefix is therefore stable.
- **Lane identity.** The hook now carries both raw ids. Normal/compression/`/branch`/gateway use `cache_scope_id`; `sub_<sha>` delegates use `sub:<scope>:<session_id>` so siblings never merge (delegate compression resets affinity by design); `/btw` and same-model review share the parent's lane and serialize; a `sub_` scope without a physical id stays ephemeral.
- **Companion protocol 5.** `open` accepts an exact existing `sessionID`; `advance` accepts every settled generation, carries the current `hermesTools`, always prompts once, replaces the full canonical snapshot, and returns its event boundary `seq`. `finish` rejects parked executors/gates, interrupts and bounded-waits for idle, keeps the native session and reports `{settled, interrupted, seq}`. A false `settled` makes Python close that session (or poison the process lane) and allocate a fresh official id with exactly one lease; late prior-generation prompt callbacks are generation-guarded and cannot poison the next generation.
- **Concurrency and retention.** Same-scope in-process calls get a bounded busy `ControlRefused`; a cross-process owner holds the scope lock for the runtime lifetime, so two processes cannot serve one durable root and prune/reset cannot delete a live idle server's data. `prune()` re-reads `last_used`/directory age under that lock before removal.
- **Verification.** Full offline suite: 121 tests pass (`unittest discover`, isolated `HERMES_HOME` before imports, core venv interpreter, explicit `PYTHONPATH`, official 2.3.0 binary). `plugins validate` and `plugins doctor` pass with only the pre-existing private-import/isolation warnings. Evidence scripts/results live in `oc-affinity-probe/` and `oc-affinity-evidence/` under the local scratch directory, not the repo.
- **Hosted bound.** Two retained turns on `deepseek-v4.1-flash-free` used one official id (`ses_e1ce9cd78ffe…`) and read 4608/4668 cached tokens on turn 2; an ephemeral control read 4480/4492 on turn 2. Both show provider-side prefix caching, so this bounded pair does **not** establish a causal retention speedup. Server TTL/routing remain outside plugin control.
- **Known limits.** Windows lock code is present but only POSIX was exercised. A hosted A/B with more runs/models is still needed for a performance claim. Interruption can orphan an unused native record between create and mapping publication (recovered as an affinity reset on next use, not exactly-once). A machine crash mid-generation is covered by durable restart plus snapshot replacement; parked tool continuations are intentionally not replayed.
