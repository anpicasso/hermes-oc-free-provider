# Native translation API

Status: contract for `oc_translate.py` (unreleased). The module performs no tool
effects. It validates names and arguments, maps a constrained set of native
calls to ordinary Hermes calls, preserves opaque native metadata, and replays
canonical history into the native snapshot. Hermes dispatches translated calls
in its own agent loop.

## Public surface

```python
plan_tools(tools, native_names=("read",)) -> WirePlan
translate_call(native_name, arguments, *, plan=None, enabled=None, forbidden_roots=()) -> (host_name, host_arguments)
native_guidance(plan) -> str | None
encode_call_id(native_name, native_arguments, host_arguments, original_id) -> str
decode_call_id(call_id) -> (wire_name, native_arguments | None, original_id) | None
native_carrier(native_parts) -> dict | None
snapshot_from_openai(messages, *, plan=None) -> Snapshot
```

`NATIVE_MAP` maps native name -> (Hermes tool, argument mapper);
`COMPAT_TOOL = "read_file"`. `WirePlan` has `register` (Hermes definitions for
`open`), `host_to_wire`, `wire_to_host`, `native` (native descriptors kept for
this request) and `enabled` (every enabled Hermes tool, replaced or not). `Snapshot` has `system_parts`, `messages` and `user_text`.

Errors are `NativeTranslationError` subclasses with `.code`:
`native.unsupported`, `native.invalid_arguments`,
`native.compatibility_unavailable` and `snapshot.invalid`.

## Tool planning

`plan_tools` receives the enabled Hermes definitions (flat
`{name, description, parameters}` or OpenAI
`{"type":"function","function":…}`) and the reserved native registry names.

- A `NATIVE_MAP` descriptor is kept when OpenCode registered it and its Hermes
  target is enabled. The target schema is then not registered, so the model
  sees one name per capability. `search_files` is replaced only when both
  `glob` and `grep` are present.
- Every other Hermes definition is registered with its own description and
  parameter schema, byte-for-byte.
- Native registry names are reserved. A non-colliding Hermes name is advertised
  unchanged; a collision becomes `hermes_<name>`, then `_hermes_<name>`, … . The
  mapping is reversible, so a Hermes tool is never dropped because OpenCode
  happens to share its name (for example Hermes `patch` → wire `hermes_patch`
  while native `patch` exists in the registry).
- If `read_file` is not among the enabled definitions, planning raises
  `CompatibilityUnavailable`: the native `read` compatibility adapter is never
  advertised without the Hermes tool that actually runs it.
- A wire name longer than 64 characters is refused.
- Each registration is `{name: wire, description, parameters, codemode: false}`.
  The provider does not opt Hermes tools into OpenCode Code Mode.
- `register` is built only from the current enabled set. It is not a capability
  grant: Hermes only receives calls for tools it already enabled.

## Native mappings

These are constrained mappings, not parity (see the audit reports). Unknown
options and unrepresentable values raise `NativeInvalidArguments` or
`NativeUnsupported` before any effect. The documented read limit cap and the
underlying Hermes output bounds still apply.

| native | Hermes | mapping | refused |
|---|---|---|---|
| `read` | `read_file` | `offset` 0/1 -> 1; `limit` 0 -> default; `limit` capped at 2000 | `offset < 0` |
| `write` | `write_file` | identity | Hermes write guards still apply (an unread existing file is refused) |
| `glob` | `search_files` target=files | `pattern`, `path`, `limit` (default 100) | `hidden: true`, `limit < 1` |
| `grep` | `search_files` target=content | `include` -> `file_glob`; `literal` escapes regex metacharacters; `caseSensitive: false` -> `(?i)`; `limit` default 100 | `limit < 1` |
| `question` | `clarify` | `header` and option `description`s are folded into the question text; labels -> `choices`; `multiple` -> `multi_select` | more than 5 questions, more than 4 options, blank labels |
| `todowrite` | `todo_list` (session store) | positional ids `"1".."n"`, `merge: false` | blank content, unknown status, more than 256 items, content over 4000 chars |

Every path argument that is absolute and points inside `forbidden_roots` (the
client passes the private runtime root) is refused. Relative native paths reach
Hermes unchanged and resolve against the Hermes working directory.

Native `edit` and `patch` are intentionally not translated: Hermes `patch`
matches fuzzily where native `edit` is exact, and native patch grammar
(`Move to`, `End of File`) is not Hermes V4A. `shell`, `skill`, `webfetch`,
`websearch`, `subagent`, `experimental_image_generate`, `execute` and every
Code Mode tool are also unmapped. An unmapped name is returned unchanged (or as
`oc_unsupported_<name>` when it collides with an enabled Hermes tool), and a
mapped native whose target is disabled returns `oc_unsupported_<name>`.

`native_guidance(plan)` returns one system part describing these limits and the
cwd rule; the client appends it after the Hermes system parts.

Results retain the Hermes tool's own JSON; native output schemas are not
synthesized. Explicit `is_error` is preserved. For mapped native calls, a JSON
`error`, `success: false`, or cancelled/timed-out/undelivered question is replayed
as an error rather than a successful result. Other Hermes results remain opaque.

Deferred discovery remains Hermes-owned: deferred tools stay behind the genuine
`tool_search`/`tool_describe`/`tool_call` bridge until Hermes supplies their schema
to a generation. A translated target is withheld from that generation's top-level
catalog, not removed from Hermes' registry or rewritten inside discovery results.

## Call provenance and canonical replay

`encode_call_id` returns:

```text
ocn1.<wire-name>.<tag>.<original-id>
```

`<wire-name>` is the name the model actually emitted. `<tag>` is `v` when the
native arguments can be recovered by inverting the Hermes mapping (the `read`
or `write` inverse), otherwise `p<compressed-payload>` carrying the exact native arguments.
This keeps history unambiguous across a facade rebuild without a process-local
side table.

`decode_call_id` returns `None` for ids the provider did not create and raises
`SnapshotError` for a malformed payload. `snapshot_from_openai` uses it so a
replayed assistant call keeps the original native wire name and native
arguments (native `read` with `offset=0` replays as `read`, not as the
normalized `read_file` arguments).

## Opaque native metadata

The runtime receives per-part metadata from OpenCode's own events (`native` on
`session.reasoning.ended`, `session.text.ended` and `session.tool.called`).
`native_carrier` turns it into one `reasoning_details` entry:

```json
{"type": "opencode-free.native_assistant",
 "parts": [{"type": "reasoning", "index": 0, "native": {...}},
           {"type": "tool-call", "index": 2, "id": "call_x", "native": {...}}]}
```

- Only parts that actually carry upstream metadata are stored; absent metadata
  is never invented.
- `index` is the part's position in the native reasoning/text/tool-call order, so
  sparse metadata does not shift onto a later part on replay. Media parts do not
  participate in that order.
- The provider declares `native_reasoning_details_type` so Hermes preserves the
  carrier across turns for this provider only.
- `snapshot_from_openai` restores `native` onto the matching part by
  `(type, position)`. It never reconstructs metadata from visible text or
  reasoning.

## `snapshot_from_openai`

- turns `system` and `developer` content into ordered `system_parts`; OpenCode's
  own system prompt is preserved and these parts are appended, never
  substituted;
- accepts user text and image/file parts. Image data URLs must be base64;
  `http://`/`https://` image URLs are carried as URL assets; other image
  schemes refuse. Audio and other media kinds are not represented; callers must
  not rely on them being preserved;
- preserves assistant text, reasoning, opaque part metadata and tool calls,
  including exact call ids and JSON arguments;
- pairs each `tool` result with its assistant call. Duplicate calls, duplicate
  results and orphan results refuse before inference;
- represents error results with `is_error`, and otherwise carries text or JSON
  content;
- requires a non-empty user message as the native prompt anchor.

## Caller obligations

- Pass the actual enabled Hermes schemas and the current native registry names
  to `plan_tools`; do not advertise a tool Hermes has not enabled.
- Pass the same plan to `snapshot_from_openai` that produced the wire names for
  the request. The client does this automatically.
- Dispatch only the Hermes calls returned by `translate_call`. Never dispatch a
  native name that has no mapper, and never synthesize a tool result the host
  did not produce.
- Treat `native.unsupported`, `native.invalid_arguments` and
  `native.compatibility_unavailable` as refusals before effects; do not retry
  them with guessed arguments.
