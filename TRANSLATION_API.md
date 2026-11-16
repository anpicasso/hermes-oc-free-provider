# Native translation API

Status: contract for `oc_translate.py` in 4.1.0. The module performs no tool
effects. It validates names and arguments, maps the single native compatibility
call to an ordinary Hermes call, preserves opaque native metadata, and replays
canonical history into the native snapshot. Hermes dispatches translated calls
in its own agent loop.

## Public surface

```python
plan_tools(tools, native_names=NATIVE_KEEP) -> WirePlan
translate_call(native_name, arguments, *, plan=None, enabled=None) -> (host_name, host_arguments)
encode_call_id(native_name, native_arguments, host_arguments, original_id) -> str
decode_call_id(call_id) -> (wire_name, native_arguments | None, original_id) | None
native_carrier(native_parts) -> dict | None
snapshot_from_openai(messages, *, plan=None) -> Snapshot
```

`NATIVE_KEEP = {"read"}` and `COMPAT_TOOL = "read_file"`. `WirePlan` has
`register` (additive tool definitions for `open`), `host_to_wire` and
`wire_to_host`. `Snapshot` has `system_parts`, `messages` and `user_text`.

Errors are `NativeTranslationError` subclasses with `.code`:
`native.unsupported`, `native.invalid_arguments`,
`native.compatibility_unavailable` and `snapshot.invalid`.

## Tool planning

`plan_tools` receives the enabled Hermes definitions (flat
`{name, description, parameters}` or OpenAI
`{"type":"function","function":…}`) and the reserved native registry names.

- Every Hermes definition is registered with its own description and parameter
  schema, byte-for-byte. There is no hard-coded subset and no reduced-option
  substitute.
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

Only one native tool is translated. Every other native name is removed from the
model-facing catalog by the companion; if a model emits one anyway, no mapper
exists and no Hermes tool runs.

| native | Hermes | accepted native options | refusals / notes |
|---|---|---|---|
| `read` | `read_file` | `path`, `offset`, `limit` | unknown options; empty/missing `path`; `offset < 0` (tail read) has no Hermes equivalent; `offset` 0 and 1 both mean line 1; `limit` must be at least 1 and is capped at 2000 |
| `write`, `edit`, `patch`, `shell`, `glob`, `grep`, `skill`, `webfetch`, `websearch`, `question`, `subagent`, `todowrite`, `execute`, `browser*`, `browser.*`, `opencode.*`, `console_watch.*` | — | none | no mapper. The name is returned unchanged (or `oc_unsupported_<name>` when it collides with an enabled Hermes tool) so stock dispatch cannot run an unrelated tool |
| `hermes_<name>` registered by `plan_tools` | `<name>` | as advertised | identity forwarding of the exact Hermes wire name and arguments |

`translate_call` has one additive path: when `plan` maps a wire name to a
Hermes tool registered in this request (for example `terminal` or
`hermes_patch`), the call keeps that tool's identity and its arguments
byte-for-byte. It does not add the tool; `plan_tools` only saw schemas Hermes
supplied.

The compatibility `read` adapter is a deliberately bounded mapping, not a
promise of every native `read` behavior. Full Hermes `read_file` remains
available independently with its own schema and semantics.

If the native `read` target is disabled in the current request,
`translate_call` returns the sentinel `oc_unsupported_read`; planning already
refuses that request upstream, so this is defence in depth. The sentinel is not
a supported tool and has no effect.

## Call provenance and canonical replay

`encode_call_id` returns:

```text
ocn1.<wire-name>.<tag>.<original-id>
```

`<wire-name>` is the name the model actually emitted. `<tag>` is `v` when the
native arguments can be recovered by inverting the Hermes mapping (the `read`
inverse), otherwise `p<compressed-payload>` carrying the exact native arguments.
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
