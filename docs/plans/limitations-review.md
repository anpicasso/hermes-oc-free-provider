# Limitations and acceptance plan

Status: 4.1.0. This is the current product boundary and acceptance checklist,
not a deployment or release record.

## Goal

Expose one OpenCode-native compatibility tool (`read` → Hermes `read_file`)
plus every genuine tool the current Hermes session enables. The official
OpenCode binary and stock Hermes stay unmodified. Hermes retains instructions,
tool availability, approvals, execution and result ownership.

## Implemented

1. **Read-only native catalog.** `read` is the only retained native descriptor.
   Every other native family (write, edit, patch, shell, glob/grep, skill,
   webfetch/websearch, question, subagent, todowrite, `execute`, browser,
   `opencode.*`, `console_watch.*`) is removed by native identity before the
   request. Genuine Hermes tools are advertised with their full schemas; a
   collision gets the reversible `hermes_<name>` wire alias.
2. **One mapper.** `oc_translate` implements only the bounded `read` adapter:
   `path` required, `offset` 0/1 = line 1, positive limit capped at 2000,
   negative offset refused. Full Hermes `read_file` remains independently
   available. Planning fails before inference when `read_file` is unavailable.
3. **No native execution.** OpenCode executes no tool, including `read`.
   Unsolicited native calls are refused with no side effect; `decide` never
   authorizes JavaScript. Hermes `execute_code`, when enabled, is an ordinary
   additive tool.
4. **Strict completion contract.** A successful completion or executable tool
   release requires terminal `[DONE]`, a supported consistent `finish_reason`,
   and valid input/output token counts (zero valid). Missing, malformed or
   conflicting required fields, stream errors and truncation refuse with no
   tool execution and no weaker retry. Valid optional cache/reasoning
   breakdowns and opaque native metadata are preserved.
5. **Binary onboarding.** Required official `opencode` >= 2.3.0. Resolution is
   the Hermes process PATH, then
   `plugins.entries.oc-free-provider.settings.binary_path`. `hermes auth add
   opencode-free` and `hermes model` share one validating onboarding helper.
   There is no environment-variable binary setting and no download path.
6. **Portable process ownership.** Launch is an argv list with `shell=False`.
   POSIX creates a session leader; Windows creates a new process group.
   Teardown owns the process tree. `/proc` is used only when present.
7. **Free-only scope.** Discovery admits only enabled `opencode` models with
   explicit zero pricing in every cost tier. No paid fallback, imported
   OpenCode login, inferred credential or widened profile.

## Deferred

- **Streaming and generation controls.** `stream=True`, non-`auto`
  `tool_choice`, `temperature`, `top_p`, token limits, `stop`,
  `response_format` and `reasoning_effort` are refused at create time.
  Buffered-only release is the supported behavior. True incremental streaming
  and control parity are tracked in
  [issue #11](https://github.com/anpicasso/hermes-oc-free-provider/issues/11).
  Unsupported non-default values are never silently ignored.
- **Auxiliary model.** `resolve_aux_model` returns `""` because auxiliary
  requests send controls this transport cannot map yet. No cheap model is
  advertised while those controls would be rejected.
- **Non-Linux platforms.** macOS and Windows share the portable startup and
  teardown code, but are not claimed verified until startup, cancellation and
  cleanup are exercised there.

## Acceptance gates

- Full offline suite with the core venv interpreter, the core checkout on
  `PYTHONPATH`, and an isolated `HERMES_HOME` set before imports: 99 tests.
- Plugin validation and doctor: manifest, import, registration and security
  scan clean.
- Bounded hosted check with stock `AIAgent` and this source tree: native
  `read` translated to Hermes `read_file`, token recovered, usage measured,
  `finish=stop`.
- Fixture output is never reported as hosted inference. A local contract error
  is never reported as an upstream HTTP status.
