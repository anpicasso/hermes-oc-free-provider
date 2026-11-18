# oc-free-provider

Native [OpenCode](https://github.com/anomalyco/opencode) inference through
Hermes's public model-provider API. The provider owns a private official
OpenCode process and a bundled companion that runs inside it. Hermes keeps
canonical instructions and history, tool availability, approvals and tool
effects. The binary is unmodified: there is no fork, proxy service, core patch,
or provider-side tool execution.

**Status: 4.2.0 source. The full offline suite (126 tests) passes against the
installed official OpenCode 2.3.0 binary with the core venv interpreter, an
isolated `HERMES_HOME` and the core checkout on `PYTHONPATH`. Bounded hosted
acceptance with the same binary passed (text plus a native `read`/Hermes
`read_file` continuation). Real Hermes CLI acceptance also confirmed conversation
continuity and the same native id after gateway restart/resume, plus distinct
native ids for two delegated siblings. Cache hits were observed, not a proven
causal performance gain.**

Scoped conversations retain one official native OpenCode session and a durable
private workspace per active profile/scope/model lane. The exact
official-generated id is stored in a profile-scoped plugin-data SQLite mapping;
it is never synthesised or reused as a Hermes transcript. Each private runtime
admits one live companion session at a time, and each request marker pins the
Hermes profile that built it: resolving it under another profile is refused
before any runtime or affinity-storage access. Hermes remains the
only history authority: every generation replaces the native snapshot, and the
companion continues to refuse native tool execution. Missing native state
allocates a fresh official id and reports an explicit affinity reset; transport
errors never count as absence. Idle process eviction keeps the mapping and
native data, while age-based retention removes inactive scopes under the same
cross-process lock. Delegate siblings share a `sub_<sha>` cache scope, so their
physical session id is part of the lane (compression resets delegate affinity);
`/btw` and same-model review share the parent's lane and serialize. A per-scope
ownership lock is held for the native runtime lifetime and released only after
the server stops.

## Compatibility and terms

This project is independent, unofficial, and not affiliated with or endorsed by
OpenCode. It does not grant access to OpenCode Zen or any model, and it is not
legal advice. It also does not claim that OpenCode approves or documents this
integration.

The provider launches the official, unmodified `opencode` executable and loads
a companion into that process through public hooks. It does not patch the
binary, reproduce its TLS stack, or run a separate inference proxy. Using the
official binary is not an endorsement or permission from OpenCode for hosted
use, and it does not guarantee free-model availability. The current OpenCode
and model-provider terms govern your use.

The unmodified binary composes its own request context. That can include a
system preamble describing the working directory, git status and OS details,
and it follows the binary's own outbound HTTP policy. The provider runs the
binary in an isolated workspace with a small environment allowlist
(`PATH`, locale, TLS certificate paths and platform loader variables) plus a
fresh per-process `OPENCODE_PASSWORD`; it refuses redirects on its own loopback
RPC. It does not intercept, rewrite or proxy the binary's upstream traffic.

Before using the plugin, review and independently comply with the current
[OpenCode Terms of Service](https://opencode.ai/legal/terms-of-service), the
[OpenCode Zen documentation](https://opencode.ai/docs/zen/), and the terms,
privacy notices and acceptable-use rules for the selected model. Use the
service only when you have the right to access it. Do not use this project to
provide inference for third parties, resell or proxy the service, scrape or
bulk-extract outputs, evade quotas or rate limits, rotate accounts or identities
for additional free usage, conceal prohibited activity, or bypass an access
restriction. Free-model labels describe pricing, not permission or privacy;
never send secrets or regulated data unless the applicable terms expressly
allow it.

## Identity

| Item | Value |
|---|---|
| Plugin | `oc-free-provider` (`model-provider`) |
| Provider | `opencode-free` (alias `oc-free`) |
| Provider module | `__init__.py` → `opencode_free` |
| Vendor protocol | Chat Completions SSE inside the native OpenCode runtime |
| Companion protocol | 5 |
| Model-facing native tools | `read` only |
| Hermes-facing streaming | none (`supports_streaming = False`; deferred, [issue #11](https://github.com/anpicasso/hermes-oc-free-provider/issues/11)) |
| Cost policy | free models only |

## Requirements

- **Official `opencode` >= 2.3.0.** Hermes never downloads, installs or
  upgrades it. Resolution order: a usable `opencode` on the Hermes process
  `PATH`, then
  `plugins.entries.oc-free-provider.settings.binary_path` in the active
  profile's `config.yaml`. Core's resolved executable is used only as a
  validated fallback hint, never ahead of PATH or config. There is **no
  environment-variable binary setting**.
  Shell aliases/functions and an interactive shell's `PATH` are not the gateway
  process environment.
- **Stock Hermes.** The plugin manifest declares Hermes `>=0.23.1`. The
  provider uses the public profile and client APIs plus a few version-pinned
  host helpers for auth and model onboarding; `plugins validate` reports those
  private imports as a warning. No Hermes core patch is required.
- **Python interpreter that runs Hermes.** No extra Python packages are
  declared. The bundled TypeScript companion uses modules supplied by the
  official OpenCode host, so there is no separate npm/Bun install step.
- **Execution compatibility.** Launch is an argv list with no shell, and the
  runtime owns its process group/session portably. Linux is exercised by the
  suite; macOS and Windows have portable startup/teardown code but are **not
  claimed verified** until startup, cancellation and cleanup are tested there.

### Configure the binary

```bash
# Preferred: a usable opencode on the Hermes process PATH.
opencode --version    # must print 2.3.0 or newer

# Otherwise save an absolute path through the native onboarding surfaces:
hermes auth add opencode-free   # interactive: validates, then saves
hermes model                    # same shared onboarding, then model picker
```

The saved value is a validated absolute path, written through Hermes'
plugin-settings writer (the same one `ctx.set_config` uses). A validated path is
never replaced by a shell command or arguments. Onboarding leaves the
configuration unchanged when cancelled or given an invalid path.

## Install into an isolated home

Do not modify the active profile for evaluation. `HERMES_HOME` selects an
isolated profile; the plugin lives under `$HERMES_HOME/plugins/`. Registration
and import start no OpenCode process; the provider starts its private runtime
lazily on the first catalog or model request.

Install a reviewed 40-character commit SHA from this public repository, not a
moving branch:

```bash
export HERMES_HOME="$(mktemp -d "$HOME/.cache/hermes-native-evaluation/home.XXXXXX")"
hermes plugins install anpicasso/hermes-oc-free-provider --ref "$VERIFIED_COMMIT_SHA" --enable
hermes plugins doctor oc-free-provider --ci
```

`plugins doctor` must report the registration `provider(s): opencode-free`.
Then either put a usable `opencode` on the process `PATH`, or run
`hermes auth add opencode-free` inside that home and save an absolute path.
Select the provider and a discovered free model through the normal picker
(`hermes model`) or an invocation override:

```bash
HERMES_HOME="$HERMES_HOME" hermes --provider oc-free -m <free-model-id>
```

The catalog is read live from the resolved binary before inference. Only enabled
`opencode` models whose input, output, cache-read and cache-write prices are
explicitly zero in **every** tier are admitted. Empty or unknown pricing is not
free. There is no paid fallback and no legacy spoofing transport.

## How a request works

1. The client resolves and validates the binary, then starts one private
   `opencode serve --stdio --port 0 --hostname 127.0.0.1` process per runtime
   owner. The child inherits only a small environment allowlist and gets a fresh
   random `OPENCODE_PASSWORD`.
2. The bundled companion is loaded from the private `opencode.json` and runs
   inside that official server. It exposes authenticated JSON RPC over loopback
   and observes each primary Chat Completions SSE response non-destructively.
3. The model-facing catalog is **native `read` plus every genuine Hermes tool
   the session exposes**, with the host's full descriptions and schemas. Every
   other native descriptor (edit, write, patch, shell, grep/glob, subagent,
   webfetch/websearch, skill, question, `execute`, browser/opencode/console
   families) is removed before the request. OpenCode executes no tool.
4. A model-emitted native `read` is translated to an ordinary Hermes
   `read_file` call. Hermes dispatches it through its normal loop, approvals and
   policy. On the next completion the canonical Hermes history is replayed as
   the native snapshot with the original wire names and arguments.
5. Text, reasoning and tool names/arguments/call IDs come from OpenCode's own
   events. The raw SSE observer only decides whether a completion is complete
   and safe to release.

## Supported surface

One native compatibility tool: `read` → Hermes `read_file`.

- `path` is required; native `offset` 0 or 1 both mean line 1; positive
  `offset`/`limit` map to `read_file` (limit capped at 2000).
- Negative `offset` (tail read) has no Hermes equivalent and is refused rather
  than silently changed.
- Full Hermes `read_file` is still advertised independently with its own schema
  and semantics; the compatibility adapter is not a promise of every native
  `read` behavior.

Every other capability (write, edit/patch, shell, search, web, delegation,
skills, todos, browser, Python execution) comes from the genuine Hermes tools
enabled in the session. They are advertised byte-for-byte. A Hermes tool whose
name collides with a native registry id (for example `patch`) is advertised
under the reversible wire alias `hermes_<name>` and never dropped.

If `read_file` is unavailable, the provider fails before inference instead of
advertising a compatibility read that cannot run.

Option restrictions and refusals are in [`TRANSLATION_API.md`](TRANSLATION_API.md).

## Completion contract

Before a successful completion is returned, or any executable tool call is
released to Hermes, the primary Chat Completions response must provide:

- terminal `[DONE]`;
- a supported, consistent `finish_reason` (`stop`, `tool_calls`, `length`);
- valid `prompt_tokens` and `completion_tokens` (zero is valid; missing,
  negative, non-integer or inconsistent totals are not).

Missing/malformed fields, an explicit stream error, truncation or unsupported
framing produce a clear provider error naming the model and the failed
contract, with **zero** tool executions and no weaker retry. Optional cache /
reasoning-token breakdowns and opaque native metadata are preserved when
supplied and never invented. A failed response is not evidence to remove a
model from discovery. When the official runtime reports an upstream HTTP error
status (for example a 429 free-usage limit), the raised `CompletionRefused`
carries it as `.status_code` for Hermes' retry/fallback classifier; a refusal
with no reported status carries `None`, never a status guessed from text.

## Limitations

- **Native JavaScript `execute` is unsupported and removed from the catalog.**
  If a model emits it anyway, the companion hard-refuses it before any
  interpreter runs and `decide` never authorizes a program. Hermes's Python
  `execute_code` is an ordinary additive tool when the session enables it.
- **Other native tools are removed, not translated.** Unsolicited native calls
  (write, edit, shell, subagent, ...) are refused with no side effect; Hermes
  never receives a tool it did not advertise.
- **No streaming and no generation controls yet.** `stream=True`,
  non-`auto` `tool_choice`, `temperature`, `top_p`, token limits, `stop`,
  `response_format` and `reasoning_effort` are refused at create time. True
  incremental streaming and control parity are deferred together in
  [issue #11](https://github.com/anpicasso/hermes-oc-free-provider/issues/11);
  unsupported non-default values are never silently ignored.
- **No auxiliary model advertised.** Auxiliary requests send controls this
  transport cannot map yet, so `resolve_aux_model` returns `""` and an auxiliary
  attempt fails loudly instead of pretending those controls were honored.
- **Free models only.** Broader authenticated providers require separate
  explicit work; this profile cannot silently widen. It does not infer
  authentication from unrelated Hermes/environment credentials or copy an
  existing OpenCode login.
- **Chat Completions SSE only.** A catalog entry whose native package is not
  `@opencode/ai/providers/openai-compatible` stays discoverable with a reason
  but refuses inference; no speculative protocol adapters exist.
- **Linux verified only.** Startup, cancellation and cleanup tests exist for
  Linux; other platforms share the portable argv/process-group code but are not
  claimed as verified.

## Evidence boundary

Offline tests run against the installed official 2.3.0 binary with the
deterministic loopback fixture model. Fixture output is never described as a
hosted response. The recorded offline command (core venv interpreter,
`PYTHONPATH=<core checkout>`, isolated `HERMES_HOME` set before Python starts):

```bash
PYTHONPATH=/path/to/hermes-core-checkout \
  /path/to/hermes-core/venv/bin/python -m unittest discover -s tests -t . -q
# Ran 99 tests ... OK
```

Hosted evidence for this iteration is bounded to one stock `AIAgent` run with
this source tree: native `read` wire name (`ocn1.read.…`) translated to
Hermes `read_file`, token recovered, measured usage, `finish=stop`.

The hosted checks are bounded: they do not prove every discovered free model,
long-running workloads or tool execution on models that only returned text.
Hosted quotas, rate limits and upstream account/service policy remain separate.

## Tests

Run from the repository root. Tests force a temporary `HERMES_HOME` **before**
Python imports so discovery cannot load an installed copy. They resolve the
official binary from the test-only `HERMES_OPENCODE_BINARY` variable or the
process `PATH`; the provider itself never reads that variable.

```bash
export TMPDIR=/path/to/scratch
export HERMES_OPENCODE_BINARY=/path/to/opencode
# Set an isolated home BEFORE Python starts; use the interpreter that matches
# the dependency environment Hermes activates at import (`hermes_bootstrap`);
# a different minor version loads incompatible compiled wheels, and features
# behind them (for example the deferred-tool `tool_call` bridge) silently drop.
export HERMES_HOME="$(mktemp -d "$TMPDIR/oc-test-home.XXXXXX")"
export PYTHONPATH=/path/to/hermes-core-checkout
/path/to/hermes-runtime-python -m unittest discover -s tests -t . -v
```

Translation-only tests need no binary:

```bash
python -m unittest discover -s tests -t . -p 'test_translation*.py' -v
```

## Documentation

- [`docs/compatibility.md`](docs/compatibility.md) — binary provenance,
  transport and cost boundary.
- [`TRANSLATION_API.md`](TRANSLATION_API.md) — native/Hermes mapping, accepted
  options and refusal rules.
- [`RUNTIME_API.md`](RUNTIME_API.md) — runtime/client lifecycle, loopback RPC,
  authentication and request controls.
- [`COMPANION_API.md`](COMPANION_API.md) — companion protocol, hooks and event
  stream.
- [`CHANGELOG.md`](CHANGELOG.md) — source changes and known limitations.
- [`LICENSE`](LICENSE), [`docs/OPENCODE-LICENSE`](docs/OPENCODE-LICENSE).
