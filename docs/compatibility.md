# Compatibility and provenance

## Official binary

- Required: official OpenCode **>= 2.3.0**, unmodified, not a fork.
- Upstream: https://github.com/anomalyco/opencode
- Hermes never downloads, installs, upgrades or redistributes it.
- Resolution order: a usable `opencode` on the Hermes process `PATH`, then
  `plugins.entries.oc-free-provider.settings.binary_path` in the active
  profile's `config.yaml`. There is no environment-variable binary setting.
- Each candidate is validated with a bounded `<binary> --version` probe (argv
  list, `shell=False`, scrubbed environment). Onboarding stores only a validated
  absolute path.
- The runtime refuses a binary below 2.3.0 before spawning the server. A newer
  release is accepted but never assumed compatible: the loaded companion must
  still report companion id `hermes-oc-native`, protocol `4`, and the same
  binary version at handshake time.
- A version >= 2.3.0 is necessary, not proof that every later release preserves
  the required public hooks. Pin a reviewed release for production rollout.

The reference build used for this repository's evidence is OpenCode v2.3.0,
release source commit `a6b15d33daf6ab7d35569f6d711fdb630b9f04e7`, extracted
executable SHA-256
`2e0e4656521ca56e5ba16a00d48429958b8105b3fa6466710a4350e68759381f`. That hash
is evidence for the tested artifact, not a runtime requirement: the provider
deliberately does not impose an exact-artifact allowlist on an owner-managed
executable.

OpenCode is MIT licensed, copyright (c) 2025 opencode. See `OPENCODE-LICENSE`.

The official binary is paired with the bundled companion, which runs inside
`opencode serve` and uses public plugin hooks and RPC. The companion is not a
fork of OpenCode. Companion id, protocol and binary-version disagreement are
rejected before inference.

## Upstream request context

The official binary is unmodified, so it composes its own request context. That
can include a system preamble describing the working directory, git status and
OS details, and it follows the binary's own outbound HTTP policy. Using the
official binary is not an endorsement or permission from OpenCode for hosted
use and does not guarantee free-model availability; current OpenCode and
model-provider terms govern your use.

The provider runs the binary in an isolated workspace with a small environment
allowlist (`PATH`, locale, TLS certificate paths and platform loader variables)
plus a fresh per-process `OPENCODE_PASSWORD`. It refuses redirects on its own
loopback companion RPC. It does not intercept, rewrite or proxy the binary's
upstream network traffic.

## Platforms

Launch is an argv list with no shell. The runtime creates a new process group
(POSIX session leader; Windows new process group) and owns teardown.

- Linux: exercised by the offline suite (startup, ownership, cancellation,
  cleanup).
- macOS / Windows: the same portable startup/teardown code paths exist, but they
  are **not claimed verified** here. Verify startup, cancellation and cleanup on
  the target platform before relying on it.

There is no installer or architecture distribution matrix in this repository.

## Transport and cost boundary

The supported vendor response protocol is **Chat Completions SSE** using native
package `@opencode/ai/providers/openai-compatible`. Discovery keeps free models
with unsupported transports visible; generation rejects them with an explicit
transport reason. Names alone never select a protocol.

`opencode serve` is HTTP/JSON plus SSE (`GET /api/event`); its `--stdio` readiness
line is not JSON-RPC. The companion RPC is a separate interface. `/api/event`
cannot replace the raw completion gate: `session.step.streamed` carries no
finish/usage and `session.step.ended` arrives after tool settlement, which would
deadlock while Hermes holds the executors. Native events therefore own
text/reasoning/tool content and opaque metadata; the raw observer owns only the
pre-settlement completion verdict.

The `opencode-free` profile (alias `oc-free`, plugin ID `oc-free-provider`)
admits only enabled `opencode` models with explicit zero input, output,
cache-read and cache-write pricing in **every** cost tier. Empty/unknown pricing
is not free. Fresh native catalog validation precedes inference. There is no
paid fallback, no imported OpenCode login, and no legacy spoofing transport.

Provider authentication is not inferred from unrelated Hermes/environment
credentials. The free profile does not copy existing OpenCode credentials.
Broader authenticated providers require a separate explicit implementation and
selection; they cannot silently widen this profile. The runtime starts OpenCode
with a private environment allowlist and a fresh per-process
`OPENCODE_PASSWORD`; it does not inherit unrelated API keys or OpenCode
configuration.

## Implementation boundary

The provider uses stock Hermes profile/client APIs plus a few version-pinned
host helpers for auth and model onboarding (`plugins validate` reports those
private imports as a warning). There is no required Hermes
core patch and no host-side tool bridge. Model tool calls are returned as
ordinary Hermes calls and executed by the stock agent loop; the provider never
dispatches a Hermes tool itself. OpenCode executes no tool, including native
`read`. Native JavaScript `execute` is refused before an interpreter runs;
every native family except `read` is removed from the model-facing catalog.

## Evidence boundary

The combined offline suite passes against the installed official 2.3.0 binary
with an isolated `HERMES_HOME` and the deterministic loopback fixture model.
Fixture output is never described as a hosted response. The bounded hosted check
in this iteration covers one text turn and one genuine native read/tool-result
continuation for the exercised model only. Other hosted models, long-running
workloads, quotas and upstream account policy remain separate.

Hosted free-tier rate limits and upstream account/service policy remain
separate. Never describe local fixture output as a model response obtained from
a hosted service.
