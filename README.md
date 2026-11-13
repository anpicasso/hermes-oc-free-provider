# oc-free-provider

An unofficial Hermes model-provider for OpenCode Zen free models. Hermes keeps
control of the agent loop and executes tools locally; the provider only handles
model discovery and inference transport.

It does **not** run `opencode serve`, ACP, or a tool-execution proxy.

## Compatibility and terms notice

This project is independent, unofficial, and not affiliated with or endorsed by
OpenCode. It is intended for a user's own internal use on systems they control.
It does not grant access to OpenCode Zen or any model, and it is not legal
advice.

OpenCode's [Zen documentation](https://opencode.ai/docs/zen/) describes Zen as
usable with other coding agents. That general interoperability statement is not
an authorization for every client, authentication method, workload, or use
case. This plugin's current public compatibility transport is not documented by
OpenCode as a supported Hermes integration and may be changed or disabled at
any time.

As of November 2026 the upstream contract explicitly excludes this transport:
OpenCode's Console documentation states that free-tier chat models are only
available from within OpenCode, and the gateway enforces that by answering
anything that does not look like the OpenCode client with HTTP 429
`FreeUsageLimitError`. The check is not a quota, a key, or a header set: it is
the request's client identity, down to the TLS stack.

This plugin gets around that check by reproducing the official client's request
— its sortable session id, its `<env>` system preamble, its `stream: true` tool
call shape, and its Bun TLS stack. That is deliberate client impersonation. It
is not a documented integration, it may stop working without notice, and whoever
runs it does so on their own judgement and responsibility. The supported paths
remain Hermes' built-in `opencode-zen` and `opencode-go` providers.

Before using the plugin, review and independently comply with the current:

- [OpenCode Terms of Service](https://opencode.ai/legal/terms-of-service)
- [OpenCode Zen documentation](https://opencode.ai/docs/zen/)
- terms, privacy notices, and acceptable-use rules for the selected model

In particular, use the service only when you have the right to access it. Do
not use this project to provide inference for third parties, resell or proxy the
service, scrape or bulk-extract outputs, evade quotas or rate limits, rotate
accounts or identities for additional free usage, conceal prohibited activity,
or bypass an access restriction. A disclaimer cannot make a prohibited use
compliant. If OpenCode requires an account, API key, paid access, or a supported
client for your intended use, do not use this compatibility transport instead.

Free-model labels describe pricing, not permission or privacy. Some free,
trial, and contributor models may use prompts or completions for model
improvement or may prohibit confidential data. Check the current Zen model
notice before every use and never send secrets or regulated data unless the
applicable terms expressly allow it.

## Bidirectional tool translation

Hermes remains the agent runtime: it applies its normal policy and approval
rules, executes tools locally, and returns each result to the model. The plugin
is a bidirectional adapter between OpenCode's tool vocabulary and Hermes':

1. **Hermes → OpenCode:** available native Hermes schemas are exposed under
   their OpenCode-compatible names. Their native duplicates are omitted, while
   unrelated Hermes tools remain available under their native names.
2. **OpenCode → Hermes:** model tool calls are translated back to native Hermes
   names and argument shapes before local dispatch.
3. **Hermes → OpenCode continuation:** recorded native calls are translated
   back to OpenCode names and arguments when conversation history is replayed,
   keeping the model on one consistent vocabulary. Tool results retain their
   call IDs and payloads. Explicit tool choices are translated the same way.

The bidirectional pairs are:

- `bash` ↔ `terminal`
- `edit` ↔ `patch`
- `glob` / `grep` ↔ `search_files`
- `read` ↔ `read_file`
- `skill` ↔ `skill_view`
- `task` ↔ `delegate_task`
- `todowrite` ↔ `todo_list`
- `webfetch` ↔ `web_extract`
- `websearch` ↔ `web_search`
- `write` ↔ `write_file`

An alias is advertised only when its target tool was supplied by Hermes. If the
model calls an unavailable compatibility alias, the request fails
closed—nothing is executed and no success is fabricated. Streamed tool
arguments are buffered until their JSON is complete before translation.

The adapters preserve executable intent, not every OpenCode runtime option.
`bash.timeout` is converted from milliseconds to seconds; todo order is
preserved but OpenCode's priority label is discarded; OpenCode `read` directory
listing and image-attachment behavior is not emulated; `webfetch`
formatting/timeout hints and advanced `websearch` crawl hints have no Hermes
equivalent; and OpenCode task metadata is passed to the Hermes subagent as
context while Hermes controls scheduling.

## Install

```bash
hermes plugins install oc-free-provider --enable
```

The plugin does not install or run the OpenCode CLI, Node.js, npm, or `npx`.

**It does need [Bun](https://bun.sh).** Since OpenCode's October 2026 gateway
change, anonymous free-tier requests are only answered when they leave from
Bun's TLS stack: the same headers and body from Python, curl, Node, or even
curl-impersonate get `429 FreeUsageLimitError`. Inference is therefore posted
through `oc_transport.js` (`bun oc_transport.js`); model discovery still uses
Python's standard-library HTTPS client. Install Bun once:

```bash
curl -fsSL https://bun.sh/install | bash
```

The plugin finds `bun` on `PATH`, or at `~/.bun/bin/bun`, and `OC_FREE_BUN`
overrides both. Without Bun every request fails immediately with that
instruction instead of a misleading 429.

Select the provider and one of its discovered models:

```bash
hermes model
# Provider: opencode-free
```

## What the gateway now requires

Each free-tier request carries the identity the gateway checks for:

- `ses_` session id in OpenCode's *descending* sortable form (12 hex chars of an
  inverted millisecond timestamp, then 14 base62), sent as `x-opencode-session`
  and repeated in the prompt;
- a system message containing OpenCode's `<env>` block with
  `Current conversation session ID`, `Working directory`, and
  `Workspace root folder`;
- `stream: true` and a non-empty tool list;
- `User-Agent: opencode/latest/<version>/cli` and `Authorization: Bearer public`;
- the Bun transport described above.

The endpoint is `https://opencode.ai/inference/openai/v1`; the older
`/zen/v1` paths no longer serve free models.

## Model discovery

OpenCode's live `/inference/v1/models` response is the authoritative catalog:
every advertised model whose id ends in `-free` is exposed, even when
[models.dev](https://models.dev/) has not published it yet. models.dev metadata
is consulted when available to exclude deprecated models, drop models without
tool calling, and include zero-cost models that do not carry the `-free` suffix
(such as `big-pickle`). Ids the relay still lists but no longer serves are
excluded explicitly. The current free catalog is served over
`/chat/completions`.

The catalog is re-read while the process runs: the relay's list is refreshed at
most every 10 minutes, models.dev metadata every 6 hours (it is a
multi-megabyte payload). A gateway process that lived for days would otherwise
keep serving the catalog it read at start-up. The discovered list is persisted
under the active Hermes profile and used when a refresh fails; that persisted
copy is dropped after a week. The bundled list is the last resort, used before
the first successful discovery or while the relay is unreachable.

Hermes shows a provider's curated floor (`fallback_models`) merged with its
live list, so the plugin pins the floor to the last discovered catalog: a
static floor would keep retired ids in the picker and hide newly launched free
models. The picker therefore lists exactly the free ids the relay currently
serves.

Model availability, pricing, retention, and data-use rules belong to OpenCode
and the underlying provider and can change independently of this repository.

## Verify

```bash
python -m unittest discover -s tests -v
hermes plugins validate . --json
python -c "import hermes_oc_client as c; print(c.OpenCodeClient().chat.completions.create(model='big-pickle', messages=[{'role':'user','content':'say PONG'}], tools=[{'type':'function','function':{'name':'read_file','parameters':{'type':'object','properties':{}}}}]).choices[0].message.content)"
```

`hermes plugins doctor .` also verifies runtime discovery, import, and provider
registration.

## License

MIT.
