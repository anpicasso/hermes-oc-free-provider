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
Model discovery and inference use Python's standard-library HTTPS client, so the
same code path works on Linux, macOS, and Windows.

Select the provider and one of its discovered models:

```bash
hermes model
# Provider: opencode-free
```

## Model discovery

The plugin intersects OpenCode Zen's live `/v1/models` response with the
OpenCode catalog metadata published by [models.dev](https://models.dev/). It
exposes only live, tool-capable models whose published input/output/cache costs
are all zero. The resulting intersection—not the full upstream responses—is
cached for the gateway process and persisted under the active Hermes profile.
Each new gateway process queries both endpoints once; if refresh fails or finds
no qualifying model, the plugin uses the last successful intersection, then
the verified entries bundled with it when no cache exists. It selects
`/chat/completions` or `/responses` according to the known model transport.

Model availability, pricing, retention, and data-use rules belong to OpenCode
and the underlying provider and can change independently of this repository.

## Verify

```bash
python -m unittest discover -s tests -v
hermes plugins validate . --json
```

`hermes plugins doctor .` also verifies runtime discovery, import, and provider
registration.

## License

MIT.
