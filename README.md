# hermes-oc-free-provider

A Hermes model-provider that calls OpenCode's free-model inference endpoint directly.
It does **not** use `opencode serve` or ACP.

## Tool ownership

Hermes owns the agent loop and executes every browser, terminal, file, MCP, and
other tool locally. The plugin forwards Hermes' native tool schemas, parses the
model's structured tool calls, and returns them to Hermes for execution.

This is an independently implemented, reverse-engineered compatibility path,
not an official OpenCode API contract. It was verified against OpenCode 1.18.31
with real text, multi-turn, and native tool-call/result round trips. OpenCode can
change its free-tier request gate without notice.

## Install

```bash
hermes plugins install anpicasso/hermes-oc-free-provider --enable
```

The plugin uses `npx --yes opencode-ai` for model discovery and the
client-version header. Inference itself is a direct HTTPS request.

Select the provider and one of its discovered free models:

```bash
hermes model
# Provider: opencode-free
```

## Model discovery

The plugin runs `opencode models opencode --pure`, exposing every model in the
local OpenCode `opencode` provider. If discovery fails, it falls back to the
last verified free-model list. It uses `/chat/completions` for OpenAI-compatible
models and `/responses` for the two Muse contributor models, matching the local
OpenCode catalog's provider metadata.

The `muse-spark-*-contributor-free` models may train on prompts and completions.
Hermes displays its contributor-tier warning and requires the normal explicit
acknowledgment before those models can run unattended.

## Safety

OpenCode's free endpoint currently requires baseline tool descriptors. The
plugin sends inert compatibility descriptors, then appends the real Hermes tool
schemas. If a model attempts a compatibility-only tool Hermes did not offer,
the request fails closed instead of executing or pretending it succeeded.

## Verify

```bash
python -m unittest discover -s tests -v
hermes plugins validate . --json
```

Hermes 0.21.3's `plugins doctor` incorrectly routes provider-only manifests
through the standalone-plugin loader and reports a missing `register()` hook.
The plugin intentionally registers only its model provider at import time.

## License

MIT.
