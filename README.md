# hermes-oc-free-provider

A Hermes model-provider that calls OpenCode's free-model inference endpoint directly.
It does **not** use `opencode serve` or ACP.

## Disclaimer

This is an **unofficial, reverse-engineered compatibility project intended only for personal/internal use**. It is not affiliated with, endorsed by, or supported by OpenCode.

Use of OpenCode's hosted inference services remains subject to the **current [OpenCode Terms of Use](https://opencode.ai/legal/terms-of-service)** and any applicable model/provider-specific terms. As of September 20, 2026, those terms include restrictions relating to service access, automated use, usage limits/access restrictions, and use for the benefit of third parties. This project does not grant any rights beyond those terms.

Users are responsible for reviewing and complying with the current terms before using this plugin. **Do not use this project to resell or proxy OpenCode inference to third parties, evade quotas or rate limits, rotate identities/accounts to obtain additional free usage, or intentionally defeat access restrictions.**

Because this integration relies on an undocumented, reverse-engineered compatibility path, OpenCode may change or disable the behavior it depends on at any time.

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
