// Test-only companion for the real agent-loop acceptance test.
//
// Reuses the packaged production companion unchanged, then binds the native
// `opencode` provider to a deterministic loopback fixture. The fixture URL is
// written next to this file by the test before the runtime starts; the child
// environment is scrubbed, so a file (not an env var) is the handoff.
import { Model, Plugin, Provider } from "@opencode/plugin"
import { readFileSync } from "node:fs"
import base from "../../../companion/index.ts"

const FIXTURE_PROVIDER = "opencode"
const FIXTURE_MODEL = "nonce"
const FIXTURE_PACKAGE = "@opencode/ai/providers/openai-compatible"

export default Plugin.define({
  id: "hermes-oc-agent-loop-test",
  async setup(ctx) {
    const fixtureURL = readFileSync(new URL("./fixture-url.txt", import.meta.url), "utf8").trim()
    if (!fixtureURL) throw new Error("agent-loop companion requires fixture-url.txt")

    const providerID = Provider.ID.make(FIXTURE_PROVIDER)
    const modelID = Model.ID.make(FIXTURE_MODEL)
    await ctx.provider.transform((editor) => {
      const existing = editor.get(FIXTURE_PROVIDER)
      if (!existing) throw new Error("agent-loop companion requires the native opencode provider")
      editor.update(FIXTURE_PROVIDER, (provider) => {
        provider.settings = { baseURL: fixtureURL, apiKey: "fixture-local" }
      })
      editor.models.set(FIXTURE_PROVIDER, [
        ...existing.models.values(),
        {
          ...Model.Info.default(providerID, modelID),
          name: "Deterministic nonce fixture (not a live model)",
          package: FIXTURE_PACKAGE,
          capabilities: { tools: true, input: ["text"], output: ["text"] },
          cost: [{ input: 0, output: 0, cache: { read: 0, write: 0 } }],
          limit: { context: 32_000, output: 4_096 },
        },
      ])
    })

    return await base.setup(ctx)
  },
})
