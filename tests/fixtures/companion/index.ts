// Test-only companion. Production `companion/index.ts` must never read a
// fixture URL or register a provider; tests load this wrapper instead by
// setting HERMES_OPENCODE_COMPANION to this directory.
//
// The wrapper reuses the production plugin unchanged, then binds the real
// `opencode` provider to the deterministic loopback fixture and adds one
// zero-price model (`nonce`). That keeps the strict production catalog policy
// (`oc_catalog.select_model`) active in tests: no test-only selector bypass.
import { Model, Plugin, Provider } from "@opencode/plugin"
import base from "../../../companion/index.ts"

const FIXTURE_PROVIDER = "opencode"
const FIXTURE_MODEL = "nonce"
const FIXTURE_PACKAGE = "@opencode/ai/providers/openai-compatible"

export default Plugin.define({
  id: "hermes-oc-native-test",
  async setup(ctx) {
    const fixtureURL = typeof ctx.options.fixtureURL === "string" ? ctx.options.fixtureURL : undefined
    if (!fixtureURL) throw new Error("test companion requires a loopback fixtureURL option")

    const providerID = Provider.ID.make(FIXTURE_PROVIDER)
    const modelID = Model.ID.make(FIXTURE_MODEL)
    await ctx.provider.transform((editor) => {
      const existing = editor.get(FIXTURE_PROVIDER)
      if (!existing) throw new Error("test companion requires the native opencode provider")
      editor.update(FIXTURE_PROVIDER, (provider) => {
        provider.settings = { baseURL: fixtureURL, apiKey: "fixture-local" }
      })
      editor.models.set(FIXTURE_PROVIDER, [
        ...existing.models.values(),
        {
          ...Model.Info.default(providerID, modelID),
          name: "Deterministic nonce fixture (not a live model)",
          capabilities: { tools: true, input: ["text"], output: ["text"] },
          cost: [{ input: 0, output: 0, cache: { read: 0, write: 0 } }],
          limit: { context: 32_000, output: 4_096 },
        },
      ])
    })

    return await base.setup(ctx)
  },
})
