// Hermes OpenCode-native companion.
//
// Runs inside the official OpenCode v2.3.0 server in a private loopback
// runtime. Two public, read-only surfaces are used for the stock
// `@opencode/ai/providers/openai-compatible` (Chat Completions) route:
//
//  * `session.hook("http.response")` observes the SSE body non-destructively
//    (pass-through TransformStream) and derives a *bounded response outcome*:
//    terminal [DONE], finish_reason, usage, error event, truncation, invalid
//    tool-call framing. It never edits the request, headers or auth.
//  * `session.hook("context")` binds the response to exactly one authorized
//    generation per advance and installs the caller's canonical snapshot.
//
// OpenCode's own parsed tool calls remain authoritative. The gate only decides
// whether they may be exposed to Hermes; executors are held from the moment
// the runner calls them. Only the native descriptors Python keeps for the
// session (translated to enabled Hermes tools) survive in the model-facing
// catalog; every other native tool — including JavaScript `execute` — is
// removed before the request and refused at dispatch if a model emits it.
//
// This file is the packaged production companion: no fixture provider, no
// `fixtureURL` option and no parser RPC. Fixture registration lives in
// `tests/fixtures/companion/` only.
//
// Contract: COMPANION_API.md. ponytail: one file, no package.json. The host
// provides @opencode/plugin.
import { Plugin, Rpc } from "@opencode/plugin"
import { Schema } from "effect"
import { createHash } from "node:crypto"

const RPC_ID = "hermes-oc-native"
const COMPANION = "hermes-oc-native"
const PROTOCOL = 5
const EVENT_LOG_LIMIT = 512
// Native descriptors Python may keep in the model-facing catalog (each mapped
// to an enabled Hermes tool by oc_translate.NATIVE_MAP). Python names the exact
// subset per generation (`nativeTools`); everything else is removed by
// identity, and a native name a Hermes registration owns stays Hermes'.
const TRANSLATED_NATIVE = new Set(["read", "write", "glob", "grep", "question", "todowrite"])

// Bounds for the response parser. Everything else is counters only, so a
// hostile or huge stream cannot grow plugin memory without limit.
const MAX_LINE_BYTES = 256 * 1024
const MAX_CALLS = 64
const MAX_CALL_ARG_BYTES = 1024 * 1024
const MAX_EVENTS = 100_000
// RPC/lifecycle bounds. A refusal is always explicit; no queue grows forever.
const MAX_SNAPSHOT_MESSAGES = 512
const MAX_SNAPSHOT_BYTES = 4 * 1024 * 1024
const MAX_SYSTEM_BYTES = 1024 * 1024
const MAX_HERMES_TOOLS = 256
const MAX_PENDING = 64
const MAX_RESULTS = 64
const MAX_NESTED_CALLS = 32
// The companion owns exactly one live session per official runtime. The tool
// registry is process-wide, so a second live session would advertise another
// caller's Hermes schemas on its next request.
const MAX_LIVE_SESSIONS = 1
const CLOSED_TOMBSTONES = 128
// One message for every JavaScript request: the model gets an actionable
// refusal and Hermes never receives JS as a Python program.
const EXECUTE_UNSUPPORTED =
  "native JavaScript execute is unsupported; use the enabled Hermes Python execute_code tool"
// Any native tool outside the session's `nativeTools` is never advertised. A model that
// emits one anyway gets an explicit refusal, never a host side effect.
const nativeRefusal = (name: string) =>
  name === "execute"
    ? EXECUTE_UNSUPPORTED
    : `native tool ${name} is not exposed by this provider; call the enabled Hermes tool instead`

// Supported Chat Completions terminal reasons. `length` is a terminal stop;
// executable tool calls are still gated separately.
const SUPPORTED_FINISH = new Set(["stop", "tool_calls", "length"])


const FORWARD = Symbol("hermes.oc.native.forward")

type Outcome = {
  generation: number | null
  requestSeq: number | null
  status: number
  ok: boolean
  done: boolean
  finish?: string
  usage?: unknown
  error?: string
  calls: number
  textChars: number
  bytes: number
  events: number
}

type ToolResult = { content: string; output?: unknown }
type PendingCall = {
  callID: string
  name: string
  input: unknown
  resolve: (result: ToolResult) => void
  reject: (error: Error) => void
  detach: () => void
}

type Session = {
  id: string
  generation: number
  authorizations: number[]
  running: number | undefined
  userText: string | undefined
  started: boolean
  // A prompt was admitted and its run has not reached idle yet. `finish`
  // interrupts/waits for this before the next canonical generation.
  promptRunning: boolean
  // The previous generation reached an explicit boundary. A retained session
  // accepts the next snapshot only after `finish` (or before its first one).
  settled: boolean
  closed: boolean
  snapshot: Array<Record<string, unknown>>
  systemParts: string[]
  pending: Map<string, PendingCall>
  calls: Map<string, Array<{ callID: string; name: string; input: unknown; native?: unknown }>>
  names: Map<string, string>
  // Native descriptors kept for this session; undefined = legacy default
  // (`read` while Hermes `read_file` is registered).
  nativeTools?: Set<string>
  stepGeneration: Map<string, number>
  events: Array<Record<string, any>>
  seq: number
  waiters: Set<() => void>
  promptError?: string
  rejectReason?: string
  activeRequest?: { generation: number; requestSeq: number }
  outcome?: Outcome
  primaryAssistant?: string
  // Execute lifecycle: a parked approval, a parked program result, and the
  // nested `tools.*` calls the JS program makes through the same executors.
  gates: Map<string, { resolve: () => void; reject: (error: Error) => void }>
  after: Map<string, { resolve: () => void }>
  released: Set<string>
  executing: Set<string>
  decisions: Map<string, { allow: boolean; reason: string }>
  childSeq: number
}

const sessions = new Map<string, Session>()
// Bounded tombstones keep close/state idempotent without retaining a whole
// transcript, pending calls or snapshot for every closed session.
const closedSessions = new Map<string, { seq: number }>()
// Effective model-facing tool names and the exact Hermes definitions this
// runtime registered. Both are re-verified at every control point.
const forwarded = new Set<string>()
const hermesDefinitions = new Map<string, string>()
let requestSeq = 0
// `open` awaits native session creation and tool registration. Admission is
// serialized across those gaps so two concurrent callers cannot both observe
// an empty slot and install two live sessions (the registry is runtime-wide).
let openAdmission: Promise<unknown> = Promise.resolve()
const admitOpen = <T>(work: () => Promise<T>): Promise<T> => {
  const result = openAdmission.then(work)
  openAdmission = result.then(
    () => undefined,
    () => undefined,
  )
  return result
}

// Stable, depth-bounded JSON projection for schema objects. Effect Schema
// instances are class objects with circular references; a plain JSON.stringify
// either throws or collapses them to `{}`, which would hide a changed output
// definition. This keeps primitive fields and marks depth/cycle cuts.
function stable(value: unknown, seen = new WeakSet<object>(), depth = 0): unknown {
  if (value === null || value === undefined) return value ?? null
  if (typeof value === "function") return `fn:${(value as Function).name}`
  if (typeof value !== "object") return value
  if (seen.has(value as object)) return "[circular]"
  if (depth >= 8) return "[depth]"
  seen.add(value as object)
  if (Array.isArray(value)) return value.slice(0, 256).map((item) => stable(item, seen, depth + 1))
  const out: Record<string, unknown> = {}
  const keys = Object.keys(value as object).sort()
  for (const key of keys.slice(0, 256)) {
    try {
      out[key] = stable((value as any)[key], seen, depth + 1)
    } catch {
      out[key] = "[unreadable]"
    }
  }
  return out
}

const signature = (tool: any) =>
  createHash("sha256")
    .update(
      JSON.stringify({
        name: tool.name,
        description: tool.description,
        input: stable(tool.input ?? null),
        output: stable(tool.output ?? null),
      }),
    )
    .digest("hex")

function emit(session: Session, type: string, data: Record<string, unknown> = {}) {
  // RPC output must be strict JSON: undefined object fields are rejected.
  const event = JSON.parse(JSON.stringify({ seq: ++session.seq, type, ...data }))
  session.events.push(event)
  if (session.events.length > EVENT_LOG_LIMIT) session.events.splice(0, session.events.length - EVENT_LOG_LIMIT)
  for (const wake of session.waiters) wake()
  session.waiters.clear()
}

function rejectPending(session: Session, reason: string) {
  session.rejectReason = reason
  for (const [callID, pending] of session.pending) {
    session.pending.delete(callID)
    pending.detach()
    pending.reject(new Error(reason))
  }
}

function sessionOf(id: string): Session {
  const session = sessions.get(id)
  if (!session) throw new Error(`unknown companion session: ${id}`)
  return session
}

function describe(session: Session) {
  return JSON.parse(
    JSON.stringify({
      sessionID: session.id,
      generation: session.generation,
      running: session.running ?? null,
      pending: Array.from(session.pending.keys()),
      gates: Array.from(session.gates.keys()),
      after: Array.from(session.after.keys()),
      decisions: Array.from(session.decisions.entries()).map(([callID, decision]) => ({ callID, ...decision })),
      executing: Array.from(session.executing),
      started: session.started,
      promptRunning: session.promptRunning,
      settled: session.settled,
      closed: session.closed,
      promptError: session.promptError ?? null,
      snapshotMessages: session.snapshot.length,
      activeRequest: session.activeRequest ?? null,
      outcome: session.outcome ?? null,
    }),
  )
}

function closedState(sessionID: string) {
  return {
    sessionID,
    generation: 0,
    running: null,
    pending: [],
    gates: [],
    after: [],
    decisions: [],
    executing: [],
    started: false,
    promptRunning: false,
    settled: true,
    closed: true,
    promptError: null,
    snapshotMessages: 0,
    activeRequest: null,
    outcome: null,
  }
}

function rememberClosed(sessionID: string) {
  closedSessions.set(sessionID, { seq: 0 })
  while (closedSessions.size > CLOSED_TOMBSTONES) {
    const oldest = closedSessions.keys().next().value
    if (oldest === undefined) break
    closedSessions.delete(oldest)
  }
}

// ---------------------------------------------------------------------------
// Bounded SSE outcome parser (OpenAI Chat Completions).
//
// It reads only what is needed to decide whether the response is a complete,
// well-formed completion. Text/tool-call bytes are counted, not retained. Each
// chunk is forwarded before parsing so the stock driver sees unchanged bytes.
// ---------------------------------------------------------------------------
function observeSse(body: ReadableStream<Uint8Array>, sink: (outcome: Outcome) => void): ReadableStream<Uint8Array> {
  const decoder = new TextDecoder()
  let buffer = ""
  let data: string[] = []
  let dataBytes = 0
  let halted = false
  let finished = false
  const state = {
    status: 200,
    bytes: 0,
    events: 0,
    done: false,
    finish: undefined as string | undefined,
    usage: undefined as unknown,
    error: undefined as string | undefined,
    textChars: 0,
    tools: new Map<number, { id?: string; name?: string; args: string }>(),
    argBytes: 0,
  }

  const fail = (message: string) => {
    if (!state.error) state.error = message
  }

  const validUsage = (usage: unknown): boolean => {
    if (!usage || typeof usage !== "object" || Array.isArray(usage)) return false
    const row = usage as Record<string, unknown>
    const prompt = row.prompt_tokens
    const completion = row.completion_tokens
    if (!Number.isInteger(prompt) || (prompt as number) < 0) return false
    if (!Number.isInteger(completion) || (completion as number) < 0) return false
    if (row.total_tokens === undefined || row.total_tokens === null) return true
    return Number.isInteger(row.total_tokens) && (row.total_tokens as number) === (prompt as number) + (completion as number)
  }

  const finalize = (ok: boolean) => {
    if (finished) return
    finished = true
    const calls = Array.from(state.tools.values())
    if (ok) {
      if (!state.done) fail("stream ended without [DONE]")
      else if (state.finish === undefined) fail("stream ended without finish_reason")
      else if (!SUPPORTED_FINISH.has(state.finish)) fail(`unsupported finish_reason ${state.finish}`)
      else if (!validUsage(state.usage)) fail("stream ended without valid prompt/completion token usage")
      else if (state.finish === "tool_calls" && calls.length === 0) fail("finish_reason tool_calls without tool call deltas")
      else if (calls.length > 0) {
        for (const call of calls) {
          if (!call.id || !call.name) {
            fail("tool call delta missing id or name")
            break
          }
          try {
            JSON.parse(call.args)
          } catch {
            fail("tool call arguments are not valid JSON")
            break
          }
        }
      }
    }
    sink({
      generation: null,
      requestSeq: null,
      status: state.status,
      ok: ok && state.error === undefined,
      done: state.done,
      finish: state.finish,
      usage: state.usage,
      error: state.error,
      calls: calls.length,
      textChars: state.textChars,
      bytes: state.bytes,
      events: state.events,
    })
  }

  const event = () => {
    if (data.length === 0) return
    const payload = data.join("\n")
    data = []
    dataBytes = 0
    state.events += 1
    if (state.events > MAX_EVENTS) return fail("too many SSE events")
    if (payload === "[DONE]") {
      state.done = true
      finalize(true)
      return
    }
    let parsed: any
    try {
      parsed = JSON.parse(payload)
    } catch {
      return fail("SSE data is not valid JSON")
    }
    if (parsed?.error) {
      fail(String(parsed.error.message ?? parsed.error.code ?? "provider error event"))
      return
    }
    const choice = Array.isArray(parsed?.choices) ? parsed.choices[0] : undefined
    if (parsed?.usage) state.usage = parsed.usage
    if (choice?.usage) state.usage = choice.usage
    const reason = choice?.finish_reason
    if (reason !== undefined && reason !== null) {
      if (state.finish !== undefined && state.finish !== reason) return fail("conflicting finish_reason values")
      state.finish = String(reason)
    }
    // Some OpenAI-compatible upstreams set finish_reason before the final
    // tool-argument fragment. Assembling continues until [DONE]; the terminal
    // contract (complete JSON, DONE, finish, usage) still decides the outcome.
    for (const tool of choice?.delta?.tool_calls ?? []) {
      const index = Number.isInteger(tool?.index) ? tool.index : state.tools.size
      const current = state.tools.get(index) ?? { id: undefined, name: undefined, args: "" }
      if (tool?.id) current.id = String(tool.id)
      if (tool?.function?.name) current.name = String(tool.function.name)
      const fragment = typeof tool?.function?.arguments === "string" ? tool.function.arguments : ""
      state.argBytes += fragment.length
      if (state.argBytes > MAX_CALL_ARG_BYTES) fail("tool call arguments exceed parser bound")
      current.args += fragment
      state.tools.set(index, current)
      if (state.tools.size > MAX_CALLS) fail("too many tool calls")
    }
    if (typeof choice?.delta?.content === "string") state.textChars += choice.delta.content.length
  }

  return body.pipeThrough(
    new TransformStream({
      transform(chunk: Uint8Array, controller: TransformStreamDefaultController) {
        state.bytes += chunk.byteLength ?? 0
        controller.enqueue(chunk)
        if (finished || halted) return
        buffer += decoder.decode(chunk, { stream: true })
        let index: number
        while ((index = buffer.indexOf("\n")) >= 0) {
          let line = buffer.slice(0, index)
          buffer = buffer.slice(index + 1)
          if (line.endsWith("\r")) line = line.slice(0, -1)
          if (line === "") {
            event()
            if (finished) return
            continue
          }
          if (line.startsWith(":")) continue
          if (line.startsWith("data:")) {
            const value = line.slice(5).replace(/^ /, "")
            dataBytes += value.length
            if (dataBytes > MAX_LINE_BYTES) {
              halted = true
              data = []
              return fail("SSE event exceeds parser bound")
            }
            data.push(value)
          }
        }
        // The bound is per line/SSE event, not per transport chunk: complete
        // small events in a large burst are processed above. Only an
        // unterminated line/event may not exceed the bound.
        if (buffer.length > MAX_LINE_BYTES) {
          halted = true
          buffer = ""
          return fail("SSE line exceeds parser bound")
        }
      },
      flush() {
        buffer += decoder.decode()
        if (buffer.length > MAX_LINE_BYTES) {
          halted = true
          buffer = ""
          fail("SSE line exceeds parser bound")
        }
        if (buffer.trim().length > 0) {
          let line = buffer.trimEnd()
          if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""))
        }
        event()
        finalize(true)
      },
      cancel() {
        finalize(true)
      },
    }),
  )
}

const JsonObject = { type: "object", additionalProperties: true } as const
const JsonValue = {
  anyOf: [
    { type: "object", additionalProperties: true },
    { type: "array", items: {} },
    { type: "string" },
    { type: "number" },
    { type: "boolean" },
    { type: "null" },
  ],
} as const

const ResultItem = {
  type: "object",
  properties: {
    callID: { type: "string" },
    content: { type: "string" },
    output: JsonValue,
    isError: { type: "boolean" },
  },
  required: ["callID"],
  additionalProperties: false,
} as const

const ResultsInput = {
  type: "object",
  properties: {
    sessionID: { type: "string" },
    results: { type: "array", items: ResultItem },
  },
  required: ["sessionID", "results"],
  additionalProperties: false,
} as const

const RpcContract = Rpc.define({
  id: RPC_ID,
  methods: {
    hello: {
      input: {
        type: "object",
        properties: { envKeys: { type: "array", items: { type: "string" } } },
        additionalProperties: false,
      },
      output: JsonObject,
    },
    // Native discovery metadata only. No settings, headers, bodies or
    // credentials ever cross this boundary.
    catalog: {
      input: { type: "object", properties: {}, additionalProperties: false },
      output: JsonObject,
    },
    open: {
      input: {
        type: "object",
        properties: {
          providerID: { type: "string" },
          modelID: { type: "string" },
          // Exact native ID allocated by OpenCode on an earlier run. The
          // companion verifies it exists before adopting it; it never mints one.
          sessionID: { type: "string" },
          // First-class Hermes tools. `codemode: true` opts one into Code Mode
          // so an approved `execute` program can call `tools.<name>`; the call
          // still parks in forwardExecutor and Hermes executes it.
          hermesTools: {
            type: "array",
            items: {
              type: "object",
              properties: {
                name: { type: "string" },
                description: { type: "string" },
                parameters: JsonObject,
                codemode: { type: "boolean" },
              },
              required: ["name", "description"],
              additionalProperties: false,
            },
          },
          // Native descriptors to keep (subset of TRANSLATED_NATIVE).
          nativeTools: { type: "array", items: { type: "string" } },
        },
        required: ["providerID", "modelID"],
        additionalProperties: false,
      },
      output: JsonObject,
      errors: { invalid_state: JsonObject },
    },
    advance: {
      input: {
        type: "object",
        properties: {
          sessionID: { type: "string" },
          generation: { type: "integer" },
          // Canonical Hermes transcript in OpenCode LLM message shape. It is
          // installed verbatim for the one authorized primary request.
          snapshot: { type: "array", items: JsonObject },
          // Hermes system/developer instructions, appended to OpenCode's own
          // system parts. They never replace the native prompt.
          systemParts: { type: "array", items: { type: "string" } },
          userText: { type: "string" },
          // Current Hermes registrations for this generation. Pure additions
          // are allowed; the session being advanced is never treated as a
          // conflicting live generation.
          hermesTools: {
            type: "array",
            items: {
              type: "object",
              properties: {
                name: { type: "string" },
                description: { type: "string" },
                parameters: JsonObject,
                codemode: { type: "boolean" },
              },
              required: ["name", "description"],
              additionalProperties: false,
            },
          },
          // Native descriptors to keep (subset of TRANSLATED_NATIVE).
          nativeTools: { type: "array", items: { type: "string" } },
        },
        required: ["sessionID", "generation"],
        additionalProperties: false,
      },
      output: JsonObject,
      errors: { invalid_state: JsonObject },
    },
    // Explicit generation boundary: settle parked executors, interrupt a
    // running prompt, discard its results, and keep the native session.
    finish: {
      input: {
        type: "object",
        properties: { sessionID: { type: "string" } },
        required: ["sessionID"],
        additionalProperties: false,
      },
      output: JsonObject,
      errors: { invalid_state: JsonObject },
    },
    read: {
      input: {
        type: "object",
        properties: { sessionID: { type: "string" }, cursor: { type: "integer" } },
        required: ["sessionID"],
        additionalProperties: false,
      },
      output: JsonObject,
      errors: { invalid_state: JsonObject },
    },
    // Hermes' verdict on one parked `execute` program. The program runs in
    // OpenCode's interpreter only after `allow`.
    decide: {
      input: {
        type: "object",
        properties: {
          sessionID: { type: "string" },
          callID: { type: "string" },
          allow: { type: "boolean" },
          reason: { type: "string" },
        },
        required: ["sessionID", "callID", "allow"],
        additionalProperties: false,
      },
      output: JsonObject,
      errors: { invalid_state: JsonObject },
    },
    // Results for nested `tools.*` calls inside one approved program. These do
    // not advance a generation: execution is still inside execute. The batch
    // is validated completely before any result is settled.
    results: {
      input: ResultsInput,
      output: JsonObject,
      errors: { invalid_state: JsonObject },
    },
    // One-revision alias of `results`; canonical name is `results`.
    settle: {
      input: ResultsInput,
      output: JsonObject,
      errors: { invalid_state: JsonObject },
    },
    state: {
      input: {
        type: "object",
        properties: { sessionID: { type: "string" } },
        required: ["sessionID"],
        additionalProperties: false,
      },
      output: JsonObject,
      errors: { invalid_state: JsonObject },
    },
    close: {
      input: {
        type: "object",
        properties: { sessionID: { type: "string" } },
        required: ["sessionID"],
        additionalProperties: false,
      },
      output: JsonObject,
      errors: { invalid_state: JsonObject },
    },
  },
  events: {},
})

// Non-secret native catalog projection. This is an allowlist: provider or
// model settings, headers, bodies, base URLs and credentials never cross the
// RPC boundary even though the live API may expose them.
const MODEL_FIELDS = [
  "id",
  "modelID",
  "providerID",
  "canonical",
  "type",
  "family",
  "name",
  "compatibility",
  "package",
  "capabilities",
  "variants",
  "time",
  "cost",
  "status",
  "enabled",
  "limit",
] as const

function publicModel(model: any, providerPackage: string | undefined) {
  const row: Record<string, any> = {}
  for (const key of MODEL_FIELDS) {
    if (model?.[key] === undefined) continue
    row[key] = model[key]
  }
  if (Array.isArray(model?.variants))
    row.variants = model.variants
      .filter((variant: any) => variant && typeof variant === "object" && typeof variant.id === "string")
      .map((variant: any) => ({ id: variant.id }))
  else delete row.variants
  if (typeof model?.package === "string") row.package = model.package
  if (providerPackage !== undefined) row.providerPackage = providerPackage
  return row
}

function publicProvider(provider: any) {
  return {
    id: String(provider?.id ?? ""),
    name: String(provider?.name ?? provider?.id ?? ""),
    package: typeof provider?.package === "string" ? provider.package : null,
  }
}

export default Plugin.define({
  id: COMPANION,
  async setup(ctx) {
    const observeCompletions = ctx.options.observeCompletions !== false

    // Read-only completion observer. Stock Chat Completions only: the hook
    // fires for every session request, so auxiliary kinds are excluded
    // explicitly, and each response is bound to one authorized request.
    if (observeCompletions) {
      await ctx.session.hook("http.response", (event: any) => {
        const session = sessions.get(String(event.sessionID))
        const kind = String(event.kind ?? "unknown")
        if (kind !== "primary") {
          if (session) emit(session, "observer_excluded", { kind })
          return
        }
        const response = event.response as Response
        const contentType = response?.headers?.get("content-type") ?? ""
        if (!session || !response?.body || !contentType.includes("text/event-stream")) return
        const active = session.activeRequest
        if (!active) {
          emit(session, "correlation_error", { reason: "response without an authorized request" })
          return
        }
        if (session.outcome) {
          emit(session, "correlation_error", { reason: "second response for one authorized request" })
          return
        }
        if (response.status !== 200) {
          session.outcome = {
            generation: active.generation,
            requestSeq: active.requestSeq,
            status: response.status,
            ok: false,
            done: false,
            error: `HTTP ${response.status}`,
            calls: 0,
            textChars: 0,
            bytes: 0,
            events: 0,
          }
          emit(session, "completion", { ...session.outcome })
          return
        }
        const generation = active.generation
        const seq = active.requestSeq
        const stream = observeSse(response.body, (outcome) => {
          const bound = { ...outcome, generation, requestSeq: seq }
          session.outcome = bound
          emit(session, "completion", { ...bound })
        })
        event.response = new Response(stream, {
          status: response.status,
          statusText: response.statusText,
          headers: response.headers,
        })
      })
    }

    // Held executor: no OpenCode implementation runs. The call is parked under
    // its native callID until Hermes returns a result over RPC.
    const forwardExecutor =
      (name: string, outputSchema: unknown) =>
      async (input: unknown, context: any): Promise<ToolResult> => {
        const session = sessions.get(String(context.sessionID))
        if (!session) throw new Error(`hermes-oc-native: tool ${name} outside a companion session`)
        if (session.closed) throw new Error(`hermes-oc-native: tool ${name} withheld: session closed`)
        if (session.rejectReason) throw new Error(`hermes-oc-native: tool ${name} withheld: ${session.rejectReason}`)
        // A nested CodeMode `tools.*` call carries the outer execute call's
        // id; give it a child id so two nested calls cannot collide.
        const nested = session.executing.has(String(context.id))
        const callID = nested ? `${String(context.id)}#${++session.childSeq}` : String(context.id)
        if (nested && session.childSeq > MAX_NESTED_CALLS) {
          emit(session, "nested_limit", { parentCallID: String(context.id), limit: MAX_NESTED_CALLS })
          throw new Error(`hermes-oc-native: nested call limit ${MAX_NESTED_CALLS} exceeded`)
        }
        if (session.pending.size >= MAX_PENDING) {
          emit(session, "pending_limit", { name, limit: MAX_PENDING })
          throw new Error(`hermes-oc-native: pending executor limit ${MAX_PENDING} exceeded`)
        }
        if (nested) emit(session, "nested_call", { callID, parentCallID: String(context.id), name, input })
        const result = await new Promise<ToolResult>((resolve, reject) => {
          const onAbort = () => {
            session.pending.delete(callID)
            reject(new Error("tool call cancelled"))
          }
          context.signal.addEventListener("abort", onAbort, { once: true })
          session.pending.set(callID, {
            callID,
            name,
            input,
            resolve,
            reject,
            detach: () => context.signal.removeEventListener("abort", onAbort),
          })
        })
        if (result.output === undefined || outputSchema === undefined) return result
        try {
          // Native tools declare decoded output types (some are classes). The
          // Hermes host supplies JSON, so decode it before core encodes it again.
          return { content: result.content, output: Schema.decodeUnknownSync(outputSchema as any)(result.output) }
        } catch (error) {
          throw new Error(`hermes-oc-native: invalid native output for ${name}: ${String(error)}`)
        }
      }

    const keptNative = (session: Session) =>
      session.nativeTools ?? new Set(hermesDefinitions.has("read_file") ? ["read"] : [])
    // Validate a requested native subset against the live registry: only a
    // translated, forwarded native descriptor that Hermes does not own.
    const nativeSubset = async (names: unknown): Promise<Set<string> | undefined> => {
      if (names === undefined) return undefined
      if (!Array.isArray(names)) throw new Error("hermes-oc-native: nativeTools must be an array")
      const ids = new Set((await ctx.tool.list()).map((tool: any) => String(tool.id)))
      const kept = new Set<string>()
      for (const raw of names) {
        const name = String(raw)
        if (!TRANSLATED_NATIVE.has(name) || !ids.has(name) || hermesDefinitions.has(name))
          throw new Error(`hermes-oc-native: native tool cannot be exposed: ${name}`)
        kept.add(name)
      }
      return kept
    }

    // Every native tool gets the waiting executor; definitions are untouched.
    // Built-in tools register after this plugin's setup, so re-reconcile at
    // every control point instead of trusting one startup snapshot.
    const nativeSignatures = new Map<string, string>()
    const ensureForwarded = async () => {
      const before = await ctx.tool.list()
      if (before.some((tool: any) => tool[FORWARD] !== true)) {
        await ctx.tool.transform((editor: any) => {
          for (const tool of editor.list()) {
            if (tool[FORWARD] === true) continue
            editor.update(tool.id, (draft: any) => {
              draft.execute = forwardExecutor(String(tool.id), (tool as any).output)
              draft[FORWARD] = true
            })
          }
        })
        await ctx.tool.reload()
      }
      const after: any[] = await ctx.tool.list()
      forwarded.clear()
      for (const tool of after) forwarded.add(String(tool.id))
      const missing = after.filter((tool) => tool[FORWARD] !== true)
      if (missing.length > 0)
        throw new Error(`hermes-oc-native: unwrapped native tools: ${missing.map((tool: any) => tool.id).join(",")}`)
      const byId = new Map(after.map((tool) => [String(tool.id), tool]))
      for (const tool of after) {
        const id = String(tool.id)
        if (hermesDefinitions.has(id)) continue
        const current = signature(tool)
        const previous = nativeSignatures.get(id)
        if (previous !== undefined && previous !== current)
          throw new Error(`hermes-oc-native: native tool definition changed: ${id}`)
        nativeSignatures.set(id, current)
      }
      for (const [name, expected] of hermesDefinitions) {
        const tool = byId.get(name)
        if (!tool || signature(tool) !== expected)
          throw new Error(`hermes-oc-native: Hermes tool definition changed after registration: ${name}`)
      }
      return after
    }

    await ensureForwarded()

    // First-class Hermes tools: registered with the host's real schema through
    // the public editor. Names must not collide with the native registry;
    // Python supplies a reversible wire alias before calling `open`, so a
    // Hermes capability is never dropped because OpenCode shares its name.
    const registerHermesTools = async (definitions: any[], exceptSessionID?: string) => {
      if (definitions.length > MAX_HERMES_TOOLS)
        throw new Error(`hermes-oc-native: too many Hermes tools: ${definitions.length} > ${MAX_HERMES_TOOLS}`)
      const current: any[] = await ctx.tool.list()
      const known = new Map(current.map((tool) => [String(tool.id), tool]))
      const pending: any[] = []
      const revised: any[] = []
      const desired = new Set<string>()
      for (const tool of definitions) {
        const name = String(tool.name ?? "")
        if (!name) throw new Error("hermes-oc-native: Hermes tool requires a name")
        if (desired.has(name)) throw new Error(`hermes-oc-native: duplicate Hermes tool name: ${name}`)
        desired.add(name)
        if (
          !tool.parameters ||
          typeof tool.parameters !== "object" ||
          Array.isArray(tool.parameters)
        )
          throw new Error(`hermes-oc-native: Hermes tool ${name} requires an object parameters schema`)
        const expected = signature({
          name,
          description: String(tool.description ?? ""),
          input: tool.parameters,
        })
        const existing = known.get(name)
        if (existing && !hermesDefinitions.has(name))
          throw new Error(`hermes-oc-native: Hermes tool names collide with native registry: ${name}`)
        const previous = hermesDefinitions.get(name)
        if (previous !== undefined) {
          if (previous === expected) continue
          revised.push({ ...tool, name, definition: expected })
          continue
        }
        pending.push({ ...tool, name, definition: expected })
      }
      const stale = Array.from(hermesDefinitions.keys()).filter((name) => !desired.has(name))
      if (pending.length === 0 && revised.length === 0 && stale.length === 0) return
      // A live session registered the previous registry when it opened. Never
      // swap a definition or remove a tool under a different live session; the
      // next `open` after the one live slot closes may revise. `advance` passes
      // its own id as `exceptSessionID`, so its next generation may update the
      // definitions it will advertise. Pure additions stay allowed.
      const live = Array.from(sessions.values()).filter(
        (session) => !session.closed && session.id !== exceptSessionID,
      )
      if (live.length > 0 && (revised.length > 0 || stale.length > 0)) {
        const affected = [...pending, ...revised].map((tool) => String(tool.name)).concat(stale)
        const what = revised.length > 0 ? "Hermes tool definition changed" : "Hermes tool removed"
        throw new Error(
          `hermes-oc-native: ${what} while ${live.length} live generation(s) still advertise the previous registry: ${affected.join(",")}`,
        )
      }
      // Removal and re-add are separate transforms so a revised definition never
      // depends on the editor accepting an in-place input/description mutation.
      if (revised.length > 0 || stale.length > 0) {
        await ctx.tool.transform((editor: any) => {
          for (const tool of revised) editor.remove(String(tool.name))
          for (const name of stale) editor.remove(name)
        })
        await ctx.tool.reload()
      }
      const added = [...pending, ...revised]
      if (added.length > 0) {
        await ctx.tool.transform((editor: any) => {
          for (const tool of added) {
            const name = String(tool.name)
            editor.add({
              name,
              description: String(tool.description ?? ""),
              input: tool.parameters,
              options: tool.codemode === true ? undefined : { codemode: false },
              [FORWARD]: true,
              execute: forwardExecutor(name, undefined),
            } as any)
          }
        })
        await ctx.tool.reload()
      }
      const after = new Map((await ctx.tool.list()).map((tool) => [String(tool.id), tool]))
      const missing = added.map((tool) => String(tool.name)).filter((name) => !after.has(name))
      if (missing.length > 0) throw new Error(`hermes-oc-native: Hermes tools were not registered: ${missing.join(",")}`)
      const leftover = stale.filter((name) => after.has(name))
      if (leftover.length > 0) throw new Error(`hermes-oc-native: stale Hermes tools were not removed: ${leftover.join(",")}`)
      for (const tool of revised) {
        const registered = after.get(String(tool.name))
        if (!registered || signature(registered) !== tool.definition)
          throw new Error(`hermes-oc-native: Hermes tool definition was not revised: ${String(tool.name)}`)
      }
      for (const name of stale) {
        hermesDefinitions.delete(name)
        forwarded.delete(name)
      }
      for (const tool of added) {
        const name = String(tool.name)
        hermesDefinitions.set(name, tool.definition)
        forwarded.add(name)
      }
    }

    // `execute` is synthesized by the snapshot, so the registry cannot replace
    // it and its native descriptor stays exactly as OpenCode emitted it (free
    // models keep a familiar tool). It is never authorized: no gate, no
    // `decide`, no JavaScript interpreter. The model receives an explicit
    // refusal naming the enabled Hermes Python `execute_code` tool, and Hermes
    // never receives a JS program to run as Python.
    // Only the exposed native `read` and genuine Hermes registrations may
    // dispatch. Everything else is refused before an executor runs, even if it
    // is hallucinated into the catalog by a model.
    await ctx.tool.hook("execute.before", async (event) => {
      const session = sessions.get(String(event.sessionID))
      if (!session) throw new Error(`hermes-oc-native: unscoped session ${event.sessionID}`)
      const name = String(event.tool)
      if (hermesDefinitions.has(name) || keptNative(session).has(name)) {
        if (!forwarded.has(name)) throw new Error(`hermes-oc-native: unscoped tool refused: ${name}`)
        return
      }
      const reason = nativeRefusal(name)
      if (name === "execute") {
        emit(session, "execute_refused", {
          callID: String(event.id),
          generation: session.generation,
          reason,
        })
      } else {
        emit(session, "native_refused", {
          callID: String(event.id),
          generation: session.generation,
          name,
          reason,
        })
      }
      throw new Error(`hermes-oc-native: ${reason}`)
    })

    // Defence in depth: execute.before refuses every program, so a completed
    // execute must never be observable here. Fail instead of parking a
    // JavaScript result for a later advance.
    await ctx.tool.hook("execute.after", async (event) => {
      if (event.tool !== "execute") return
      const session = sessions.get(String(event.sessionID))
      if (!session || session.closed) return
      emit(session, "execute_unexpected", {
        callID: String(event.id),
        generation: session.generation,
        status: String(event.status ?? "unknown"),
      })
      throw new Error(`hermes-oc-native: ${EXECUTE_UNSUPPORTED}`)
    })

    // Code Mode's `fetch` bypasses the wrapped registry. Hermes owns network
    // access, so deny it outright.
    await ctx.permission.hook("evaluate", (event) => {
      if (event.action !== "webfetch" || !sessions.has(String(event.sessionID))) return
      event.effect = "deny"
      event.message = "hermes-oc-native: network access is owned by Hermes"
    })

    // One authorized generation per advance. The context hook also installs the
    // canonical snapshot so OpenCode's own terminal/failed step state never
    // leaks into the next Hermes invocation.
    await ctx.session.hook("context", async (event) => {
      const session = sessions.get(String(event.sessionID))
      if (!session) throw new Error(`hermes-oc-native: unscoped session ${event.sessionID}`)
      if (session.closed) throw new Error("hermes-oc-native: session closed")
      // Tool registration can settle after `advance` (for example a provider
      // gains an image tool). Wrap it before the model request rather than
      // advertising an executor Hermes does not own.
      await ensureForwarded()
      // One catalog policy: keep the native `read` compatibility descriptor and
      // every genuine Hermes registration; remove every other native family.
      // Identity, not a name filter, so a Hermes tool that shares a native name
      // is preserved (Python owns the reversible wire alias).
      const exposed = event.tools as Record<string, unknown>
      const removed: string[] = []
      // A native descriptor is advertised only when Python kept it for this
      // generation (its Hermes target is enabled).
      const kept = keptNative(session)
      for (const name of Object.keys(exposed)) {
        if (hermesDefinitions.has(name)) continue
        if (kept.has(name)) continue
        delete exposed[name]
        removed.push(name)
      }
      if (removed.length > 0) emit(session, "catalog_filtered", { removed })
      // Fail closed if a kept tool has no wrapper: an unwrapped executor would
      // let OpenCode run a tool Hermes did not authorize.
      for (const name of Object.keys(exposed)) {
        if (!forwarded.has(name)) {
          emit(session, "tool_guard", { name, reason: "unwrapped tool in model request" })
          throw new Error(`hermes-oc-native: unwrapped tool in model request: ${name}`)
        }
      }
      const generation = session.authorizations.shift()
      if (generation === undefined) {
        emit(session, "unauthorized", { message: "model request without a Hermes generation authorization" })
        throw new Error("hermes-oc-native: unauthorized model request")
      }
      session.running = generation
      session.activeRequest = { generation, requestSeq: ++requestSeq }
      session.outcome = undefined
      // Always replace the native transcript, including an explicitly empty
      // one (advance rejects empty snapshots, so this is never a fallback to
      // OpenCode's retained history).
      event.messages = session.snapshot.map((message) => ({ ...message }))
      // Genuine OpenCode system parts stay first and untouched; Hermes
      // instructions are appended as extra text parts (never a replacement).
      if (session.systemParts.length > 0)
        event.system.push(...session.systemParts.map((text) => ({ type: "text" as const, text })))
      emit(session, "snapshot_applied", {
        generation,
        messages: event.messages.length,
        systemParts: session.systemParts.length,
        nativeSystemParts: event.system.length - session.systemParts.length,
      })
      emit(session, "authorized", { generation, requestSeq: session.activeRequest.requestSeq })
    })

    await ctx.session.hook("title", (event) => {
      event.result = "Hermes session"
    })
    await ctx.session.hook("compaction", () => {
      throw new Error("hermes-oc-native: compaction is owned by Hermes")
    })
    await ctx.session.hook("retry", (event) => {
      event.decision = { retry: false }
    })

    void (async () => {
      for await (const event of ctx.event.subscribe({})) {
        const payload = event as any
        const body = payload.data ?? {}
        const session = sessions.get(String(body.sessionID))
        if (!session) continue
        if (payload.type === "session.step.started") {
          const messageID = String(body.assistantMessageID)
          if (session.primaryAssistant === undefined) session.primaryAssistant = messageID
          if (session.running !== undefined) session.stepGeneration.set(messageID, session.running)
          emit(session, "step_started", { assistantMessageID: messageID, primary: messageID === session.primaryAssistant })
          continue
        }
        if (payload.type === "session.tool.input.started") {
          session.names.set(String(body.id), String(body.name))
          continue
        }
        if (payload.type === "session.tool.called") {
          const list = session.calls.get(String(body.assistantMessageID)) ?? []
          list.push({
            callID: String(body.id),
            name: session.names.get(String(body.id)) ?? String(body.name),
            input: body.input,
            native: body.native,
          })
          session.calls.set(String(body.assistantMessageID), list)
          continue
        }
        if (payload.type === "session.tool.failed") {
          emit(session, "tool_failed", {
            assistantMessageID: String(body.assistantMessageID),
            callID: String(body.id),
            error: body.error,
          })
          continue
        }
        if (payload.type === "session.text.ended") {
          emit(session, "text", {
            assistantMessageID: String(body.assistantMessageID),
            text: String(body.text ?? ""),
            native: body.native,
          })
          continue
        }
        if (payload.type === "session.reasoning.ended") {
          emit(session, "reasoning", {
            assistantMessageID: String(body.assistantMessageID),
            text: String(body.text ?? ""),
            native: body.native,
          })
          continue
        }
        if (payload.type === "session.step.streamed") {
          const messageID = String(body.assistantMessageID)
          const generation = session.stepGeneration.get(messageID) ?? session.running ?? null
          const calls = session.calls.get(messageID) ?? []
          const primary = messageID === session.primaryAssistant
          const outcome = session.outcome
          const positive =
            outcome?.ok === true &&
            outcome.generation === generation &&
            (outcome.finish === "tool_calls" || outcome.finish === "stop" || outcome.finish === "length")
          if (!positive) {
            const reason = outcome
              ? `response completion rejected: ${outcome.error ?? outcome.finish ?? "unknown"}`
              : "response completion was not observable before tool dispatch"
            rejectPending(session, reason)
            emit(session, "response_unverified", { generation, assistantMessageID: messageID, calls, reason, outcome: outcome ?? null })
            // No call is exposed. The rejected executors settle the step as a
            // failure; the tool side effect never runs.
            emit(session, "step_streamed", { generation, assistantMessageID: messageID, primary, calls: [], completionValidated: false })
            continue
          }
          if (calls.length === 0) {
            emit(session, "step_streamed", {
              generation,
              assistantMessageID: messageID,
              primary,
              calls,
              completionValidated: true,
              completion: {
                requestSeq: outcome.requestSeq,
                finish: outcome.finish ?? null,
                usage: outcome.usage ?? null,
              },
            })
            continue
          }
          emit(session, "step_streamed", {
            generation,
            assistantMessageID: messageID,
            primary,
            calls,
            completionValidated: true,
            completion: {
              requestSeq: outcome.requestSeq,
              finish: outcome.finish ?? null,
              usage: outcome.usage ?? null,
            },
          })
          continue
        }
        if (payload.type === "session.step.ended") {
          const messageID = String(body.assistantMessageID)
          emit(session, "step_ended", {
            generation: session.stepGeneration.get(messageID) ?? session.running ?? null,
            assistantMessageID: messageID,
            primary: messageID === session.primaryAssistant,
            finish: body.finish,
            rawFinish: body.rawFinish,
            usage: body.tokens,
            cost: body.cost,
            native: body.native,
          })
          continue
        }
        if (payload.type === "session.step.failed") {
          const messageID = String(body.assistantMessageID)
          emit(session, "step_failed", {
            generation: session.stepGeneration.get(messageID) ?? session.running ?? null,
            assistantMessageID: messageID,
            primary: messageID === session.primaryAssistant,
            error: body.error,
          })
        }
      }
    })().catch((error) => {
      for (const session of sessions.values()) emit(session, "event_stream_error", { message: String(error) })
    })

    // Atomic nested-result settlement. Every entry is validated before any
    // executor is released; a stale or duplicate id cannot half-apply a batch.
    const settleBatch = async (input: any, context: any) => {
      const id = String(input.sessionID)
      if (closedSessions.has(id)) return context.error("invalid_state", "session closed", { sessionID: id })
      const session = sessionOf(id)
      if (session.closed) return context.error("invalid_state", "session closed", { sessionID: session.id })
      const results = input.results
      if (!Array.isArray(results) || results.length === 0)
        return context.error("invalid_state", "results requires a non-empty array", { sessionID: session.id })
      if (results.length > MAX_RESULTS)
        return context.error("invalid_state", `results exceeds ${MAX_RESULTS} entries`, { count: results.length })
      const seen = new Set<string>()
      const accepted: Array<{ callID: string; pending: PendingCall; result: any }> = []
      for (const result of results) {
        const callID = String(result?.callID ?? "")
        if (!callID) return context.error("invalid_state", "result has no callID", {})
        if (seen.has(callID)) return context.error("invalid_state", "duplicate callID in result batch", { callID })
        seen.add(callID)
        const pending = session.pending.get(callID)
        if (!pending)
          return context.error("invalid_state", "no pending nested executor for result", { callID })
        accepted.push({ callID, pending, result })
      }
      for (const { callID, pending, result } of accepted) {
        session.pending.delete(callID)
        pending.detach()
        if (result.isError) pending.reject(new Error(String(result.content ?? "tool denied")))
        else
          pending.resolve({
            content: String(result.content ?? ""),
            ...(result.output === undefined ? {} : { output: result.output }),
          })
        emit(session, "nested_settled", { callID, name: pending.name, isError: result.isError === true })
      }
      return { settled: accepted.length, session: describe(session) }
    }

    await ctx.rpc.register(RpcContract, {
      hello: async () => {
        const tools: any[] = await ensureForwarded()
        const listed: any = await ctx.plugin.list()
        const plugins = Array.isArray(listed) ? listed : (listed?.data ?? [])
        return {
          protocol: PROTOCOL,
          companion: COMPANION,
          binaryVersion: ctx.app.version,
          location: { directory: ctx.location.directory, project: ctx.location.project.canonical },
          sessions: sessions.size,
          closedSessions: closedSessions.size,
          plugins: plugins.map((plugin: any) => plugin.id ?? plugin.name ?? String(plugin)),
          tools: tools.map((tool: any) => ({
            id: tool.id,
            name: tool.name,
            definition: signature(tool),
            outputDefinition: signature({ name: "", description: "", input: null, output: tool.output ?? null }),
            codemode: tool.options?.codemode ?? null,
            namespace: tool.options?.namespace ?? null,
            forwarded: tool[FORWARD] === true,
            // Registered by this provider from a Hermes schema, not a native tool.
            hermes: hermesDefinitions.has(String(tool.id)),
          })),
        }
      },

      // Live public discovery. Model records keep the native cost tiers,
      // capabilities, limits, status and enabled flag; owning provider package
      // is attached as providerPackage. Per-model/per-provider overlay secrets
      // are never copied.
      catalog: async () => {
        const [modelResult, providerResult]: any[] = await Promise.all([
          ctx.model.list(),
          ctx.provider.list(),
        ])
        const models: any[] = Array.isArray(modelResult?.data) ? modelResult.data : []
        const providers: any[] = Array.isArray(providerResult?.data) ? providerResult.data : []
        const packages = new Map<string, string>()
        for (const provider of providers) {
          if (typeof provider?.id === "string" && typeof provider?.package === "string")
            packages.set(provider.id, provider.package)
        }
        return {
          models: models
            .filter((model: any) => model && typeof model === "object")
            .map((model: any) => publicModel(model, packages.get(String(model.providerID)))),
          providers: providers
            .filter((provider: any) => provider && typeof provider === "object")
            .map(publicProvider),
        }
      },

      open: async (input: any, context: any) =>
        admitOpen(async () => {
          const providerID = String(input.providerID)
          const modelID = String(input.modelID)
          if (!providerID || !modelID)
            return context.error("invalid_state", "providerID and modelID are required", {})
          const requested =
            typeof input.sessionID === "string" && input.sessionID ? input.sessionID : undefined
          // Admission and native-id validation finish before the runtime-wide
          // tool registry is touched: a refused open must not leave a ghost
          // definition or replace the one live session.
          if (requested && sessions.has(requested))
            return context.error(
              "invalid_state",
              `session ${requested} is already open in this runtime; close it first`,
              { sessionID: requested },
            )
          if (requested && closedSessions.has(requested))
            return context.error(
              "invalid_state",
              `session ${requested} was closed in this runtime; allocate a new native session`,
              { sessionID: requested, closed: true },
            )
          if (sessions.size >= MAX_LIVE_SESSIONS)
            return context.error("invalid_state", `live session limit ${MAX_LIVE_SESSIONS} reached`, {
              sessions: sessions.size,
            })
          // Reuse only an exact native ID allocated by OpenCode on an earlier
          // run. A missing ID is reported; the caller allocates a fresh official
          // one and replaces its mapping (never recreate an invented id).
          let info: any
          let createdNative = false
          if (requested) {
            try {
              const existing: any = await ctx.session.get({ sessionID: requested })
              info = existing?.data ?? existing
            } catch (error) {
              return context.error("invalid_state", `native session ${requested} no longer exists`, {
                sessionID: requested,
                missing: true,
              })
            }
            if (!info || String(info.id ?? info.sessionID ?? "") !== requested)
              return context.error("invalid_state", `native session ${requested} returned a different id`, {
                sessionID: requested,
                missing: true,
              })
          } else {
            const created: any = await ctx.session.create({
              location: { directory: ctx.location.directory },
              model: { providerID, id: modelID },
            })
            info = created?.data ?? created
            createdNative = true
          }
          let nativeTools: Set<string> | undefined
          try {
            // Exact current catalog: undefined/empty removes prior Hermes
            // registrations so a disabled tool cannot remain callable.
            // Validate the native subset first so a refusal leaves the registry untouched.
            nativeTools = await nativeSubset(input.nativeTools)
            await registerHermesTools(Array.isArray(input.hermesTools) ? input.hermesTools : [])
          } catch (error) {
            if (createdNative) {
              // A failed admission must not leave the freshly created native
              // session behind; tool definitions are validated before they are
              // applied, so a refused registration does not add one.
              try {
                await ctx.session.remove({ sessionID: String(info?.id ?? info?.sessionID ?? "") })
              } catch {
                // Best effort: no companion session is admitted either way.
              }
            }
            return context.error("invalid_state", String((error as Error)?.message ?? error), {})
          }
          const session: Session = {
            id: String(info.id ?? info.sessionID),
            generation: 0,
            authorizations: [],
            running: undefined,
            userText: undefined,
            started: false,
            promptRunning: false,
            settled: true,
            closed: false,
            snapshot: [],
            systemParts: [],
            gates: new Map(),
            after: new Map(),
            released: new Set(),
            executing: new Set(),
            decisions: new Map(),
            childSeq: 0,
            pending: new Map(),
            calls: new Map(),
            names: new Map(),
            nativeTools,
            stepGeneration: new Map(),
            events: [],
            seq: 0,
            waiters: new Set(),
          }
          sessions.set(session.id, session)
          emit(session, "opened", { model: `${providerID}/${modelID}`, reused: Boolean(requested) })
          return {
            sessionID: session.id,
            location: ctx.location.directory,
            model: `${providerID}/${modelID}`,
            reused: Boolean(requested),
          }
        }),

      advance: async (input: any, context: any) => {
        const id = String(input.sessionID)
        if (closedSessions.has(id)) return context.error("invalid_state", "session closed", { sessionID: id })
        const session = sessionOf(id)
        if (session.closed) return context.error("invalid_state", "session closed", { sessionID: session.id })
        if (input.generation !== session.generation + 1)
          return context.error("invalid_state", "generation must be the next one", {
            expected: session.generation + 1,
            received: input.generation,
          })
        if (session.authorizations.length > 0)
          return context.error("invalid_state", "a generation permit is already pending", {
            sessionID: session.id,
          })
        const afterRelease = session.after.size > 0
        const deniedRelease =
          session.executing.size === 0 &&
          session.gates.size > 0 &&
          Array.from(session.decisions.values()).some((decision) => !decision.allow)
        if ((session.gates.size > 0 || session.executing.size > 0) && !afterRelease && !deniedRelease)
          return context.error("invalid_state", "execute is still parked or running", { sessionID: session.id })
        // A retained session accepts the next canonical generation only after
        // the previous one reached the explicit boundary. Never advance over a
        // parked executor or a live run.
        if (!session.settled)
          return context.error("invalid_state", "previous generation must be finished first; call finish", {
            sessionID: session.id,
          })
        if (!Array.isArray(input.snapshot))
          return context.error("invalid_state", "advance requires a canonical snapshot array", {
            sessionID: session.id,
          })
        if (input.snapshot.length === 0)
          return context.error("invalid_state", "snapshot must not be empty; native history is never reused", {
            sessionID: session.id,
          })
        if (input.snapshot.length > MAX_SNAPSHOT_MESSAGES)
          return context.error("invalid_state", `snapshot exceeds ${MAX_SNAPSHOT_MESSAGES} messages`, {
            messages: input.snapshot.length,
          })
        const snapshotBytes = Buffer.byteLength(JSON.stringify(input.snapshot), "utf8")
        if (snapshotBytes > MAX_SNAPSHOT_BYTES)
          return context.error("invalid_state", `snapshot exceeds ${MAX_SNAPSHOT_BYTES} bytes`, {
            bytes: snapshotBytes,
          })
        const systemParts = Array.isArray(input.systemParts) ? input.systemParts.map(String) : []
        const systemBytes = Buffer.byteLength(JSON.stringify(systemParts), "utf8")
        if (systemBytes > MAX_SYSTEM_BYTES)
          return context.error("invalid_state", `system parts exceed ${MAX_SYSTEM_BYTES} bytes`, {
            bytes: systemBytes,
          })
        await ensureForwarded()
        // Late tools enter on the next generation; pure additions are allowed
        // while this session advertises the previous registry. An omitted list
        // leaves the registry untouched (direct-protocol callers).
        let nativeTools = session.nativeTools
        try {
          if (input.nativeTools !== undefined) nativeTools = await nativeSubset(input.nativeTools)
          if (Array.isArray(input.hermesTools)) await registerHermesTools(input.hermesTools, session.id)
        } catch (error) {
          return context.error("invalid_state", String((error as Error)?.message ?? error), {})
        }
        session.nativeTools = nativeTools
        session.generation = input.generation
        session.settled = false
        session.rejectReason = undefined
        // New primary request: no step/message state from the previous
        // generation may decide which assistant message is primary.
        session.primaryAssistant = undefined
        session.calls.clear()
        session.names.clear()
        session.stepGeneration.clear()
        session.outcome = undefined
        session.activeRequest = undefined
        session.snapshot = input.snapshot
        session.systemParts = systemParts
        if (typeof input.userText === "string") session.userText = input.userText
        session.authorizations.push(input.generation)
        emit(session, "advanced", { generation: input.generation })
        // An explicit advance releases a parked execute result (or a denied
        // program) exactly once. That release, not the program finishing, is
        // what lets the model see it and what authorizes the next request.
        for (const [callID, decision] of session.decisions) {
          if (decision.allow) continue
          session.decisions.delete(callID)
          const gate = session.gates.get(callID)
          session.gates.delete(callID)
          if (!gate) continue
          session.released.add(callID)
          emit(session, "execute_released", { callID, allowed: false })
          gate.reject(new Error(decision.reason))
        }
        for (const [callID, gate] of session.after) {
          session.after.delete(callID)
          session.released.add(callID)
          emit(session, "execute_released", { callID, allowed: true })
          gate.resolve()
        }
        // Every canonical generation is one new prompt. The context hook
        // replaces the transcript, so OpenCode's retained history is never
        // replayed as context.
        session.started = true
        session.promptRunning = true
        const generation = input.generation
        void ctx.session
          .prompt({ sessionID: session.id, text: session.userText ?? "hermes-generation" })
          .then(async () => {
            if (session.generation === generation) emit(session, "prompt_settled", { generation })
            // Admission is not completion. Wait for the run to reach idle; a
            // tool-parked run stays active until `finish` rejects its executors.
            try {
              await ctx.session.wait({ sessionID: session.id })
            } catch (error) {
              if (session.generation === generation) session.promptError = String(error)
            }
            if (session.generation === generation) {
              session.promptRunning = false
              emit(session, "prompt_idle", { generation })
            }
          })
          .catch((error: unknown) => {
            // A late rejection from a prior generation must not poison the
            // current one. Report it as diagnostic only.
            if (session.generation !== generation) {
              emit(session, "prompt_error", { message: String(error), generation, stale: true })
              return
            }
            session.promptError = String(error)
            session.promptRunning = false
            emit(session, "prompt_error", { message: String(error), generation })
          })
        return {
          generation: input.generation,
          authorized: true,
          seq: session.seq,
          session: describe(session),
        }
      },

      // Explicit generation boundary. Discards parked executor results, stops
      // any in-flight inference, and keeps the native session for the next
      // canonical snapshot.
      finish: async (input: any, context: any) => {
        const id = String(input.sessionID)
        if (closedSessions.has(id)) return { settled: true, interrupted: false, seq: 0, skipped: "closed" }
        const session = sessionOf(id)
        if (session.closed) return context.error("invalid_state", "session closed", { sessionID: session.id })
        session.rejectReason = "generation finished"
        rejectPending(session, "generation finished")
        for (const [callID, gate] of session.gates) {
          session.gates.delete(callID)
          gate.reject(new Error("generation finished"))
        }
        for (const [callID, gate] of session.after) {
          session.after.delete(callID)
          session.released.add(callID)
          gate.resolve()
        }
        session.decisions.clear()
        let interrupted = false
        const wasActive = session.promptRunning || session.activeRequest !== undefined
        if (wasActive) {
          try {
            interrupted = Boolean(await ctx.session.interrupt({ sessionID: session.id }))
          } catch {
            // already idle or removed by teardown; the bounded wait decides
          }
        }
        // Admission is not completion. Wait for the run to leave the active
        // set; rejecting the parked executors above lets a tool-parked run end.
        let timedOut = false
        let waitFailed = false
        await Promise.race([
          ctx.session.wait({ sessionID: session.id }).catch(() => {
            waitFailed = true
          }),
          new Promise<void>((resolve) => {
            setTimeout(() => {
              timedOut = true
              resolve()
            }, 10_000)
          }),
        ])
        const settled = !timedOut && !waitFailed
        session.promptRunning = false
        session.settled = settled
        session.running = undefined
        session.activeRequest = undefined
        session.outcome = undefined
        session.primaryAssistant = undefined
        session.authorizations.length = 0
        session.calls.clear()
        session.names.clear()
        session.rejectReason = undefined
        emit(session, "finished", { settled, interrupted })
        return { settled, interrupted, seq: session.seq, session: describe(session) }
      },

      // Kept for protocol compatibility, but never authorizes JavaScript. With
      // execute.before hard-refusing there is no gate to release; a stale or
      // confused caller gets the same explicit unsupported result.
      decide: async (input: any, context: any) => {
        const id = String(input.sessionID)
        if (closedSessions.has(id)) return context.error("invalid_state", "session closed", { sessionID: id })
        const session = sessionOf(id)
        if (session.closed) return context.error("invalid_state", "session closed", { sessionID: session.id })
        return context.error("invalid_state", EXECUTE_UNSUPPORTED, {
          sessionID: session.id,
          callID: String(input.callID),
        })
      },

      results: async (input: any, context: any) => await settleBatch(input, context),

      settle: async (input: any, context: any) => await settleBatch(input, context),

      read: async (input: any, context: any) => {
        const id = String(input.sessionID)
        const cursor = Number(input.cursor ?? 0)
        if (closedSessions.has(id)) return { cursor, events: [], state: closedState(id) }
        const session = sessionOf(id)
        let waited = false
        while (true) {
          const events = session.events.filter((event) => event.seq > cursor)
          if (events.length > 0) return { cursor: events[events.length - 1]!.seq, events, state: describe(session) }
          if (session.closed || waited || context.signal?.aborted)
            return { cursor, events: [], state: describe(session) }
          waited = true
          // Bounded wait: the RPC must return before the HTTP client gives up,
          // otherwise a timed-out caller leaves a dangling waiter per poll.
          await new Promise<void>((resolve) => {
            const abort = () => wake()
            const wake = () => {
              session.waiters.delete(wake)
              context.signal?.removeEventListener?.("abort", abort)
              clearTimeout(timer)
              resolve()
            }
            const timer = setTimeout(wake, 900)
            session.waiters.add(wake)
            context.signal?.addEventListener?.("abort", abort, { once: true })
          })
        }
      },

      state: async (input: any) => {
        const id = String(input.sessionID)
        if (closedSessions.has(id)) return closedState(id)
        return describe(sessionOf(id))
      },

      close: async (input: any) => {
        const id = String(input.sessionID)
        if (closedSessions.has(id)) return { closed: true, skipped: "already closed", session: closedState(id) }
        const session = sessionOf(id)
        // Visible before any await: later RPCs see a closed session even if
        // interrupt/remove hangs.
        session.closed = true
        session.rejectReason = "session closed"
        session.promptRunning = false
        rejectPending(session, "session closed")
        for (const [callID, gate] of session.gates) {
          session.gates.delete(callID)
          gate.reject(new Error("session closed"))
        }
        for (const [callID, gate] of session.after) {
          session.after.delete(callID)
          session.released.add(callID)
          gate.resolve()
        }
        session.decisions.clear()
        try {
          await ctx.session.interrupt({ sessionID: session.id })
        } catch {
          // idle session: nothing to interrupt
        }
        try {
          await ctx.session.remove({ sessionID: session.id })
        } catch {
          // already gone
        }
        emit(session, "closed", {})
        // Drop every generation-sized structure before forgetting the session;
        // a bounded tombstone keeps idempotence without retaining transcripts.
        session.snapshot = []
        session.systemParts = []
        session.pending.clear()
        session.gates.clear()
        session.after.clear()
        session.decisions.clear()
        session.executing.clear()
        session.released.clear()
        session.calls.clear()
        session.names.clear()
        session.stepGeneration.clear()
        session.events = []
        session.waiters.clear()
        session.primaryAssistant = undefined
        session.outcome = undefined
        session.activeRequest = undefined
        sessions.delete(session.id)
        rememberClosed(session.id)
        return { closed: true, session: closedState(session.id) }
      },
    })
  },
})
