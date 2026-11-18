# Changelog

## Unreleased

### Fixed
- Interrupts and stale-call aborts now stop an in-flight request: the client
  kills its Bun process from `cancel()`/`close()`, including while it waits for
  the upstream status line, and during startup (a Bun spawned for a cancelled
  request is refused before it receives the request). A truncated stream raises instead of ending as a
  complete reply. Idea from pasei-dev `ac2ad68`.
- Reasoning-token usage (`completion_tokens_details.reasoning_tokens`, and
  `output_tokens_details` on the Responses route) is preserved. Idea from
  pasei-dev `62e12d6`, without its request-body debug dump.
- Synchronous non-streaming `create()` performs the request immediately, so it
  runs inside Hermes' interruptible worker and latency measurement. Under an
  event loop it stays awaitable and runs off the loop. Idea from pasei-dev
  `9841760`.

### Added
- `get_model_context_length` from the discovered model's models.dev row.
  Idea from pasei-dev `e31b61d`.
- Top-level `reasoning_effort` for chat-route models whose models.dev row
  declares effort levels, clamped down to those levels and omitted rather
  than raised. Nothing is sent for unknown or toggle-only models. Idea from pasei-dev `29ddb25`.
