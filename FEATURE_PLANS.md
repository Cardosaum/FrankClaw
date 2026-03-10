# FrankClaw Feature Plans

These are the major OpenClaw feature families that FrankClaw does not implement yet.
Each plan is intentionally standalone so we can execute one at a time without destabilizing the current `web` + `telegram` core.

## Planning Rules

- Keep the existing runtime path as the single execution path.
- Add one feature family at a time behind explicit config gates.
- Do not widen trust boundaries before the storage, auth, and audit model for that feature exists.
- Every feature family needs unit tests first, then at least one integration or e2e path before claiming parity.
- If a feature would reintroduce OpenClaw-style sprawl, shrink the scope before building it.

## Recommended Order

1. Integration and e2e coverage for the current core
2. Additional channels
3. Rich channel behavior
4. Web control UI and WebChat
5. Tools runtime
6. Skills and plugin system
7. Webhooks and broader gateway control plane
8. Tailscale and remote-access ops
9. Companion nodes and apps
10. Voice
11. Canvas

## 1. Additional Channels

Goal:
Add high-value messaging channels beyond `web` and `telegram` without changing the core runtime contract.

Initial target set:
- `discord`
- `slack`
- `signal` or `whatsapp`, but not both in the same pass

Do not include in this plan:
- node/device channels
- voice transport
- channel-specific tool actions

Dependencies:
- Current runtime stays the single message execution path
- Delivery metadata schema stays stable across channels
- Integration test harness for channel adapters

Phases:
1. Define a channel capability matrix and security profile per channel.
2. Implement one adapter at a time in `frankclaw-channels`.
3. Add channel-specific config validation in `frankclaw-core`.
4. Add ingress policy defaults: DM policy, mention gating, reply threading, payload caps.
5. Add adapter integration tests with mocked upstream APIs.

Security constraints:
- Default new channels to `dm_policy = pairing`
- Cap inbound and outbound payload sizes per channel
- Avoid permissive markup or rich text until escaping is correct
- Persist account/thread/reply identifiers needed for retries and edits

Acceptance:
- A message received on the new channel gets a model reply through the shared runtime
- Session scoping remains isolated per sender or thread
- Pairing and allowlist rules still apply
- Channel-specific retries behave predictably under rate limiting

## 2. Rich Channel Behavior

Goal:
Bring the supported channels closer to OpenClaw’s routing and delivery behavior without adding more channel count.

Scope:
- Telegram edit support
- reply tags and thread-aware replies
- chunking and split-delivery policy
- richer group routing rules
- streaming or pseudo-streaming where the channel supports it

Dependencies:
- Delivery metadata persisted per reply
- Stable outbound retry logic
- Channel capability flags enforced by the gateway

Phases:
1. Persist outbound context needed for edit/delete/retry.
2. Add channel-specific edit implementations starting with Telegram.
3. Add chunking policy in one shared delivery layer, not per handler.
4. Add group routing config and tests for mention, reply-tag, and thread modes.
5. Add streaming only where the adapter can edit in place safely.

Security constraints:
- Never stream or edit into a channel/thread different from the recorded origin
- Treat stored reply metadata as sensitive session state
- Reject edits when platform context is missing or stale

Acceptance:
- Telegram edit-in-place works for tracked replies
- Long outputs are chunked predictably
- Group routing remains deny-by-default unless explicitly enabled

## 3. Web Control UI and WebChat

Goal:
Add a first-party browser UI for control-plane operations and chat without weakening the gateway’s local-first posture.

Scope:
- WebChat
- basic admin/control UI
- session browsing
- pairing review
- model/channel health views

Do not include:
- public internet exposure by default
- broad config editing UI in the first pass

Dependencies:
- Stable WS methods for chat, sessions, channels, models
- Auth story for browser clients
- Server-side event streaming already in place

Phases:
1. Define a minimal UI surface and auth model.
2. Add static asset serving or a separate bundled frontend.
3. Build WebChat first, then add read-only operator screens.
4. Add pairing approval and session inspection actions.
5. Add limited config editing only after validation and audit hooks exist.

Security constraints:
- Keep loopback bind as default
- Require explicit auth for browser sessions
- Redact secrets from all browser-visible payloads
- CSRF and origin checks if cookie auth is ever introduced

Acceptance:
- A local browser can chat, inspect sessions, inspect models/channels, and approve pairings
- No secrets or raw provider keys are exposed to the UI

## 4. Tools Runtime

Goal:
Add a hardened tool runtime instead of copying OpenClaw’s broad tool surface directly into the gateway.

Scope:
- start with read-only or low-risk tools
- explicit tool policies per agent
- one execution runtime for all tools

Defer:
- browser automation
- exec
- device/node actions
- channel-side action tools

Dependencies:
- Agent policy model in config
- Structured audit logging
- Sandboxed execution strategy

Phases:
1. Define tool policy types and allowlists in `frankclaw-core`.
2. Build a dedicated tool runtime crate with a narrow host API.
3. Add safe read-only tools first, such as session inspection.
4. Add approval and audit hooks.
5. Add higher-risk tools only after sandboxing and operator controls are proven.

Security constraints:
- Tools default to disabled
- Every tool call must be attributable in audit logs
- No direct shell or browser execution in the first pass
- No tool inherits broader permissions from the gateway process by default

Acceptance:
- Tools can be enabled per agent with explicit policy
- Tool calls execute outside the gateway request handler and are fully auditable

## 5. Skills and Plugin System

Goal:
Replace the current placeholder plugin SDK with a constrained extension model that does not undermine trust boundaries.

Scope:
- skill discovery and packaging
- plugin manifest validation
- capability restrictions
- workspace-local skills first

Defer:
- remote marketplace
- automatic install/update
- arbitrary code-loading without process isolation

Dependencies:
- Tool runtime
- Stable config model for agent capabilities
- Clear extension trust model

Phases:
1. Define skill/plugin manifests and capability declarations.
2. Add loader and validator logic.
3. Support local workspace skills first.
4. Add explicit enable/disable and audit visibility.
5. Add distribution or registry support only after local isolation is solid.

Security constraints:
- Extensions are opt-in and disabled by default
- Every extension declares capabilities up front
- Validation must fail closed on malformed manifests
- No extension gets network, filesystem, or exec privileges implicitly

Acceptance:
- A workspace skill can be installed, validated, enabled for one agent, and observed in audit logs

## 6. Webhooks and Broader Gateway Control Plane

Goal:
Fill in the larger OpenClaw gateway surface without allowing new inputs to bypass the runtime or security model.

Scope:
- webhook ingestion
- more WS admin methods
- presence and typing indicators
- richer session mutation APIs

Dependencies:
- Current WS methods remain stable
- Audit logging exists
- Shared runtime path remains the only execution path

Phases:
1. Add signed webhooks with strict size and replay limits.
2. Add read-only gateway methods first.
3. Add mutation methods only with auth-role checks and audit logs.
4. Add presence/typing as event-only surfaces.
5. Add admin mutation paths gradually, not as one large protocol drop.

Security constraints:
- Signed or token-authenticated webhooks only
- No webhook may inject directly into arbitrary sessions without policy checks
- Presence events must not leak private channel metadata across clients

Acceptance:
- Webhooks, presence, and session/admin methods work without bypassing auth or runtime policy

## 7. Tailscale and Remote Access Ops

Goal:
Support remote access patterns similar to OpenClaw while keeping local-first defaults and refusing dangerous exposure modes.

Scope:
- Tailscale identity-header mode
- optional serve/funnel helpers
- remote gateway operator guidance

Defer:
- fully automatic network reconfiguration in the first pass
- public exposure defaults

Dependencies:
- Trusted proxy and Tailscale auth modes already exist
- Browser UI or remote operator surface is useful enough to justify exposure

Phases:
1. Add config validation for remote-access modes.
2. Add explicit operator commands for checking remote exposure state.
3. Support tailnet-only access before any public mode.
4. Add public exposure only with mandatory password/token auth and explicit confirmation.

Security constraints:
- Loopback remains the default bind mode
- Tailnet/public exposure must refuse startup without valid auth
- Never silently change network exposure on behalf of the operator

Acceptance:
- Operators can expose the gateway remotely in a bounded, auditable way without weakening default local security

## 8. Companion Nodes and Apps

Goal:
Add device-local execution surfaces only after the gateway and tool trust boundaries are mature.

Scope:
- node pairing
- node inventory and capability discovery
- device-local actions on a paired node
- later, companion apps

Defer:
- full macOS/iOS/Android product surfaces in one pass
- arbitrary remote exec on nodes

Dependencies:
- Pairing model for device trust
- Tool runtime
- Node RPC protocol and capability model

Phases:
1. Define node identity, pairing, and capability advertisement.
2. Add a minimal node RPC protocol.
3. Build a simple reference node before full apps.
4. Add one safe device action family first.
5. Expand only after stable pairing, audit, and revocation exist.

Security constraints:
- Node trust is separate from DM pairing
- Every node action requires capability checks
- Node revocation must be immediate and durable
- Device-local actions must never silently fall back to gateway-host execution

Acceptance:
- A paired node can advertise capabilities and perform one bounded action family with full audit trails

## 9. Voice

Goal:
Add speech input and output without turning the gateway into an always-on ambient surveillance surface by default.

Scope:
- speech-to-text pipeline
- text-to-speech output
- push-to-talk or explicit activation first

Defer:
- wake-word detection
- continuous ambient listening
- cross-device voice handoff

Dependencies:
- Node/app surfaces or a local client that can capture audio
- Media pipeline for audio blobs
- Session routing for voice-originated turns

Phases:
1. Add audio ingestion and transcription.
2. Add TTS output for explicit chat responses.
3. Add push-to-talk flow.
4. Add wake-word or talk mode only after privacy controls are proven.

Security constraints:
- Voice capture must be explicit and user-driven in the first pass
- Audio retention must be bounded and configurable
- Transcript and media storage rules apply equally to audio

Acceptance:
- A user can submit spoken input, get a transcript-backed assistant reply, and optionally receive TTS output

## 10. Canvas

Goal:
Add a visual workspace only after tools, node surfaces, and web UI are stable enough to support it safely.

Scope:
- render-only surface first
- structured UI updates from trusted internal components
- no arbitrary code-eval canvas in the first pass

Dependencies:
- Web UI or companion app surface
- Tool runtime and policy model
- Event protocol extensions

Phases:
1. Define a narrow canvas state model.
2. Add server-push events for canvas updates.
3. Build a simple read-only or controlled-render canvas.
4. Add limited interactive actions.
5. Consider A2UI-style richer behaviors only after policy and sandboxing are stable.

Security constraints:
- No arbitrary JS or code execution through canvas payloads
- Canvas updates must be attributable to a session or trusted subsystem
- Sensitive data shown in canvas must follow the same auth boundaries as chat

Acceptance:
- The assistant can render and update a bounded visual workspace without introducing an untrusted code channel

## 11. Onboarding, Packaging, and Operator Breadth

Goal:
Improve operator experience without rebuilding all of OpenClaw’s installation surface too early.

Scope:
- onboarding helper
- config bootstrap
- richer `doctor`
- status/update commands
- packaging guidance

Dependencies:
- Stable config schema
- Stable runtime and supported feature set

Phases:
1. Expand `doctor` into a real environment and config verifier.
2. Add `status` and richer diagnostics.
3. Add a minimal onboarding helper for the FrankClaw-supported scope.
4. Add packaging and service-install documentation or commands.
5. Add update-channel logic only if release engineering actually needs it.

Security constraints:
- Bootstrap flows must not write insecure defaults
- Generated configs must default to loopback, auth on non-loopback, and encrypted sessions

Acceptance:
- A new operator can install, validate, configure, and run the supported FrankClaw scope without manual spelunking

## Cross-Cutting Work That Should Happen Before or Alongside Any Major Plan

- Integration and e2e coverage for the current `web` and `telegram` core
- Transcript and reply metadata schema versioning
- Better test fixtures for mocked providers and mocked channels
- Audit log verification tests
- Performance and backpressure tests on inbound/outbound queues
