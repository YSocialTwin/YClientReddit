# External Memory Package Pipeline

## Goal

This document defines a detailed pipeline to evolve the current memory implementation into an **external Python package** that can be imported by both:

- YClientReddit
- YClient

The end state is:

- one reusable package
- one standardized public API
- multiple pluggable memory backends
- no regression in current YClientReddit behavior by default
- a migration path that lets both clients adopt the package incrementally

This pipeline builds directly on the prior analysis in:

- [docs/agent_memory_report.md](/Users/rossetti/PycharmProjects/YClientReddit/docs/agent_memory_report.md)
- [docs/agent_memory_parametric_refactor_report.md](/Users/rossetti/PycharmProjects/YClientReddit/docs/agent_memory_parametric_refactor_report.md)

## Target Outcome

The target package should provide:

- a stable import surface
- a standard engine interface
- shared data contracts
- capability-based backend selection
- reusable policies for safety and prompt shaping
- backend implementations such as:
  - current hybrid semantic memory
  - simple recent-interaction memory

Illustrative target usage:

```python
from yclient_memory import build_memory_engine

memory = build_memory_engine(
    backend="hybrid_semantic",
    config=memory_config,
    runtime=runtime_adapter,
)
```

The package must be importable equivalently by both clients, meaning:

- same public API
- same request/response types
- same backend names
- same capability model

## Strategic Principles

### 1. Preserve current behavior first

YClientReddit’s current behavior is the reference implementation for the first external package release.

### 2. Package the abstraction, not the current file layout

The package should not export `base_agent.py`-style helpers directly. It should expose:

- contracts
- engine interface
- factory
- policies
- supported backends

### 3. Separate product policies from backend implementation

Shared safety and prompting rules must live in the package as reusable policies, not be reimplemented per client.

### 4. Keep client-specific concerns outside the package

The package should not depend directly on:

- YClientReddit `Agent`
- YClient `Agent`
- specific database models in either repo
- direct thread/post storage access in either repo

Those should be accessed through adapters.

### 5. Support incremental adoption

Both clients should be able to adopt the package gradually.

## Recommended Package Shape

Suggested external package name:

- `yclient-memory`

Suggested Python import:

- `yclient_memory`

Suggested structure:

```text
yclient_memory/
  pyproject.toml
  README.md
  src/yclient_memory/
    __init__.py
    factory.py
    contracts.py
    capabilities.py
    errors.py
    engine.py
    config.py
    runtime.py
    logging.py
    policies/
      __init__.py
      prompt_policy.py
      safety_policy.py
      affect_policy.py
      decay_policy.py
    backends/
      __init__.py
      hybrid_semantic/
        __init__.py
        engine.py
        api_client.py
        llm_helpers.py
        reflection.py
      simple_recent/
        __init__.py
        engine.py
    tests/
      unit/
      integration/
      contract/
      fixtures/
```

## Public API Recommendation

The public API should be intentionally small.

### Public exports

- `build_memory_engine`
- `MemoryEngine`
- `MemoryCapabilities`
- request/response contracts
- package version

### Example public imports

```python
from yclient_memory import build_memory_engine
from yclient_memory import MemoryCapabilities
from yclient_memory.contracts import ReplyMemoryRequest, ReplyMemoryContext
```

## Standardized External Signature

To satisfy the requirement of “multiple memory systems having the same external signature,” define a stable engine protocol.

Recommended top-level protocol:

```python
class MemoryEngine(Protocol):
    def capabilities(self) -> MemoryCapabilities: ...

    def build_reply_context(self, request: ReplyMemoryRequest) -> ReplyMemoryContext: ...
    def build_browse_context(self, request: BrowseMemoryRequest) -> BrowseMemoryContext: ...
    def build_post_style_context(self, request: PostStyleRequest) -> PostStyleContext: ...

    def record_comment(self, event: CommentMemoryEvent) -> None: ...
    def record_vote(self, event: VoteMemoryEvent) -> None: ...
    def record_post(self, event: PostMemoryEvent) -> None: ...

    def relationship_signal(self, request: RelationshipSignalRequest) -> RelationshipSignal: ...

    def maintenance_tick(self, request: MaintenanceTickRequest) -> MaintenanceTickResult: ...
```

The key point is:

- every backend exposes the same methods
- richer backends advertise more capabilities
- simpler backends return empty/default outputs where appropriate

## Standard Contracts

The package should define shared typed contracts for both clients.

### Read-side contracts

- `ReplyMemoryRequest`
- `ReplyMemoryContext`
- `BrowseMemoryRequest`
- `BrowseMemoryContext`
- `PostStyleRequest`
- `PostStyleContext`

### Write-side contracts

- `CommentMemoryEvent`
- `VoteMemoryEvent`
- `PostMemoryEvent`
- `MaintenanceTickRequest`
- `MaintenanceTickResult`

### Shared analytical contracts

- `RelationshipSignalRequest`
- `RelationshipSignal`
- `MemoryCueSet`
- `MemoryCapabilities`
- `MemoryDiagnostics`

### Important rule

Contracts should contain both:

- structured fields
- optional rendered text fields

This prevents the package from becoming too prompt-format-specific while still supporting current prompt flows.

## Runtime Adapter Layer

The package should not know how YClientReddit or YClient store or fetch their own domain data.

Instead, introduce a runtime adapter protocol.

Example responsibilities:

- get author info for a post/comment
- get thread root id
- get recent root posts
- fetch raw post text
- call LLM with a prompt and config
- log decision events
- return persona snapshot
- expose round id and user id context

Example conceptual interface:

```python
class MemoryRuntime(Protocol):
    def llm_json(self, prompt_key: str, variables: dict, config: dict | None = None) -> dict | list | str: ...
    def llm_text(self, prompt_key: str, variables: dict, config: dict | None = None) -> str: ...

    def get_author_id_and_username(self, post_id: int) -> tuple[int | None, str | None]: ...
    def get_thread_root_id(self, post_id: int) -> int | None: ...
    def get_recent_root_posts(self, round_id: int, limit: int, rounds_back: int) -> list[dict]: ...
    def get_post_text(self, post_id: int) -> str: ...

    def persona_snapshot(self) -> dict: ...
    def decision_log(self, payload: dict) -> None: ...
```

This layer is what makes one package reusable across both clients.

## Why the runtime adapter matters

Without it, the package would hard-code assumptions about:

- agent object shape
- thread model
- post APIs
- logging format
- LLM invocation

That would make it impossible to reuse cleanly between YClientReddit and YClient.

## Backend Model

The external package should launch with two backends.

### Backend 1: `hybrid_semantic`

This is the packaged form of the current YClientReddit memory system.

Responsibilities:

- structured context fetch
- semantic retrieval
- social cards
- thread cards
- community digest
- reflection synthesis
- optional high-affect recall

This backend should preserve current YClientReddit behavior as closely as possible.

### Backend 2: `simple_recent`

This is the alternative low-complexity backend.

Responsibilities:

- recent per-user interaction history
- recent per-thread history
- simple relationship counters
- deterministic reply/browse context
- no reflections
- no semantic search
- no community digest LLM synthesis

This backend gives both clients a low-dependency option.

## Package Boundary Decisions

These decisions are important before implementation starts.

### Keep inside the package

- contracts
- engine interface
- backend implementations
- policy helpers
- capability model
- safety logic
- common formatting logic for memory cues

### Keep outside the package

- client-specific `Agent` classes
- client-specific prompt catalogs unless standardized
- direct DB models of either client
- direct HTTP clients unrelated to memory
- content-generation orchestration not specifically about memory

### Optional standardization choice

If YClientReddit and YClient both use prompt-driven memory planning, prompt keys and variables can also be standardized inside the package.

If not, keep prompt execution in the runtime adapter and let the package request semantic operations by intent.

## Recommended Packaging Strategy

Use a dedicated repository for the package rather than embedding it permanently inside one client repo.

Recommended:

- new repo: `yclient-memory`

Reasons:

- cleaner release/versioning
- shared CI
- shared ownership
- easier dependency management across clients

Intermediate step:

- build the package in-tree first
- extract to a dedicated repo once contracts stabilize

This reduces early coordination cost.

## Delivery Pipeline

The recommended pipeline is a multi-stage migration.

## Stage 0: Alignment and scope lock

Duration:

- 2 to 4 days

Goals:

- align both client teams on package boundary
- decide initial supported backends
- agree on non-regression expectations
- decide whether prompts are standardized inside or outside the package

Deliverables:

- package charter
- ownership model
- API scope v1
- list of shared runtime adapter responsibilities

Exit criteria:

- both clients accept the external package direction
- v1 interface scope is frozen

## Stage 1: Behavior inventory and contract extraction

Duration:

- 4 to 6 days

Goals:

- inventory current memory-driven behaviors in YClientReddit
- identify likely shared behaviors needed by YClient
- define typed contracts and capabilities

Tasks:

1. Build a method-to-responsibility matrix from current `_memory_*` logic.
2. Classify each behavior as:
   - backend concern
   - shared policy
   - runtime adapter concern
   - client orchestration concern
3. Write contract definitions.
4. Write capability flags.

Deliverables:

- `contracts.py`
- `capabilities.py`
- interface design doc

Exit criteria:

- contracts cover all current YClientReddit memory use cases
- no critical current feature lacks a representation

## Stage 2: Characterization test harness

Duration:

- 4 to 7 days

Goals:

- freeze current YClientReddit behavior before extraction
- define comparable scenarios for future parity checks

Tasks:

1. Add scenario fixtures for:
   - reply context in subtle mode
   - reply context in legacy mode
   - vote-signal-only flow
   - browse context
   - post style context
   - relationship signal
   - reflection trigger cadence
2. Capture expected outputs or expected invariants.
3. Add decision-log-based comparison tests where exact text may vary.

Deliverables:

- golden fixtures
- scenario tests
- regression harness

Exit criteria:

- the current implementation is measurably characterized

## Stage 3: In-tree package skeleton

Duration:

- 3 to 5 days

Goals:

- create the package structure without moving logic yet

Tasks:

1. Create `yclient_memory` package skeleton in-tree.
2. Add `build_memory_engine`.
3. Add empty backend modules.
4. Add runtime adapter interfaces.
5. Add package-level tests for contracts and capability objects.

Deliverables:

- importable package skeleton
- no production behavior change

Exit criteria:

- YClientReddit can import the package locally

## Stage 4: Wrap current YClientReddit memory as `hybrid_semantic`

Duration:

- 1 to 2 weeks

Goals:

- preserve current behavior through the new package interface

Tasks:

1. Move or wrap current memory read-path logic into `hybrid_semantic`.
2. Move or wrap current memory write-path logic into `hybrid_semantic`.
3. Move shared helper logic into policies where appropriate.
4. Implement a YClientReddit runtime adapter.
5. Replace direct agent memory calls with package calls.

Migration rule:

- do not simplify behavior to fit the abstraction

Deliverables:

- working `HybridSemanticMemoryEngine`
- YClientReddit runtime adapter
- green characterization tests

Exit criteria:

- YClientReddit behaves equivalently with package-backed memory

## Stage 5: Extract shared policies

Duration:

- 4 to 6 days

Goals:

- remove backend-independent logic from client code and backend code

Candidates to move:

- vote artifact filtering
- prompt memory sanitization
- memory cue generation
- cue formatting
- callback verification
- decay helper logic
- safety guardrails against invented continuity

Deliverables:

- reusable policy modules

Exit criteria:

- `hybrid_semantic` backend gets simpler
- YClientReddit agent orchestration gets thinner

## Stage 6: Implement `simple_recent`

Duration:

- 4 to 7 days

Goals:

- provide a second backend with identical external signature

Tasks:

1. Implement recent interaction stores.
2. Implement simple relationship scoring.
3. Implement reply/browse context builders.
4. Implement post-style behavior as empty or heuristic-only.
5. Advertise limited capabilities.

Deliverables:

- `SimpleRecentMemoryEngine`
- backend-specific tests

Exit criteria:

- simple backend works end-to-end in YClientReddit through the same package API

## Stage 7: Package externalization

Duration:

- 3 to 5 days

Goals:

- move from in-tree package to dedicated external package repository

Tasks:

1. Create dedicated repo.
2. Move package code and tests.
3. Add `pyproject.toml`.
4. Add versioning policy.
5. Add release automation.
6. Publish internal artifact or private package.

Deliverables:

- standalone package repo
- installable package artifact

Exit criteria:

- YClientReddit consumes the package via dependency, not local import path hacks

## Stage 8: Integrate into YClient

Duration:

- 1 to 2 weeks depending on YClient architecture

Goals:

- adopt the same package in YClient with a client-specific runtime adapter

Tasks:

1. Implement YClient runtime adapter.
2. Map YClient’s thread/post/user concepts to shared contracts.
3. Integrate package-backed memory calls.
4. Add YClient-specific integration tests.
5. Validate capability compatibility.

Deliverables:

- YClient adapter
- YClient integration coverage

Exit criteria:

- YClient imports and runs the same package successfully

## Stage 9: Cross-client parity and backend compatibility matrix

Duration:

- 4 to 6 days

Goals:

- make backend behavior explicit across both clients

Tasks:

1. Build a compatibility matrix:
   - backend vs client vs feature
2. Mark supported, degraded, unsupported flows.
3. Validate import/API stability.
4. Validate configuration compatibility.

Deliverables:

- compatibility matrix
- support policy for backends

Exit criteria:

- both clients know which backends support which behaviors

## Stage 10: Stabilization and v1 release

Duration:

- 3 to 5 days

Goals:

- finalize package v1

Tasks:

1. Freeze public API.
2. Add migration guides for both clients.
3. Add changelog and versioned release notes.
4. Tag `v1.0.0` when parity is acceptable.

Deliverables:

- versioned external package
- migration documentation
- release checklist

Exit criteria:

- package is ready for normal shared consumption

## Detailed Workstreams

## Workstream A: Contract design

Objectives:

- define stable data shapes
- minimize future breaking changes

Key tasks:

- use typed dataclasses or pydantic models
- include diagnostics/meta fields
- define clear null/empty behavior
- document capability-dependent fields

Success criteria:

- both clients can consume outputs without backend-specific branches

## Workstream B: Runtime adapters

Objectives:

- isolate all client-specific behavior

Key tasks:

- define adapter protocol
- implement Reddit adapter first
- implement YClient adapter second
- keep adapter responsibilities narrow

Success criteria:

- memory package has no direct dependency on either client’s agent class internals

## Workstream C: Shared policy extraction

Objectives:

- avoid safety drift across backends and clients

Key tasks:

- centralize vote filtering
- centralize memory prompt safety rules
- centralize callback verification
- centralize decay helpers

Success criteria:

- the same safety behavior applies regardless of backend or client

## Workstream D: Backend implementation

Objectives:

- provide multiple implementations behind one API

Key tasks:

- hybrid semantic backend parity
- simple recent backend correctness
- capability advertising

Success criteria:

- backends are swappable by config, not by code changes

## Workstream E: Distribution and release

Objectives:

- make the package consumable in both repos

Key tasks:

- package build configuration
- dependency policy
- semantic versioning
- private or internal package registry setup

Success criteria:

- both clients install the same versioned package artifact

## Configuration Strategy

The package should support a stable config schema with backend-specific options.

Suggested top-level config:

```json
{
  "memory_enabled": true,
  "memory_backend": "hybrid_semantic",
  "memory_backend_fallback": "simple_recent",
  "memory_backend_options": {
    "hybrid_semantic": {
      "prompt_mode": "subtle_forum",
      "vote_signal_only": true
    },
    "simple_recent": {
      "pair_history_limit": 8,
      "thread_history_limit": 12
    }
  }
}
```

Important rule:

- keep current YClientReddit config keys supported through an adapter or translation layer

This avoids forcing a simultaneous config migration.

## Testing Pipeline

This migration needs more than normal unit tests.

## 1. Contract tests

Validate:

- all backends return the same contract types
- required fields are always present
- capability flags are coherent

## 2. Characterization tests

Validate:

- package-backed hybrid backend matches current YClientReddit behavior

## 3. Backend tests

Validate:

- hybrid semantic logic
- simple recent logic

## 4. Runtime adapter tests

Validate:

- Reddit adapter correctness
- YClient adapter correctness

## 5. Cross-client integration tests

Validate:

- same package version works in both clients

## 6. Compatibility matrix tests

Validate:

- unsupported capabilities fail safely

## 7. Regression tests

Priority regression areas:

- subtle forum memory injection
- vote-signal-only suppression
- browse relationship prioritization
- high-affect callback behavior where enabled
- digest/reflection cadence behavior

## Versioning and Release Model

Recommended:

- semantic versioning
- public API freeze at `1.0.0`
- no breaking contract changes without major version bump

Suggested initial release phases:

- `0.1.x`: in-tree experimental
- `0.2.x`: external package, Reddit only
- `0.3.x`: Reddit stable + simple backend
- `0.4.x`: YClient adapter integrated
- `1.0.0`: both clients supported under stable API

## Migration Strategy for YClientReddit

Recommended sequence:

1. add package as in-tree module
2. wrap current backend
3. replace direct memory calls in `Agent`
4. prove parity
5. externalize package repo
6. switch dependency to packaged artifact

This minimizes disruption.

## Migration Strategy for YClient

Recommended sequence:

1. import contracts and engine factory
2. implement runtime adapter
3. use `simple_recent` first if YClient needs lower adoption risk
4. later adopt `hybrid_semantic` only if the client supports the required services

This reduces coordination risk because YClient can start with the simpler backend.

## Main Risks

## Risk 1: hidden coupling to YClientReddit internals

The current logic is heavily bound to `Agent`.

Mitigation:

- runtime adapter
- contract-first extraction

## Risk 2: package API too Reddit-shaped

If the contracts encode Reddit-specific assumptions, YClient reuse will be awkward.

Mitigation:

- use generic names like `thread_root_id`, `target_post_id`, `actor_user_id`
- avoid Reddit-only terminology in core contracts

## Risk 3: package API too abstract

If the interface is too generic, preserving the hybrid backend becomes hard.

Mitigation:

- capability flags
- diagnostics/meta fields
- optional structured sub-objects

## Risk 4: regressions in current behavior

Mitigation:

- characterization tests first
- hybrid backend parity gate
- no default backend change

## Risk 5: prompt and LLM behavior drift across clients

Mitigation:

- standardize memory-specific prompt intents
- keep prompt rendering in runtime adapters where necessary

## Risk 6: release coordination burden

Mitigation:

- versioned package
- explicit compatibility matrix
- clear deprecation policy

## Success Criteria

The migration should be considered successful only if all of the following are true.

### Technical success

- the package is installable independently
- both YClientReddit and YClient import the same package
- both use the same external memory signature
- multiple backends are selectable by config

### Behavioral success

- YClientReddit hybrid backend shows no meaningful regressions
- simple backend works predictably and safely
- unsupported capabilities degrade cleanly

### Product success

- memory backends can be swapped without agent code rewrites
- the package enables faster experimentation across both clients
- shared safety behavior is centralized

### Maintenance success

- new memory backends can be added without touching core client code
- both clients can upgrade the package through normal dependency management

## Recommended First Implementation Slice

To keep risk under control, the first slice should be:

1. define contracts and runtime adapter protocol
2. build in-tree package skeleton
3. wrap current YClientReddit memory as `hybrid_semantic`
4. prove parity
5. add `simple_recent`
6. externalize the package
7. integrate into YClient

This is the fastest path to a real shared package without forcing premature generalization.

## Final Recommendation

The external package target is realistic and strategically sound.

The right path is:

- extract contracts first
- preserve the current hybrid backend as the reference backend
- standardize access through a runtime adapter and engine interface
- add a simple backend as the first alternative
- package externally only after the seam is stable

This approach gives:

- low regression risk
- strong reuse potential
- a clean path for both YClientReddit and YClient to share one memory package with multiple interchangeable memory systems behind a single public signature
