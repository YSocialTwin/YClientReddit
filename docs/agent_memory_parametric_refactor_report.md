# Parametric Agent Memory Refactor Report

## Objective

This report evaluates whether the current agent memory system can be made parametric, meaning:

- the memory implementation is extracted behind an abstract contract
- different memory models can be plugged in without changing agent behavior flow
- the current memory system remains the default and preserves behavior
- a simpler alternative memory backend is added as an option

This document does **not** propose immediate code changes. It describes:

- feasibility
- refactor scope and weight
- a target architecture
- migration steps
- a roadmap
- success criteria
- a low-complexity alternative memory system to introduce alongside the current one

## Short Answer

Yes, it is possible.

It is also a **meaningful refactor**, not a light cleanup.

The current memory system is deeply embedded in [`y_client/classes/base_agent.py`](/Users/rossetti/PycharmProjects/YClientReddit/y_client/classes/base_agent.py). There are hundreds of `_memory_*` references and the logic is entangled with:

- reply generation
- thread browsing
- mention handling
- voting
- post generation
- reflection synthesis
- prompt shaping
- decision logging
- relationship prioritization

So the correct framing is:

- **architecturally feasible**
- **operationally desirable**
- **moderate-to-heavy refactor**

## Feasibility Assessment

## Why it is feasible

The existing implementation already has a fairly coherent internal shape. Even though it lives in one large class, it naturally separates into a few responsibilities:

### Read-side memory services

- fetch structured context
- semantic search
- prompt context building
- conversation cue generation
- browse/post style context generation

### Write-side memory services

- event recording
- social card update
- thread card update
- community digest update
- reflection generation

### Utility and policy services

- decay and forgetting
- memory sanitization
- vote artifact filtering
- callback verification
- high-affect recall orchestration

That natural clustering is exactly what makes extraction possible.

## Why it is not trivial

The implementation is not a standalone module today. It is tightly coupled to `Agent` state and utilities:

- `self.prompts`
- `self.llm_config`
- agent persona fields
- `self.base_url`
- `self._decision_log`
- text cleaning helpers
- thread/post lookup helpers
- topic/sentiment lookup logic
- reply/post generation flows

In practice, the current memory logic is partly:

- domain logic
- orchestration logic
- prompt-building logic
- infrastructure adapter logic
- policy logic

Those concerns need to be separated before a clean “memory backend interface” exists.

## Refactor Weight Estimate

## Overall estimate

I would classify the refactor as:

- **medium-heavy** if the goal is only to support one additional simple backend while preserving the current one as default
- **heavy** if the goal is a fully generalized plugin system with clean backend independence for all advanced features

## Practical effort bands

### Option A: pragmatic parametric extraction

Scope:

- keep the current memory behavior intact
- introduce a small abstract interface
- move current implementation into a default backend adapter
- add one simpler alternative backend
- keep some advanced logic still in agent-level orchestration for phase 1

Estimated cost:

- roughly **1.5 to 3 weeks** of focused engineering work

Risk:

- moderate

This is the recommended approach.

### Option B: full backend-neutral memory architecture

Scope:

- fully separate storage, retrieval, summarization, reflection, prompt policies, and affect-specific logic
- make all advanced behaviors backend-optional and capability-driven
- minimize direct memory branching inside `Agent`

Estimated cost:

- roughly **3 to 6+ weeks**

Risk:

- medium-high

This is cleaner long term but unnecessary as the first move.

## What makes the refactor heavy

The heaviest part is not moving methods into new files. The heavy part is defining the correct seam.

The seam must preserve current behavior in:

- subtle forum mode
- legacy tiered mode
- vote-signal-only behavior
- community digest behavior
- reflection cadence behavior
- high-affect optional callback flow
- relationship-priority browse behavior

If the seam is chosen poorly, the team will either:

- duplicate lots of logic across backends
- or keep the `Agent` class full of backend-specific conditionals

## Recommended Architectural Direction

The best approach is **not** “one abstract base class with 80 methods.”

The better model is:

- a small top-level memory engine interface
- a stable set of data contracts
- optional backend capabilities
- one orchestration layer in the agent

## Target structure

Suggested package:

```text
y_client/memory/
  __init__.py
  contracts.py
  capabilities.py
  engine.py
  context.py
  backends/
    __init__.py
    hybrid_semantic.py
    simple_recent.py
  adapters/
    llm_tools.py
    api_client.py
  policies/
    prompt_policy.py
    affect_policy.py
    safety_policy.py
```

## Core idea

Split the current system into four layers.

### 1. Contracts layer

Defines the stable request/response objects:

- `MemoryReadRequest`
- `ReplyMemoryContext`
- `BrowseMemoryContext`
- `PostStyleContext`
- `MemoryWriteEvent`
- `MemoryWriteResult`
- `RelationshipSignal`
- `MemoryCapabilities`

This is the key enabler for parametric backends.

### 2. Engine interface

Defines the backend-facing contract:

- read reply context
- read browse context
- read post style context
- record comment/post/vote event
- compute relationship signal
- optional reflection maintenance hook
- optional maintenance hook per round

### 3. Policies layer

Keeps backend-independent behavior rules outside storage code:

- vote artifact suppression
- memory prompt mode behavior
- high-affect callback policy
- nuanced cue generation
- fallback rules under degraded retrieval

This is important. Many current `_memory_*` methods are not storage backends. They are behavior policies and should stay shared.

### 4. Backend implementations

Each backend implements the engine interface.

Examples:

- `HybridSemanticMemoryEngine`
- `SimpleRecentMemoryEngine`

## Recommended Interface Shape

The interface should be capability-driven and compact.

Example conceptual interface:

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

    def on_round_tick(self, round_id: int) -> None: ...
```

The interface should return normalized data structures rather than prompt fragments whenever possible.

## Critical Design Decision: What Stays Shared vs What Becomes Backend-Specific

This is the most important part of the refactor.

## Should stay shared

These are policies, not backend implementations:

- prompt mode selection: `legacy` vs `subtle_forum`
- vote artifact filtering
- memory cue formatting
- callback verification logic
- reply rewrite decision logic
- high-affect enforcement rules
- safety rules against invented continuity
- decision logging shape

Reason:

These rules define product behavior and safety, and should remain consistent across memory backends.

## Should move behind the backend

- where memory is stored
- whether semantic search exists
- how structured context is fetched
- whether thread/social/community summaries exist
- whether reflections exist
- how decay is applied
- how event histories are retained

Reason:

These are implementation choices of the memory model.

## Should become capability-gated

Some features should exist only when the backend says it supports them:

- semantic search
- reflections
- community digest
- high-affect recall packs
- thread cards
- social cards
- relationship ranking beyond basic recency

Example:

- the simple backend may support `reply_context` and `relationship_signal`
- the hybrid backend supports everything

## Proposed Capabilities Model

Example capability flags:

- `supports_semantic_search`
- `supports_social_cards`
- `supports_thread_cards`
- `supports_community_digest`
- `supports_reflections`
- `supports_high_affect_recall`
- `supports_relationship_priority`
- `supports_degraded_mode_reporting`

This lets the `Agent` orchestration degrade cleanly without backend-specific branching everywhere.

## Recommended Refactor Strategy

Do this as a **strangler refactor**, not a rewrite.

The current implementation must remain the reference behavior while the new interface is introduced around it.

## Phase 0: Freeze current behavior

Before changing structure:

- document current behavior modes
- snapshot current config defaults
- expand characterization tests around current memory behavior

Required additions before refactor:

- subtle reply context golden tests
- browse context tests
- vote-signal-only tests
- relationship signal tests
- reflection trigger cadence tests
- social card decay/resummarization tests

This phase reduces regression risk.

## Phase 1: Introduce contracts only

Add:

- memory request/response dataclasses or typed dictionaries
- memory capability model
- a factory to instantiate a memory engine from config

Do **not** move behavior yet.

Goal:

- create the shape of the future seam without changing runtime behavior

## Phase 2: Wrap the existing implementation as the default backend

Create `HybridSemanticMemoryEngine`, but initially it can still delegate into existing logic extracted from `base_agent.py`.

The important constraint is:

- `Agent` starts calling `self.memory_engine`
- but `self.memory_engine` preserves exactly the old behavior

This is the safest turning point in the refactor.

## Phase 3: Extract shared policies

Move shared logic out of `Agent` and out of the backend into policy modules:

- cue formatting
- vote artifact filtering
- prompt memory sanitization
- callback verification
- high-affect callback rewrite policy

This reduces duplication and prevents every backend from reimplementing safety rules differently.

## Phase 4: Add the simple alternative backend

Implement the alternative simple engine behind the same interface.

Do not change the default backend yet.

Add config like:

- `memory_backend = "hybrid_semantic"` default
- `memory_backend = "simple_recent"` alternative

## Phase 5: Integrate capability-based fallbacks

Update the `Agent` orchestration to ask:

- what can this backend do?

Then:

- skip reflections if unsupported
- skip semantic recall if unsupported
- use simpler prompt memory context if only recent history exists

This phase makes the parametric design real rather than nominal.

## Phase 6: Validate parity and rollout

Use:

- unit tests
- characterization tests
- A/B evaluation with identical seeds/configs where feasible
- decision-log diffing

Only after parity is demonstrated should the new structure be considered complete.

## Recommended Alternative Simple Memory System

The best simple alternative is **Recent Interaction Memory**, not a second semantic system.

## Why this alternative

It is:

- easy to reason about
- cheap to maintain
- cheap to test
- easy to compare against the current system
- sufficient for continuity in many Reddit-like interactions

It also makes a good fallback backend for environments where:

- the memory service is unavailable
- semantic search is too costly
- deterministic behavior is preferred

## Proposed backend: `SimpleRecentMemoryEngine`

### Core behavior

Store only recent, explicit, structured memory in process or in a simple local table:

- last `N` interactions with each user
- last `N` interactions in each thread
- rolling relationship counters
- no semantic embeddings
- no reflections
- no community digest LLM synthesis

### Data retained

For each user pair:

- last few comments exchanged
- relation counts: agree/disagree/funny/helpful/hostile
- simple trust/affinity/conflict counters
- last interaction round

For each thread:

- last few salient claims
- recent participants
- my recent role heuristic

Global:

- maybe last few authored posts for topic freshness

### Retrieval logic

No semantic search.

Reply memory is built from:

- recent same-user interactions
- recent same-thread interactions
- simple relationship counters

Thread browse memory is built from:

- thread-local recent history only

Post style memory:

- either disabled
- or very lightweight heuristics from recent root posts without LLM summarization

### Advantages

- deterministic
- much smaller surface area
- no embedding dependency
- no external memory search service dependency
- easier to unit test
- lower hallucination risk

### Tradeoffs

- no cross-thread semantic recall
- no reflection synthesis
- weaker long-range topical continuity
- less adaptive community style modeling

This is acceptable because it is proposed as an alternative backend, not a replacement for the current default.

## Suggested Configuration Additions

Add new config knobs:

- `memory_backend`
- `memory_backend_options`
- `memory_backend_fallback`

Example:

```json
{
  "agents": {
    "memory_enabled": true,
    "memory_backend": "hybrid_semantic",
    "memory_backend_fallback": "simple_recent"
  }
}
```

Additional backend-specific settings:

### For `simple_recent`

- `memory_simple_pair_history_limit`
- `memory_simple_thread_history_limit`
- `memory_simple_relationship_decay_lambda`
- `memory_simple_enable_post_style_heuristics`

### For `hybrid_semantic`

Reuse existing keys as-is.

This is critical for non-regression. Existing config should continue to work unchanged when `memory_backend` is omitted.

## Non-Regression Requirement

The current implementation should remain the default behavior.

That implies:

### Rule 1

If `memory_backend` is unset:

- instantiate the current hybrid semantic implementation

### Rule 2

All existing config keys keep their current meaning.

### Rule 3

Existing tests for current memory behavior must still pass unchanged.

### Rule 4

The new abstraction must not force simplifications into the current backend.

This is a common refactor failure mode:

- the abstraction becomes too weak
- advanced current behavior gets flattened to match the simple backend

That must be avoided.

The abstraction should be rich enough to preserve the current backend, while capability flags allow simpler backends to do less.

## Detailed Implementation Steps

## Step 1: Create a behavior inventory

Document every memory-driven behavior currently affecting `Agent`:

- reply context injection
- browse context injection
- post style shaping
- vote signal handling
- comment/write event handling
- reflection triggering
- relationship target prioritization
- high-affect callback flow

Output:

- a one-page matrix of feature -> current method -> backend responsibility -> shared policy responsibility

This is the prerequisite for clean extraction.

## Step 2: Define typed contracts

Introduce memory contracts first.

Suggested objects:

- `ReplyMemoryRequest`
- `ReplyMemoryContext`
- `BrowseMemoryRequest`
- `BrowseMemoryContext`
- `PostStyleMemoryRequest`
- `PostStyleMemoryContext`
- `CommentMemoryEvent`
- `VoteMemoryEvent`
- `PostMemoryEvent`
- `RelationshipSignalRequest`
- `RelationshipSignal`
- `MemoryCapabilities`

Important requirement:

- contracts should carry structured data, not only ready-made strings

This keeps future backends flexible.

## Step 3: Introduce the memory engine factory

Create a factory:

- `build_memory_engine(agent, config) -> MemoryEngine`

In phase 1, this can return only the current backend wrapper.

## Step 4: Add an adapter around the current implementation

Wrap the existing memory behavior in `HybridSemanticMemoryEngine`.

Initially this may still delegate to extracted methods close to their current logic.

The goal is not beauty yet. The goal is:

- move call sites in `Agent` to the interface
- keep runtime behavior stable

## Step 5: Extract shared policy helpers

Move helpers that should not vary per backend:

- `_memory_is_vote_artifact`
- `_memory_sanitize_prompt_memory_text`
- `_memory_build_conversation_cues`
- `_memory_format_conversation_cues`
- `_memory_reply_references_recalled_item`

Potentially also:

- callback rewrite policy
- high-affect enforcement logic

These should become policy helpers used by the agent and by backend outputs.

## Step 6: Replace direct `_memory_*` call sites in `Agent`

Incrementally replace direct calls with:

- `self.memory_engine.build_reply_context(...)`
- `self.memory_engine.build_browse_context(...)`
- `self.memory_engine.build_post_style_context(...)`
- `self.memory_engine.record_comment(...)`
- `self.memory_engine.record_vote(...)`
- `self.memory_engine.record_post(...)`
- `self.memory_engine.relationship_signal(...)`

This is where most integration churn will happen.

## Step 7: Introduce capability checks

Examples:

- if backend lacks `supports_reflections`, skip reflection-specific flows
- if backend lacks `supports_semantic_search`, skip cross-thread callback candidates
- if backend lacks `supports_community_digest`, post style context becomes empty or heuristic

This lets the simple backend fit cleanly.

## Step 8: Implement `SimpleRecentMemoryEngine`

Keep it intentionally small:

- recent per-user deque
- recent per-thread deque
- simple counters
- optional round-based decay

No semantic indexing.
No LLM summarization.
No reflections.

## Step 9: Add backend-selection tests

Required coverage:

- hybrid backend default parity
- simple backend basic continuity behavior
- capability fallback correctness
- no-regression in vote handling
- no-regression in prompt mode selection

## Step 10: Add migration docs and rollout guardrails

Document:

- backend selection
- current default remains hybrid
- simple backend intended use cases
- unsupported features by backend

## Suggested Roadmap

## Milestone 1: Stabilize and characterize

Duration:

- 2 to 4 days

Deliverables:

- current-behavior inventory
- expanded tests for existing memory behavior
- dependency map of `_memory_*` responsibilities

Success condition:

- refactor can begin with a stable safety net

## Milestone 2: Contracts and factory

Duration:

- 2 to 3 days

Deliverables:

- memory contracts
- capability model
- memory engine factory

Success condition:

- no behavior change
- current tests remain green

## Milestone 3: Default backend wrapper

Duration:

- 4 to 6 days

Deliverables:

- `HybridSemanticMemoryEngine`
- `Agent` call sites migrated to the interface

Success condition:

- decision-log parity acceptable
- prompt outputs and write paths remain behaviorally equivalent

## Milestone 4: Shared policy extraction

Duration:

- 3 to 5 days

Deliverables:

- safety/prompt policy helpers
- reduced direct memory logic inside `Agent`

Success condition:

- lower coupling
- no backend-specific leakage into prompt policies

## Milestone 5: Simple backend implementation

Duration:

- 3 to 5 days

Deliverables:

- `SimpleRecentMemoryEngine`
- config-driven backend selection
- tests for simple backend behavior

Success condition:

- simple backend usable end-to-end for reply/post/vote flows

## Milestone 6: Evaluation and hardening

Duration:

- 3 to 5 days

Deliverables:

- parity report
- performance comparison
- behavior comparison across backends
- final migration notes

Success condition:

- hybrid backend shows no meaningful regressions
- simple backend behaves predictably and safely

## Success Evaluation Plan

Evaluation should be explicit. This refactor should not be approved on structural cleanliness alone.

## 1. Parity evaluation for current backend

Goal:

- prove that `HybridSemanticMemoryEngine` behaves like the current implementation

Metrics:

- existing tests pass
- new characterization tests pass
- decision-log output remains stable for representative scenarios
- prompt memory blocks remain equivalent or semantically identical
- vote-signal-only behavior unchanged

Recommended method:

- create frozen scenario fixtures with known thread/context inputs
- compare generated memory contexts before and after refactor

## 2. Functional evaluation for simple backend

Goal:

- prove that the alternative backend is coherent, not merely compilable

Metrics:

- reply context is present when recent history exists
- no invented continuity
- no vote leakage
- relationship signal updates from comments/votes still work
- thread-local continuity is preserved

## 3. Safety evaluation

Goal:

- confirm that abstraction did not weaken memory safety

Checks:

- vote artifacts stay out of prompt-visible memory
- degraded/no-history modes do not fabricate memory
- callback verification still works where supported
- unsupported capabilities fail closed, not open

## 4. Performance and complexity evaluation

Goal:

- ensure the refactor does not create undue runtime or maintenance cost

Checks:

- backend selection overhead negligible
- no increase in LLM calls for current backend
- no extra network calls introduced accidentally
- code ownership becomes clearer

## 5. Developer-experience evaluation

Goal:

- make it easier to add or test new memory models

Checks:

- adding a new backend requires implementing only the interface
- most policy behavior is reusable
- integration points are documented and typed

## Key Risks During Refactor

## Risk 1: interface too shallow

If the interface is too minimal, the hybrid backend will leak custom hooks everywhere.

Mitigation:

- use richer contracts plus capability flags

## Risk 2: interface too broad

If the interface mirrors every current `_memory_*` method, the abstraction adds no value.

Mitigation:

- group by product use case, not by current helper count

## Risk 3: regressions in subtle prompt mode

The current config relies on `subtle_forum` behavior.

Mitigation:

- add dedicated regression tests around subtle reply context and prompt assembly

## Risk 4: backend-specific conditionals spread through `Agent`

If not controlled, the agent becomes:

- `if backend == hybrid`
- `if backend == simple`

everywhere.

Mitigation:

- use capabilities and normalized outputs

## Risk 5: current backend gets simplified to fit the new abstraction

This would be a product regression disguised as architecture work.

Mitigation:

- the abstraction must preserve advanced current behavior, not flatten it

## Risk 6: “simple backend” becomes feature creep

If the simple backend starts re-adding semantic search, reflections, and digest synthesis, the value of having a simple backend disappears.

Mitigation:

- define strict non-goals for `SimpleRecentMemoryEngine`

## Recommended Non-Goals for Phase 1

Do not attempt all of these in the first refactor:

- replacing the external memory service
- redesigning prompt behavior at the same time
- changing current default config semantics
- making reflections backend-agnostic in perfect detail on day one
- removing all memory helper methods from `Agent` immediately

The first goal is a safe seam, not maximal elegance.

## Final Recommendation

This structural change is worthwhile.

The current memory system is sophisticated enough that a parametric architecture would create real value:

- easier experimentation with alternative memory models
- cleaner testing
- safer evolution of the current hybrid backend
- simpler fallback options when semantic memory is unavailable or unnecessary

The recommended path is:

1. preserve the current hybrid implementation as the default backend
2. extract a compact, capability-driven memory engine interface
3. move shared prompt/safety policies out of the backend
4. add `SimpleRecentMemoryEngine` as the first alternative backend
5. validate parity aggressively before changing defaults

## Bottom-Line Estimate

If done pragmatically and safely:

- **refactor heaviness**: medium-heavy
- **recommended first scope**: moderate extraction plus one simple backend
- **regression risk**: manageable if characterization tests are added first

The refactor should be treated as an **architectural extraction with parity constraints**, not as a rewrite and not as a cleanup task.
