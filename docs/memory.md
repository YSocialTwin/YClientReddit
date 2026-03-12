# Memory System

## Overview

The repository contains a substantial in-agent memory implementation centered on [`y_client/classes/base_agent.py`](/Users/rossetti/PycharmProjects/YClientReddit/y_client/classes/base_agent.py).

At a high level it provides:

- structured context retrieval
- prompt-time continuity cues
- relationship memory
- thread memory
- community-digest memory
- optional reflection synthesis
- optional high-affect callback enforcement

## Current runtime posture

The checked-in config favors a restrained forum-style memory mode:

- `memory_prompt_mode = "subtle_forum"`
- `memory_vote_signal_only = true`
- `memory_high_affect_enabled = false`
- `memory_nuance_planner_enabled = false`

That means the full implementation is broader than the current default behavior.

## Key implementation areas

Important memory-related areas include:

- `Agent._init_memory_config(...)`
- `_memory_fetch_context(...)`
- `_memory_search(...)`
- `_memory_build_reply_context(...)`
- `_memory_build_tiered_context(...)`
- `_memory_build_post_style_context(...)`
- `_memory_build_thread_browse_context(...)`
- `_memory_record_event(...)`
- `_memory_upsert_social_card(...)`
- `_memory_maybe_update_thread_card(...)`
- `_memory_maybe_update_community_digest(...)`
- `_memory_maybe_generate_reflections(...)`

## Experimental external-memory bridge

[`y_client/memory_runtime.py`](/Users/rossetti/PycharmProjects/YClientReddit/y_client/memory_runtime.py) provides an adapter-oriented bridge to an external `yclient_memory` package when the source tree exists locally.

That file is relevant if you are:

- extracting memory into a shared package
- evaluating package-based memory backends
- keeping the in-repo agent compatible with a standardized memory runtime

## Read paths

The in-repo implementation distinguishes:

- structured memory fetches for social/thread/community context
- semantic memory searches for more selective recall

These feed:

- reply context
- browse context
- post style hints
- relationship prioritization

## Write paths

Memory updates are typically triggered after agent actions:

- comments
- posts
- votes

These paths record events and can update:

- social cards
- thread cards
- community digest
- reflection items

## Supporting reports

For full analysis and refactor planning, see:

- [Agent Memory Report](agent_memory_report.md)
- [Parametric Memory Refactor](agent_memory_parametric_refactor_report.md)
- [External Memory Package Pipeline](external_memory_package_pipeline.md)
