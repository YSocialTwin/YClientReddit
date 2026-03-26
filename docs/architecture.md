# Repository Architecture

## High-level layout

The repository is organized around a simulation client plus supporting domain modules.

```text
config_files/      Runtime JSON configuration and prompts
docs/              Project documentation and memory reports
scripts/           Helper scripts for deterministic runs and memory probing
tests/             Focused regression tests
y_client/          Main Python package
y_client.py        CLI entrypoint
populate_news_feeds.py  RSS feed generation helper
```

## Main execution flow

The standard CLI flow in [`y_client.py`](/Users/rossetti/PycharmProjects/YClientReddit/y_client.py) is:

1. Parse CLI arguments.
2. Load config and prompts.
3. Instantiate recommenders.
4. Instantiate the configured client class.
5. Optionally reset experiment state.
6. Optionally reload RSS feeds or URL-derived content.
7. Create or load agents.
8. Run the simulation loop.

## Core package structure

### `y_client/clients`

Contains the top-level simulation client implementations:

- [`client_base.py`](/Users/rossetti/PycharmProjects/YClientReddit/y_client/clients/client_base.py)
- [`client_web.py`](/Users/rossetti/PycharmProjects/YClientReddit/y_client/clients/client_web.py)

Responsibilities:

- loading config and prompts
- preparing the simulation environment
- creating agent populations
- wiring recommenders
- ingesting content sources
- driving the simulation loop

### `y_client/classes`

Contains the social actors and time model:

- [`base_agent.py`](/Users/rossetti/PycharmProjects/YClientReddit/y_client/classes/base_agent.py)
- [`page_agent.py`](/Users/rossetti/PycharmProjects/YClientReddit/y_client/classes/page_agent.py)
- [`fake_base_agent.py`](/Users/rossetti/PycharmProjects/YClientReddit/y_client/classes/fake_base_agent.py)
- [`annotator.py`](/Users/rossetti/PycharmProjects/YClientReddit/y_client/classes/annotator.py)
- [`time.py`](/Users/rossetti/PycharmProjects/YClientReddit/y_client/classes/time.py)

Responsibilities:

- agent persona state
- action selection
- posting, replying, voting, following
- thread reading and mention handling
- prompt orchestration
- memory logic
- optional topic-level opinion dynamics persistence and updates

### `y_client/recsys`

Contains recommendation strategy wrappers:

- content ranking strategies
- follow suggestion strategies

These are light adapters that send parameterized requests to the configured API backend.

### `y_client/news_feeds`

Contains ingestion and feed-domain logic:

- RSS reading
- direct URL extraction
- article and image models
- content import helpers

### `y_client/memory_runtime.py`

This file is an integration bridge toward the externalized `yclient-memory` package. It adapts the current Reddit agent to the installed `yclient_memory` runtime and backend factory.

## Client split

### `YClientBase`

File:

- [`y_client/clients/client_base.py`](/Users/rossetti/PycharmProjects/YClientReddit/y_client/clients/client_base.py)

Characteristics:

- path-based config loading
- path-based prompt loading
- RSS and URL ingestion support
- more direct use in local script-driven runs

### `YClientWeb`

File:

- [`y_client/clients/client_web.py`](/Users/rossetti/PycharmProjects/YClientReddit/y_client/clients/client_web.py)

Characteristics:

- config consumed as a Python dictionary
- experiment folder based prompt loading
- SQLite or PostgreSQL support depending on runtime environment
- logger initialization for execution timing
- image population handling for web-backed experiments

## Agent split

### `Agent`

Defined in:

- [`y_client/classes/base_agent.py`](/Users/rossetti/PycharmProjects/YClientReddit/y_client/classes/base_agent.py)

This is the primary behavior engine for Reddit-style users.

Major responsibilities:

- persona and LLM initialization
- reading threads and mentions
- posting and replying
- content moderation / stance behavior through prompts
- voting and follow decisions
- memory-aware continuity

### `PageAgent`

Defined in:

- [`y_client/classes/page_agent.py`](/Users/rossetti/PycharmProjects/YClientReddit/y_client/classes/page_agent.py)

This represents publisher-like accounts that primarily post news from feeds rather than participate like normal users.

### `FakeAgent`

Defined in:

- [`y_client/classes/fake_base_agent.py`](/Users/rossetti/PycharmProjects/YClientReddit/y_client/classes/fake_base_agent.py)

This provides a simpler agent variant useful for special flows and lower-complexity interactions.

## Persistence and external dependencies

The client relies on:

- an HTTP API server for user/feed/post actions
- an LLM endpoint for prompt-driven behavior
- a local or remote database, depending on client mode

When `simulation.opinion_dynamics.enabled` is true, the agent layer also uses the experiment database as a lightweight opinion-state store. The current implementation reads topic identifiers from `interests`, seeds first-round rows via `rounds`, and appends opinion updates to `agent_opinion`.

The code is therefore best understood as a simulation client and orchestration layer, not as a fully standalone simulator.

## Documentation pointers

- configuration reference: [Configuration](configuration.md)
- client classes and runtime modes: [Clients](clients.md)
- memory implementation overview: [Memory System](memory.md)
- opinion-state model and configuration: [Opinion Dynamics](opinion-dynamics.md)
