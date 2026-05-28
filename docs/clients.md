# Clients

## Overview

The repository contains two main client implementations under [`y_client/clients/`](/Users/rossetti/PycharmProjects/YClientReddit/y_client/clients):

- `YClientBase`
- `YClientWeb`

Both clients coordinate the simulation, but they differ in how they load configuration, prompts, and persistence backends.

## `YClientBase`

File:

- [`y_client/clients/client_base.py`](/Users/rossetti/PycharmProjects/YClientReddit/y_client/clients/client_base.py)

### Constructor shape

```python
YClientBase(
    config_filename,
    prompts_filename=None,
    agents_filename=None,
    graph_file=None,
    agents_output="agents.json",
    owner="admin",
)
```

### Key responsibilities

- load config JSON from disk
- load prompts and merge them with defaults
- initialize simulation clocks and feed holders
- load optional graph files for initial networks
- reset experiments
- load RSS or URL-based news
- create or load agent populations

### When to use it

Use `YClientBase` when you want a file-driven local run using the standard CLI entrypoint.

## `YClientWeb`

File:

- [`y_client/clients/client_web.py`](/Users/rossetti/PycharmProjects/YClientReddit/y_client/clients/client_web.py)

### Constructor shape

```python
YClientWeb(
    config_file,
    data_base_path,
    agents_filename=None,
    agents_output="agents.json",
    owner="admin",
    first_run=False,
    network=None,
    log_file="agent_execution.log",
    llm=True,
)
```

### Key responsibilities

- accept config as a Python object
- load prompts from the experiment directory
- initialize structured logging
- connect to PostgreSQL if `DATABASE_URL` is set
- otherwise use an experiment SQLite database
- prepare image-post data when needed

### When to use it

Use `YClientWeb` when running the client inside a larger experiment or web-backed environment that manages an experiment folder and database state externally.

## Prompt loading behavior

Both clients merge experiment prompts with defaults when possible. This means:

- prompt files can override defaults selectively
- missing prompt keys can still be supplied by bundled defaults

## Recommender setup

Both clients expose `set_recsys(...)` and expect:

- one content recommender
- one follow recommender

The standard CLI initializes these from class names passed as arguments.

## Experiment state and outputs

Common output artifacts:

- generated agent snapshots
- copied config or prompts
- experiment databases
- logging output

See [Running Simulations](running-simulations.md) and [Usage Examples](examples.md) for concrete commands.
