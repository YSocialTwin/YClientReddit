# YClientReddit

YClientReddit is a Reddit-style adaptation of the YSocial client stack. It keeps the simulation-driven architecture of the original project while reshaping agent behavior, prompts, feeds, and interaction flows around:

- root posts and comment threads
- mention/reply handling
- voting instead of purely broadcast interaction
- subreddit-like discussion dynamics
- memory-aware continuity for forum conversations
- stress/reward-aware interaction feedback and churn
- reciprocal follow/unfollow evaluation, including secondary follows

## What is in this repository

The repository contains:

- the simulation entrypoint in [`y_client.py`](/Users/rossetti/PycharmProjects/YClientReddit/y_client.py)
- the client implementations in [`y_client/clients/`](/Users/rossetti/PycharmProjects/YClientReddit/y_client/clients)
- the agent classes in [`y_client/classes/`](/Users/rossetti/PycharmProjects/YClientReddit/y_client/classes)
- content and follow recommenders in [`y_client/recsys/`](/Users/rossetti/PycharmProjects/YClientReddit/y_client/recsys)
- feed ingestion helpers in [`y_client/news_feeds/`](/Users/rossetti/PycharmProjects/YClientReddit/y_client/news_feeds)
- simulation configuration and prompts in [`config_files/`](/Users/rossetti/PycharmProjects/YClientReddit/config_files)
- helper scripts in [`scripts/`](/Users/rossetti/PycharmProjects/YClientReddit/scripts)
- tests in [`tests/`](/Users/rossetti/PycharmProjects/YClientReddit/tests)

## Documentation map

Start here if you need:

- environment setup and how to run the simulator: [Getting Started](getting-started.md)
- all runtime knobs and JSON structure: [Configuration](configuration.md)
- stress/reward and reciprocal-follow behavior: [Social Feedback Loop](social-feedback.md)
- the high-level code layout and execution flow: [Repository Architecture](architecture.md)
- memory-specific behavior and its reports: [Memory System](memory.md)
- concrete commands for common tasks: [Usage Examples](examples.md)

## Typical workflow

1. Install the Python dependencies.
2. Check `config_files/config.json` and `config_files/prompts.json`.
3. Start the API/LLM services expected by the config.
4. Run `python y_client.py` with the desired arguments.
5. Inspect `experiments/`, logs, and optional memory probe output.

## Documentation tooling

This repository now includes MkDocs support. To build the docs locally:

```bash
python -m pip install -r requirements_docs.txt
mkdocs serve
```

Then open the local URL printed by MkDocs.
