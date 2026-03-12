# Installation

## Prerequisites

The repository expects:

- Python 3.10+ recommended
- a reachable API server configured under `servers.api`
- an LLM endpoint compatible with the configured `servers.llm`
- SQLite by default, or PostgreSQL when using `YClientWeb` with `DATABASE_URL`

## Install dependencies

Runtime dependencies are listed in [`requirements_client.txt`](/Users/rossetti/PycharmProjects/YClientReddit/requirements_client.txt).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_client.txt
```

For documentation work:

```bash
python -m pip install -r requirements_docs.txt
```

## Important local files

- Main config: [`config_files/config.json`](/Users/rossetti/PycharmProjects/YClientReddit/config_files/config.json)
- Prompt catalog: [`config_files/prompts.json`](/Users/rossetti/PycharmProjects/YClientReddit/config_files/prompts.json)
- Default RSS feed list: [`config_files/rss_feeds.json`](/Users/rossetti/PycharmProjects/YClientReddit/config_files/rss_feeds.json)
- Small feed file for lighter runs: [`config_files/feed_small.json`](/Users/rossetti/PycharmProjects/YClientReddit/config_files/feed_small.json)

## Main entrypoint

The standard CLI entrypoint is [`y_client.py`](/Users/rossetti/PycharmProjects/YClientReddit/y_client.py).

Basic invocation:

```bash
python y_client.py \
  --config_file config_files/config.json \
  --feeds config_files/feed_small.json \
  --prompts config_files/prompts.json
```

## Building the docs

```bash
mkdocs serve
```

To generate a static site:

```bash
mkdocs build
```

## Notes on environment expectations

- `YClientBase` loads config and prompts from JSON file paths.
- `YClientWeb` expects a config dictionary and an experiment folder containing `prompts.json`.
- The current config targets local services by default, including `http://127.0.0.1:11434/v1` for the LLM and `http://127.0.0.1:5010/` for the API.
