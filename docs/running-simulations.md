# Running Simulations

## Standard simulation run

Use [`y_client.py`](/Users/rossetti/PycharmProjects/YClientReddit/y_client.py) for normal runs.

Example:

```bash
python y_client.py \
  -c config_files/config.json \
  -f config_files/feed_small.json \
  -p config_files/prompts.json \
  -o admin
```

## Useful CLI flags

- `-c`, `--config_file`: simulation config JSON
- `-f`, `--feeds`: RSS feed definition file
- `-p`, `--prompts`: prompts JSON
- `-a`, `--agents`: pre-existing agents file
- `-o`, `--owner`: owner username
- `-r`, `--reset`: reset experiment state
- `-n`, `--news`: reload RSS feeds
- `-x`, `--crecsys`: content recommender class name
- `-y`, `--frecsys`: follow recommender class name
- `-g`, `--graph`: CSV graph file for the starting social graph
- `--news_source`: `rss` or `urls`
- `--urls_file`: file containing source URLs when `--news_source=urls`
- `--max_urls`: sample size cap for URL ingestion

## Loading RSS feeds

```bash
python y_client.py \
  -c config_files/config.json \
  -f config_files/rss_feeds.json \
  -p config_files/prompts.json \
  --news
```

## Loading direct URLs instead of RSS

```bash
python y_client.py \
  -c config_files/config.json \
  -p config_files/prompts.json \
  --news_source urls \
  --urls_file urls.txt \
  --max_urls 250
```

## Running from the helper script

[`scripts/run_forum_simulation.py`](/Users/rossetti/PycharmProjects/YClientReddit/scripts/run_forum_simulation.py) is a programmatic runner useful for reproducible or integration-driven runs.

Example:

```bash
python scripts/run_forum_simulation.py \
  --client-root . \
  --config config_files/config.json \
  --prompts config_files/prompts.json \
  --output-dir run_outputs/forum_run \
  --seed 7
```

## Output locations

Typical artifacts are written under:

- [`experiments/`](/Users/rossetti/PycharmProjects/YClientReddit/experiments)
- [`run_outputs/`](/Users/rossetti/PycharmProjects/YClientReddit/run_outputs)

Common generated files:

- agent snapshot JSON
- copied prompts
- copied current config
- SQLite database files
- optional probe results
