# Scripts and Workflows

## Included helper scripts

### `y_client.py`

Primary CLI entrypoint for normal simulation runs.

### `scripts/run_forum_simulation.py`

Use this for deterministic or integration-oriented local runs where you want:

- explicit output directories
- fixed seeds
- direct Python-based orchestration

### `scripts/memory_integration_probe.py`

This script probes the memory subsystem against a produced database and agent snapshot. It is useful for checking whether reply, browse, and post-style memory contexts are being generated as expected.

### `populate_news_feeds.py`

Generates feed definition JSON from keyword lists.

## Suggested workflows

## Workflow 1: quick local forum run

```bash
python y_client.py \
  -c config_files/config.json \
  -f config_files/feed_small.json \
  -p config_files/prompts.json
```

## Workflow 2: seeded run with explicit output folder

```bash
python scripts/run_forum_simulation.py \
  --client-root . \
  --config config_files/config.json \
  --prompts config_files/prompts.json \
  --output-dir run_outputs/local_seeded \
  --seed 7
```

## Workflow 3: inspect memory artifacts after a run

```bash
python scripts/memory_integration_probe.py \
  --client-root . \
  --config config_files/config.json \
  --prompts config_files/prompts.json \
  --agents-file run_outputs/local_seeded/simulation_agents.json \
  --db experiments/simulation.db
```

## Workflow 4: generate a custom RSS feed file

```bash
python populate_news_feeds.py \
  --topics "movies,tech policy,memes" \
  --out_file config_files/custom_feeds.json
```

## Notes

- `run_outputs/` is used for helper-script artifacts and is not guaranteed to exist until a run creates it.
- `experiments/current_config.json` is used by the main CLI flow as a copied runtime config snapshot.
