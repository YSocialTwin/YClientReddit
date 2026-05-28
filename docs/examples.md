# Usage Examples

## Run the default simulation

```bash
python y_client.py \
  -c config_files/config.json \
  -f config_files/feed_small.json \
  -p config_files/prompts.json
```

## Run with custom recommenders

```bash
python y_client.py \
  -c config_files/config.json \
  -p config_files/prompts.json \
  -x HotRanking \
  -y PreferentialAttachment
```

## Start from an existing agent snapshot

```bash
python y_client.py \
  -c config_files/config.json \
  -p config_files/prompts.json \
  -a experiments/simulation_agents.json
```

## Reset experiment state before running

```bash
python y_client.py \
  -c config_files/config.json \
  -p config_files/prompts.json \
  --reset true
```

## Use direct URLs as the content source

```bash
python y_client.py \
  -c config_files/config.json \
  -p config_files/prompts.json \
  --news_source urls \
  --urls_file urls.txt \
  --max_urls 500
```

## Seeded run through the helper script

```bash
python scripts/run_forum_simulation.py \
  --client-root . \
  --config config_files/config.json \
  --prompts config_files/prompts.json \
  --output-dir run_outputs/seeded_demo \
  --seed 42
```

## Probe memory after a run

```bash
python scripts/memory_integration_probe.py \
  --client-root . \
  --config config_files/config.json \
  --prompts config_files/prompts.json \
  --agents-file run_outputs/seeded_demo/simulation_agents.json \
  --db experiments/simulation.db
```

## Build the documentation site

```bash
python -m pip install -r requirements_docs.txt
mkdocs serve
```

## Run focused tests

```bash
pytest tests/test_forum_memory_prompt_mode.py
pytest tests/test_high_affect_memory_callback.py
pytest tests/test_length_enforcement.py
```
