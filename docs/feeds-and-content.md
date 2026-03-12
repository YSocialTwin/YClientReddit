# Feeds and Content

## Content sources

The repository supports two main news ingestion modes:

- RSS feeds
- direct URL lists

## RSS configuration

Default feed files:

- [`config_files/rss_feeds.json`](/Users/rossetti/PycharmProjects/YClientReddit/config_files/rss_feeds.json)
- [`config_files/feed_small.json`](/Users/rossetti/PycharmProjects/YClientReddit/config_files/feed_small.json)

These files contain feed definitions used by `YClientBase.load_rrs_endpoints(...)`.

## URL ingestion

When running:

```bash
python y_client.py --news_source urls --urls_file urls.txt
```

the client uses URL ingestion instead of RSS and processes articles with the URL reader logic from [`y_client/news_feeds/url_reader.py`](/Users/rossetti/PycharmProjects/YClientReddit/y_client/news_feeds/url_reader.py).

## Feed helper script

[`populate_news_feeds.py`](/Users/rossetti/PycharmProjects/YClientReddit/populate_news_feeds.py) generates Bing RSS feed definitions from a topic list.

Example:

```bash
python populate_news_feeds.py \
  --topics "ai,streaming wars,us politics" \
  --suffix "reddit" \
  --out_file config_files/generated_feeds.json
```

## Feed-related package area

The feed modules live under [`y_client/news_feeds/`](/Users/rossetti/PycharmProjects/YClientReddit/y_client/news_feeds).

Important files:

- `feed_reader.py`
- `url_reader.py`
- `client_modals.py`

These handle:

- reading sources
- normalizing articles and images
- persisting or exposing them to the simulation

## Page agents and content publishing

[`y_client/classes/page_agent.py`](/Users/rossetti/PycharmProjects/YClientReddit/y_client/classes/page_agent.py) represents news-page style agents that:

- fetch articles from associated feeds
- generate a post via prompts
- avoid duplicate article posting

This is the main path for publisher-like content accounts in the simulation.
