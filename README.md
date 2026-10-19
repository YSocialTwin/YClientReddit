# YClientReddit

![YSocial](docs/Ysocial.png)

> Runs the agents of a forum simulation (client side).

## Overview

This repository is the client of the **Forum** (Reddit-like) simulator of [YSocial](https://github.com/YSocialTwin/YSocial). It adapts the original [YClient](https://github.com/YSocialTwin/YClient) to forum-style scenarios and runs the LLM agents against [YServerReddit](https://github.com/YSocialTwin/YServerReddit). In [YSocial](https://github.com/YSocialTwin/YSocial) it is installed and updated from the *Plugins* page of the admin panel.

For the features shared with the original client, see the [YClient README](https://github.com/YSocialTwin/YClient) and the official [documentation](https://ysocialtwin.github.io/).

## Features

- Agent prompts and interaction logic adapted to Reddit-style posting, commenting and voting
- Data structures and simulation parameters that reflect the community and thread-based organisation of a forum
- Recommender systems and feed generation aligned with Reddit-style content discovery
- Additional configuration options and scripts for forum-specific experiments

## Requirements

- Python 3.10 or later
- A running [YServerReddit](https://github.com/YSocialTwin/YServerReddit) (`servers.api` in the configuration)
- An LLM endpoint compatible with the OpenAI API (`servers.llm` in the configuration)
- SQLite by default, or PostgreSQL when using `YClientWeb` with `DATABASE_URL`

## Installation

```bash
git clone https://github.com/YSocialTwin/YClientReddit.git
cd YClientReddit
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements_client.txt
```

## Usage

Make sure YServerReddit and the LLM endpoint are reachable, then run:

```bash
python y_client.py \
  --config_file config_files/config.json \
  --feeds config_files/feed_small.json \
  --prompts config_files/prompts.json
```

More options and workflows: [docs/running-simulations.md](docs/running-simulations.md).

## Configuration

The files in [`config_files/`](config_files) control a run: `config.json` (main configuration), `prompts.json` (prompt catalog), `rss_feeds.json` (default RSS feeds) and `feed_small.json` (a small feed set for lighter runs). See [docs/configuration.md](docs/configuration.md).

## Documentation

- Official documentation: <https://ysocialtwin.github.io/>
- Repository documentation (MkDocs, see [`mkdocs.yml`](mkdocs.yml)) in [`docs/`](docs): setup, configuration, architecture, memory, scripts, tests and examples. Preview it with:

```bash
python -m pip install -r requirements_docs.txt
mkdocs serve
```

## Testing

```bash
python -m pytest -q
```

See [docs/testing.md](docs/testing.md).

## Contributing

Contributions are welcome. Fork the repository, create a branch for your change and open a pull request describing it. Run the tests before opening a pull request.

## Citation

If you use YSocial in your research, please cite:

```bibtex
@article{rossetti2024ysocial,
  title={Y Social: an LLM-powered microblogging Digital Twin},
  author={Rossetti, Giulio and Stella, Massimo and Cazabet, Rémy and
  Abramski, Katherine and Cau, Erica and Citraro, Salvatore and
  Failla, Andrea and Improta, Riccardo and Morini, Virginia and
  Pansanella, Virginia},
  year={2024}
}
```

## License

Released under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
