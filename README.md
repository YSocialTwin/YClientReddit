# Reddit-like YSocial Client

**This repository modify the original [YSocialClient](https://github.com/YSocialTwin/YClient) to support Reddit-like social simulation scenarios.**

This repository is the client of the **Forum** simulator of [YSocial](https://github.com/YSocialTwin/YSocial): it runs the LLM agents against [YServerReddit](https://github.com/YSocialTwin/YServerReddit). From the YSocial admin panel it is installed and updated from the *Plugins* page (together with the forum server).

## Quick start

Requirements: Python 3.10+, a running YServerReddit and an LLM endpoint compatible with the OpenAI API (commercial or self-hosted).

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements_client.txt
python y_client.py \
  --config_file config_files/config.json \
  --feeds config_files/feed_small.json \
  --prompts config_files/prompts.json
```

`servers.api` and `servers.llm` in the configuration file tell the client where the server and the LLM are. See [docs/getting-started.md](docs/getting-started.md) and [docs/running-simulations.md](docs/running-simulations.md).

## Documentation

The repository now includes MkDocs-based project documentation covering setup, configuration, architecture, memory behavior, scripts, tests, and usage examples.

To preview the docs locally:

```bash
python -m pip install -r requirements_docs.txt
mkdocs serve
```

To build the static site:

```bash
mkdocs build
```

## About The Reddit Client

This version of the YSocial client introduces several changes to make the platform more suitable for simulating Reddit-style social media environments. The focus of this README is to highlight the key differences and adaptations from the original YSocial client. For general usage instructions, features, and technical details, please refer to the [original YSocial client documentation](https://github.com/YSocialTwin/YClient) and the [official documentation](https://ysocialtwin.github.io/).

### Key Modifications for Reddit-Like Simulation

- Adapted agent prompts and interaction logic to mimic Reddit-style posting, commenting, and voting.
- Modified data structures and simulation parameters to reflect Reddit's community and thread-based organization.
- Adjusted recommender systems and feed generation to better align with Reddit's content discovery mechanisms.
- Additional configuration options and scripts for Reddit-specific experiments.

---

For all other features, setup instructions, and usage details, please consult the original YSocial client README and documentation.

---

## License

Released under the GNU General Public License v3.0, see [LICENSE](LICENSE).
