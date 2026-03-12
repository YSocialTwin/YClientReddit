# Reddit-like YSocial Client 

**This repository modify the original [YSocialClient](https://github.com/YSocialTwin/YClient) to support Reddit-like social simulation scenarios.**

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
