# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Fixed
- LLM endpoints with an API key: Ollama endpoints that need a key (hosted ones) now get it as a bearer token.

### Added
- Forum share re-enabled on request: `SHARE` stays at 0 unless the client configuration has `share.enabled: true` or the federation is on with `crosspost_mode: agent_action` (default off, documented); the configured likelihood is kept (missing or 0 becomes 0.05); with `agent_action` the `/share` payload carries `crosspost_to` (`federation.share_fields`, no random draw otherwise).
- Cross-post as an agent action (opt-in, `federation.crosspost_mode: agent_action`): after posting, the agent draws once per platform where it is also active (`GET /federation/destinations`) with `federation.crosspost_propensity`, and sends `crosspost_to` ([{member, text}]) with the post; the text is the model's version (prompts `crosspost_adapt_<platform>`, EN and IT) or, without a model or on a failed reply, the same words cut and shaped for the platform. Default mode unchanged.
- Federation attention (opt-in): the agent's number of actions in a slot is scaled by the multiplier of the day that the server publishes (`GET /federation/attention`), rounded at random with the module's own generator; with the federation off or multiplier 1 nothing changes.
- Federation (opt-in, `y_client/federation.py`): with the `federation` section of the client configuration enabled, agents that are not active on the platform (listed by `GET /federation/inactive_users`) are not sampled and do not act, and an agent may mark a post for cross-post (`federation.crosspost_propensity`, own seeded generator, extra `crosspost` field of `/post`). Without the section nothing changes.
- Content recommenders `ReverseChronoComments`, `CommonInterests`, `CommonUserInterests`, `SimilarUsersReactions` and `SimilarUsersPosts`, so every recommender offered by YWeb for forum clients exists in the client (before, selecting them made the agents fail to load).
- `y_client.recsys.CONTENT_RECSYS` lists the available content recommenders; the CLI `-x` option validates against it and shows them in its help.
- Tests for the content recommenders (`tests/test_content_recsys.py`).

### Fixed
- `ReverseChrono`, `ReverseChronoPopularity`, `ReverseChronoFollowers` and `ReverseChronoFollowersPopularity` ignored `n_posts` (the limit was fixed to 10).
- The `articles` flag no longer stays set on the recommender after an article read.
- Every agent gets its own copy of the recommenders (the instance was shared and its `uid` overwritten at each read).
- `HotRanking` docstring and docs describe the actual formula; documented `MostCommented` and the available modes.

## [2.1.1] - 2026-10-22

### Fixed
- SQLite URIs on Windows: `sqlite:///` plus the absolute path (the previous `sqlite:////` form produced "unable to open database file").
- `SyntaxWarning` for the unescaped regex in the feed reader (raw string).

## [2.1.0] - 2026-10-20

### Added
- Topic sampling weighted by frequency in the moving window (only topics with count > 0); exactly one topic per generated post.
- Hashtags in canonical form: no leading `#`, apostrophes and accents kept, case-insensitive identity.
- Prompts i18n: prompts moved out of the code into `prompts_internal.json`, read through `prompt_store` with language support.
- Prompts i18n: protocol tokens and tolerant parsers (no `eval`); Italian internal prompt pack.
- Prompts i18n: localized follow action and reciprocal-follow verb; persona fragments moved into the packs.
- Prompts i18n: forum prompts read from `data_schema/prompts/en`; the client reads the prompt language and shows profile values in it.
- Memory M5: the memory engine renders its texts in the simulation prompt language.
- Memory M2: the log shows which `yclient_memory` is loaded (path, version, language); errors name the Agent Memory plugin and warn when the package ignores the language.
- `NEWS` defined in `protocol_tokens` as a legacy alias of `SHARE_LINK`.

### Changed
- Requirements aligned with the library versions shipped by YWeb.
- Shorter, purpose-oriented plugin description in `meta/info.json`.
- License: GPL-3.0-only, declared in `LICENSE`, `meta/info.json` and the README.
- README restructured on the common layout; citation updated to the published YSocial reference (Big Data & Society, 2026).

[Unreleased]: https://github.com/YSocialTwin/YClientReddit/compare/v2.1.1...HEAD
[2.1.1]: https://github.com/YSocialTwin/YClientReddit/compare/v2.1.0...v2.1.1
[2.1.0]: https://github.com/YSocialTwin/YClientReddit/releases/tag/v2.1.0
