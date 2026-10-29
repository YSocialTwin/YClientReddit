# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
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
