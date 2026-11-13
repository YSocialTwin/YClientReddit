# Recommenders

## Overview

Recommendation logic is defined in [`y_client/recsys/`](../y_client/recsys). These classes are thin strategy wrappers that encode request parameters and delegate actual ranking or suggestion work to the API server.

## Content recommenders

Defined in:

- [`y_client/recsys/ContentRecSys.py`](../y_client/recsys/ContentRecSys.py)

Available strategies (class name → server `mode`):

| Class | `mode` | Behaviour |
| --- | --- | --- |
| `ContentRecSys` | `default` | Random posts |
| `ReverseChrono` | `rchrono` | Newest posts first |
| `ReverseChronoPopularity` | `rchrono_popularity` | Most reacted posts first, newest first on ties |
| `ReverseChronoFollowers` | `rchrono_followers` | Newest posts from followed accounts (`followers_ratio`, default 0.6), completed with the newest from everybody |
| `ReverseChronoFollowersPopularity` | `rchrono_followers_popularity` | As above, most reacted first |
| `ReverseChronoComments` | `rchrono_comments` | Most commented threads (followed accounts first) |
| `CommonInterests` | `common_interests` | Posts on the user's topics of interest |
| `CommonUserInterests` | `common_user_interests` | Posts reacted to by users with common interests |
| `SimilarUsersReactions` | `similar_users_react` | Posts liked by similar users |
| `SimilarUsersPosts` | `similar_users_posts` | Posts written by similar users |
| `HotRanking` | `hot` | Reddit-style hot: `sign(net) * log10(|net| + 1) + round / round_decay`, plus a small jitter for threads with few votes |
| `TopRanking` | `top` | Threads by net score (likes − dislikes), no time decay |
| `MostCommented` | `most_commented` | Threads by number of comments |

The names are listed in `y_client.recsys.CONTENT_RECSYS`; all the classes honour `n_posts`.

### Notes

- `ReverseChronoFollowersPopularity` is the default in the main CLI.
- `HotRanking`, `TopRanking` and `MostCommented` rank threads only (no comments) and are the more Reddit-like modes.
- All recommenders eventually call API endpoints such as `/read`, `/read_mentions`, and `/search`.
- Each agent works on its own copy of the recommender, so a recommender can be shared safely by the experiment.

### Choosing the recommender

- CLI: `-x/--crecsys <ClassName>` applies one recommender to all the agents.
- Population file: with `YClientWeb` each agent reads its own `rec_sys` (and `frec_sys`) from the population JSON, so agents can use different recommenders. YWeb writes this field from the *Content RecSys* menu of the forum client.
- The recommender is created when the agents are loaded; changing it requires restarting the client.

## Follow recommenders

Defined in:

- [`y_client/recsys/FollowRecSys.py`](../y_client/recsys/FollowRecSys.py)

Available strategies include:

- `FollowRecSys`
- `CommonNeighbors`
- `Jaccard`
- `AdamicAdar`
- `PreferentialAttachment`

### Notes

- `PreferentialAttachment` is the default in the main CLI.
- These recommenders call the `/follow_suggestions` API endpoint.

## Changing recommenders from the CLI

```bash
python y_client.py \
  -c config_files/config.json \
  -p config_files/prompts.json \
  -x HotRanking \
  -y Jaccard
```

## Programmatic example

```python
from y_client.recsys import HotRanking, PreferentialAttachment

content = HotRanking()
follow = PreferentialAttachment(leaning_bias=1.5)
experiment.set_recsys(content, follow)
```
