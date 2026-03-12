# Recommenders

## Overview

Recommendation logic is defined in [`y_client/recsys/`](/Users/rossetti/PycharmProjects/YClientReddit/y_client/recsys). These classes are thin strategy wrappers that encode request parameters and delegate actual ranking or suggestion work to the API server.

## Content recommenders

Defined in:

- [`y_client/recsys/ContentRecSys.py`](/Users/rossetti/PycharmProjects/YClientReddit/y_client/recsys/ContentRecSys.py)

Available strategies include:

- `ContentRecSys`
- `ReverseChrono`
- `ReverseChronoPopularity`
- `ReverseChronoFollowers`
- `ReverseChronoFollowersPopularity`
- `HotRanking`
- `TopRanking`

### Notes

- `ReverseChronoFollowersPopularity` is the default in the main CLI.
- `HotRanking` and `TopRanking` are more Reddit-like ranking modes.
- All recommenders eventually call API endpoints such as `/read`, `/read_mentions`, and `/search`.

## Follow recommenders

Defined in:

- [`y_client/recsys/FollowRecSys.py`](/Users/rossetti/PycharmProjects/YClientReddit/y_client/recsys/FollowRecSys.py)

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
