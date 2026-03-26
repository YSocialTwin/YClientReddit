# Opinion Dynamics

Opinion dynamics support is available in the Reddit client and is disabled by default.

When enabled, each agent maintains a per-topic scalar opinion and updates it after selected interactions using a bounded-confidence rule. The implementation was ported from the main `YClient` repository so both clients can evolve topic stance in a consistent way.

## Current model

The currently supported model is `bounded_confidence`.

For a topic with current opinion `x` and observed author opinion `y`, the agent updates only when `|x - y| <= epsilon`:

```text
x_next = clamp(x + mu * (y - x) + theta, 0.0, 1.0)
```

Parameters:

- `epsilon`: confidence bound for whether influence applies
- `mu`: convergence strength toward the observed opinion
- `theta`: additive drift term applied on update
- `cold_start`: initialization strategy for missing topic opinions

Supported `cold_start` values in the current port:

- `neutral`: initialize to `0.5`
- `random`: initialize uniformly in `[0, 1]`
- `positive`: initialize to `0.75`
- `negative`: initialize to `0.25`
- `author`: initialize from the observed author opinion when available

## Configuration

Opinion dynamics is configured under `simulation.opinion_dynamics` in [`config_files/config.json`](/Users/rossetti/PycharmProjects/YClientReddit/config_files/config.json).

Example:

```json
{
  "simulation": {
    "opinion_dynamics": {
      "enabled": true,
      "model_name": "bounded_confidence",
      "parameters": {
        "epsilon": 0.25,
        "mu": 0.5,
        "theta": 0.0,
        "cold_start": "neutral"
      }
    }
  }
}
```

The checked-in default keeps `enabled` set to `false`, so existing runs do not change unless you opt in explicitly.

## Runtime behavior

The current port adds three main behaviors:

1. Agent state can persist an `opinions` map keyed by topic name.
2. Initial topic opinions can be seeded into the experiment database on first load.
3. Reply and reaction flows can record updated opinions into `agent_opinion`.

The implementation relies on the experiment database containing:

- `interests`
- `rounds`
- `agent_opinion`

If those tables are unavailable, the logic exits quietly and leaves the rest of the simulation unchanged.

## Persistence model

Opinion state is represented in two places:

- in-memory on the agent as `agent.opinions`
- in the experiment database as rows in `agent_opinion`

Seed behavior:

- when an agent with configured opinions is loaded
- when opinion dynamics is enabled
- when no previous `agent_opinion` rows exist for that agent

Interaction behavior:

- self-authored posts can record the agent's current opinions for the post topics
- comments can update opinions from the post author's latest recorded topic opinion
- reactions can do the same

## Compatibility notes

The port is intentionally conservative:

- the feature is default-off
- unsupported `model_name` values currently fall back to `bounded_confidence`
- failures in opinion recording are swallowed at the action-call sites so posting and reaction flows do not fail

That keeps the existing simulation behavior stable while enabling opinion tracking for experiments that explicitly opt in.

## Tests

Focused regression coverage lives in [`tests/test_opinion_dynamics.py`](/Users/rossetti/PycharmProjects/YClientReddit/tests/test_opinion_dynamics.py).

Current checks cover:

- initial seeding into `agent_opinion`
- bounded-confidence updates inside the confidence radius
- no convergence when the author opinion falls outside the configured radius
