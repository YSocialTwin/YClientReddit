# Testing

## Test suite location

Tests are under [`tests/`](/Users/rossetti/PycharmProjects/YClientReddit/tests).

Current focused test files include:

- [`tests/test_forum_memory_prompt_mode.py`](/Users/rossetti/PycharmProjects/YClientReddit/tests/test_forum_memory_prompt_mode.py)
- [`tests/test_high_affect_memory_callback.py`](/Users/rossetti/PycharmProjects/YClientReddit/tests/test_high_affect_memory_callback.py)
- [`tests/test_forum_reply_quality.py`](/Users/rossetti/PycharmProjects/YClientReddit/tests/test_forum_reply_quality.py)
- [`tests/test_length_enforcement.py`](/Users/rossetti/PycharmProjects/YClientReddit/tests/test_length_enforcement.py)
- [`tests/test_opinion_dynamics.py`](/Users/rossetti/PycharmProjects/YClientReddit/tests/test_opinion_dynamics.py)

## What the tests emphasize

The current suite is most detailed around forum reply behavior and memory:

- subtle memory prompt mode
- high-affect callback logic
- reply quality gates
- length enforcement
- opinion-dynamics seeding and bounded-confidence updates

## Running tests

If `pytest` is available in your environment:

```bash
pytest tests
```

Run a focused subset:

```bash
pytest tests/test_forum_memory_prompt_mode.py
pytest tests/test_high_affect_memory_callback.py
pytest tests/test_opinion_dynamics.py
```

## Suggested documentation-time smoke checks

When changing docs or examples, the most useful behavioral tests are:

```bash
pytest tests/test_forum_memory_prompt_mode.py
pytest tests/test_high_affect_memory_callback.py
pytest tests/test_length_enforcement.py
pytest tests/test_opinion_dynamics.py
```

These cover the most user-visible forum, memory, and opinion-evolution guardrails.
