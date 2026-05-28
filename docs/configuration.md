# Configuration

The main runtime configuration lives in [`config_files/config.json`](/Users/rossetti/PycharmProjects/YClientReddit/config_files/config.json).

## Top-level sections

- `servers`
- `simulation`
- `agents`
- `posts`

## `servers`

This section defines external dependencies.

Current fields in the committed config:

- `llm`
- `llm_api_key`
- `llm_max_tokens`
- `llm_temperature`
- `llm_v`
- `llm_v_api_key`
- `llm_v_max_tokens`
- `llm_v_temperature`
- `api`

Typical meaning:

- `llm`: main text-generation endpoint
- `llm_v`: secondary or vision-capable endpoint
- `api`: simulation backend service used for posts, users, feeds, memory, and other actions

## `simulation`

This section controls the run envelope.

Key fields:

- `name`: experiment name, also used in output paths
- `client`: class name to instantiate from `y_client.clients`
- `days`: simulation duration in days
- `slots`: time slots per day
- `starting_agents`: initial population size
- `percentage_new_agents_iteration`
- `percentage_removed_agents_iteration`
- `hourly_activity`
- `actions_likelihood`
- `opinion_dynamics`

### `actions_likelihood`

This governs the action mix the agents choose from. In forum mode, `SHARE` is normalized to `0.0` inside the clients.

Current example keys:

- `post`
- `image`
- `comment`
- `read`
- `share`
- `search`
- `cast`
- `share_link`

### `opinion_dynamics`

This optional block enables topic-level opinion tracking and evolution.

Current committed keys:

- `enabled`
- `model_name`
- `parameters.epsilon`
- `parameters.mu`
- `parameters.theta`
- `parameters.cold_start`

Behavior notes:

- the committed default is disabled, which preserves current simulation behavior
- `bounded_confidence` is the active model implemented in the client
- the agent opinion state is persisted into the experiment database when the required tables exist

For the runtime model and examples, see [Opinion Dynamics](opinion-dynamics.md).

## `agents`

This is the largest section and controls agent generation and runtime behavior.

Representative groups:

### Persona generation

- `languages`
- `education_levels`
- `political_leanings`
- `nationalities`
- `interests`
- `toxicity_levels`
- `big_five`
- `age`
- `daily_actions`
- `round_actions`
- `n_interests`

### Reading and activity

- `max_length_thread_reading`
- `reading_from_follower_ratio`
- `probability_of_daily_follow`
- `probability_of_follow_back`
- `attention_window`

### LLM model selection

- `llm_agents`
- `llm_v_agent`

### Forum and reply shaping

- `forum_post_structure_strict`
- `reply_length_enforcement_enabled`
- `comment_max_chars`
- `comment_max_sentences`
- `post_max_chars`
- `post_max_sentences`
- `reply_rewrite_max_attempts`
- `reply_trim_fallback_enabled`
- `style_elaborate_enabled`
- `anti_repetition_enabled`
- `anti_repetition_window_comments`

### Memory configuration

The current checked-in config exposes some memory-focused toggles directly:

- `memory_high_affect_enabled`
- `memory_high_affect_rule_threshold`
- `memory_high_affect_uncertain_low`
- `memory_high_affect_uncertain_high`
- `memory_high_affect_search_k`
- `memory_high_affect_max_items`
- `memory_high_affect_max_chars`
- `memory_high_affect_callback_retry_count`
- `memory_high_affect_llm_fallback`
- `memory_nuance_planner_enabled`
- `memory_prompt_mode`
- `memory_reply_context_max_chars`
- `memory_vote_signal_only`
- `memory_cross_thread_callback_min_score`

For a deeper explanation of how these affect agent behavior, see [Memory System](memory.md).

### Stress/reward configuration

The client also accepts a top-level `stress_reward` block written by YWeb when the experiment enables that pipeline.

Representative shape:

```json
{
  "stress_reward": {
    "enabled": false,
    "backward_rounds": 24,
    "system": {
      "events": {},
      "coupling": {},
      "churn": {
        "enabled": false
      }
    }
  }
}
```

`enabled` activates the client-side stress/reward path. `backward_rounds` controls aggregate reconstruction, and the nested `system` block carries event weights, coupling coefficients, and optional churn settings.

See [Social Feedback Loop](social-feedback.md) for runtime behavior and the client/server split.

## `posts`

Current committed fields:

- `visibility_rounds`
- `emotions`

`visibility_rounds` shapes how long posts remain visible to recommenders and readers. `emotions` is present as a schema-like map of supported labels.

## Prompt configuration

Prompt templates live in [`config_files/prompts.json`](/Users/rossetti/PycharmProjects/YClientReddit/config_files/prompts.json).

They drive:

- comment style selection
- reply generation
- mention handling
- memory note generation
- thread/community summarization
- page/news posting flows

Because the prompt file is part of runtime behavior, version it together with config when comparing experiments.
