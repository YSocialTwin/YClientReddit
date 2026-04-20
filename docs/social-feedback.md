# Social Feedback Loop

YClientReddit now includes the same high-level social feedback features introduced in the microblogging client, adapted to forum-style content and threads:

- stress/reward reconstruction with optional churn
- reciprocal follow and unfollow behavior

As in the rest of the stack, the client remains responsible for LLM work and behavioral decisions, while the server remains responsible for database persistence.

## Stress, Reward, And Churn

The forum client reads a top-level `stress_reward` block from the generated client configuration. The structure mirrors the one used by the microblogging client:

```json
{
  "stress_reward": {
    "enabled": true,
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

When enabled, the client refreshes the acting user’s aggregate `stress` and `reward` values before action execution. If `stress_reward.system.churn.enabled` is also true, the client computes a churn probability from the current aggregate state and may ask the server to set `left_on` for that user.

Directed forum interactions then produce variation updates for the target user. In the current implementation, this applies to:

- reactions
- comments
- shares where enabled by the runtime

For textual interactions, the client can call the prompt templates:

- `agent_comment_stress_reward_annotation`
- `agent_post_stress_reward_annotation`

Those prompts live in `config_files/prompts.json` and extract structured fields used by the forum-side `StressRewardSystem`.

## Reciprocal Follow And Unfollow

Forum simulations now support follow-back and unfollow-back decisions after a real follow or unfollow action has occurred.

The main tuning knob is:

- `agents.probability_of_follow_back`

If the probability gate passes, the affected peer evaluates the reciprocal edge. Rule-based agents use only the configured probability. LLM-backed agents also inspect the profile of the initiating user before deciding.

This mechanism applies to:

- direct follow/unfollow actions
- secondary follow actions triggered from content interactions

Before submitting the reciprocal edge update, the client asks the server whether the reverse edge already exists through `/check_follow_relationship`. That prevents duplicate follow rows and no-op unfollows.

## Why This Matters In Forum Mode

These features make the forum runtime more responsive to thread-level interaction. Reply pressure, supportive or hostile signals, and reciprocal relationship changes now feed back into the user state and network structure instead of remaining isolated per action.
