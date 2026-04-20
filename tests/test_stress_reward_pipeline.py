from y_client.classes.base_agent import Agent, _stress_reward_settings_from_config


class _FakeStressRewardSystem:
    def __init__(self):
        self.current_calls = []
        self.comment_calls = []
        self.churn_calls = []

    def compute_current_stress_reward(self, **kwargs):
        self.current_calls.append(kwargs)
        return {"stress": 0.3, "reward": 0.6}

    def compute_comment_delta(self, **kwargs):
        self.comment_calls.append(kwargs)
        return {"delta_stress": 0.12, "delta_reward": -0.04}

    def churn_enabled(self):
        return True

    def compute_churn_probability(self, **kwargs):
        self.churn_calls.append(kwargs)
        return 0.9


def test_stress_reward_settings_default_to_disabled():
    assert _stress_reward_settings_from_config({})["enabled"] is False
    assert _stress_reward_settings_from_config({"simulation": {"stress_reward": {"enabled": True}}})[
        "enabled"
    ] is True


def test_refresh_stress_reward_state_updates_agent_cache():
    agent = Agent.__new__(Agent)
    agent.user_id = 17
    agent.base_url = "http://example.test"
    agent.stress_reward_enabled = True
    agent.stress_reward_settings = {"backward_rounds": 9}
    agent.stress_reward_system = _FakeStressRewardSystem()
    agent.current_stress_reward = {"stress": 0.0, "reward": 0.0}
    agent._stress_reward_last_tid = None

    state = agent.refresh_stress_reward_state(11, force=True)

    assert state == {"stress": 0.3, "reward": 0.6}
    assert agent.current_stress == 0.3
    assert agent.current_reward == 0.6


def test_apply_stress_reward_comment_uses_annotation_and_persists_variations():
    agent = Agent.__new__(Agent)
    agent.user_id = 5
    agent.stress_reward_enabled = True
    agent.stress_reward_system = _FakeStressRewardSystem()
    agent.get_username_from_post = lambda post_id: (99, "peer")
    agent.refresh_stress_reward_state = (
        lambda tid, force=False, user_id=None: {"stress": 0.25, "reward": 0.4}
    )
    agent._annotate_stress_reward_text = lambda prompt_key, text, target_user_id=None: {
        "tone": "hostile",
        "directness": 0.8,
        "support_strength": 0.0,
    }
    persisted = []
    agent._persist_stress_reward_variations = (
        lambda target_user_id, tid, delta_payload, action=None: persisted.append(
            (target_user_id, tid, delta_payload, action)
        )
        or True
    )

    applied = agent._apply_stress_reward_comment(post_id=14, text="you are wrong", tid=7)

    assert applied is True
    assert agent.stress_reward_system.comment_calls == [
        {
            "tone": "hostile",
            "current_stress": 0.25,
            "current_reward": 0.4,
            "directness": 0.8,
            "support_strength": 0.0,
        }
    ]
    assert persisted == [
        (99, 7, {"delta_stress": 0.12, "delta_reward": -0.04}, "comment:hostile")
    ]


def test_evaluate_stress_reward_churn_marks_agent_left_when_probability_hits():
    agent = Agent.__new__(Agent)
    agent.user_id = 5
    agent.left_on = None
    agent.stress_reward_enabled = True
    agent.stress_reward_churn_enabled = True
    agent.stress_reward_system = _FakeStressRewardSystem()
    agent.refresh_stress_reward_state = (
        lambda tid, force=False, user_id=None: {"stress": 0.7, "reward": 0.1}
    )
    churned = []
    agent.churn_system = lambda tid: churned.append(tid) or '{"status": 200}'
    agent._stress_reward_clamp01 = Agent._stress_reward_clamp01

    class _FixedRng:
        @staticmethod
        def random():
            return 0.05

    assert agent.evaluate_stress_reward_churn(13, rng=_FixedRng()) is True
    assert agent.left_on == 13
    assert churned == [13]


def test_stress_prompt_block_uses_five_point_likert_scale():
    agent = Agent.__new__(Agent)
    agent.stress_reward_enabled = True
    agent.refresh_stress_reward_state = lambda tid, force=False, user_id=None: {
        "stress": 0.86,
        "reward": 0.1,
    }

    prompt_block = agent._stress_prompt_block(9)

    assert "extremely stressed" in prompt_block
    assert "5/5" in prompt_block


def test_stress_prompt_block_is_omitted_when_feature_disabled():
    agent = Agent.__new__(Agent)
    agent.stress_reward_enabled = False

    prompt = agent._append_stress_level_to_prompt(base_prompt="Write a comment", tid=3)

    assert prompt == "Write a comment"
