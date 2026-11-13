"""Follow probabilities have a default on every agent, whichever initialiser built it."""

from y_client.classes.base_agent import Agent
from y_client.classes.fake_base_agent import FakeAgent


def test_follow_probabilities_default_to_zero_on_the_class():
    for cls in (Agent, FakeAgent):
        assert cls.probability_of_daily_follow == 0.0
        assert cls.probability_of_secondary_follow == 0.0
        assert cls.probability_of_follow_back == 0.0


def test_an_instance_without_the_attribute_set_reads_the_default():
    agent = FakeAgent.__new__(FakeAgent)
    assert agent.probability_of_secondary_follow == 0.0
