import json

from y_client.classes.base_agent import Agent


class _FakeAssistantAgent:
    response_text = "NONE"

    def __init__(self, *args, **kwargs):
        self.chat_messages = {}

    def initiate_chat(self, other, message=None, silent=True, max_turns=1):
        other.chat_messages[self] = [{"content": self.response_text}]

    def reset(self):
        return None


def _build_agent():
    agent = Agent.__new__(Agent)
    agent.name = "tester"
    agent.user_id = 21
    agent.base_url = "http://example.test"
    agent.llm_config = {"config_list": [{"model": "llama3.2"}]}
    agent.prompts = {
        "agent_roleplay_simple": "persona",
        "agent_roleplay_base": "persona",
        "handler_instructions_simple": "handler",
        "handler_action": "{actions}",
    }
    agent._Agent__effify = lambda template, **kwargs: template.format(**kwargs) if kwargs else template
    agent._Agent__get_post = lambda post_id: "toxic forum comment"
    return agent


def test_report_posts_server_payload(monkeypatch):
    agent = _build_agent()
    calls = []

    monkeypatch.setattr("y_client.classes.base_agent.AssistantAgent", _FakeAssistantAgent)
    monkeypatch.setattr(
        "y_client.classes.base_agent.post",
        lambda url, headers=None, data=None: calls.append((url, json.loads(data))),
    )
    _FakeAssistantAgent.response_text = "REPORT_OFFENSIVE"

    report_type = agent.report(post_id=3, tid=8)

    assert report_type == "offensive"
    assert calls[0][0].endswith("/report")
    assert calls[0][1] == {"user_id": 21, "post_id": 3, "type": "offensive", "tid": 8}


def test_read_action_reports_after_reaction(monkeypatch):
    agent = _build_agent()
    reaction_calls = []
    report_calls = []

    monkeypatch.setattr("y_client.classes.base_agent.AssistantAgent", _FakeAssistantAgent)
    monkeypatch.setattr("y_client.classes.base_agent.random.sample", lambda seq, n: [seq[0]])
    _FakeAssistantAgent.response_text = "READ"
    agent.read = lambda: json.dumps([12])
    agent.reaction = lambda post_id, tid, check_follow=True: reaction_calls.append(
        (post_id, tid, check_follow)
    )
    agent.report = lambda post_id, tid: report_calls.append((post_id, tid))

    agent.select_action(tid=6, actions=["READ"])

    assert reaction_calls == [(12, 6, True)]
    assert report_calls == [(12, 6)]
