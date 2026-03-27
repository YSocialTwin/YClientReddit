from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "y_client" / "llm" / "autogen_compat.py"


def _load_module():
    module_name = "reddit_langchain_autogen_compat_under_test"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


compat_module = _load_module()
AssistantAgent = compat_module.AssistantAgent
MultimodalConversableAgent = compat_module.MultimodalConversableAgent


def test_assistant_agent_single_reply(monkeypatch):
    calls = []

    def fake_invoke_text_model(*, llm_config, system_prompt, user_prompt):
        calls.append((system_prompt, user_prompt))
        return f"{system_prompt}|{user_prompt}"

    monkeypatch.setattr(compat_module, "_invoke_text_model", fake_invoke_text_model)

    initiator = AssistantAgent(name="handler", system_message="handler", max_consecutive_auto_reply=0)
    peer = AssistantAgent(name="writer", system_message="writer", max_consecutive_auto_reply=1)

    initiator.initiate_chat(peer, message="prompt", max_turns=1, silent=True)

    assert initiator.chat_messages[peer] == [{"content": "writer|prompt"}]
    assert peer.chat_messages[initiator] == [{"content": "writer|prompt"}]
    assert initiator.last_message(peer) == {"content": "writer|prompt"}
    assert calls == [("writer", "prompt")]


def test_assistant_agent_two_step_exchange(monkeypatch):
    def fake_invoke_text_model(*, llm_config, system_prompt, user_prompt):
        return f"{system_prompt}>{user_prompt}"

    monkeypatch.setattr(compat_module, "_invoke_text_model", fake_invoke_text_model)

    initiator = AssistantAgent(name="handler", system_message="handler", max_consecutive_auto_reply=1)
    peer = AssistantAgent(name="writer", system_message="writer", max_consecutive_auto_reply=1)

    initiator.initiate_chat(peer, message="prompt", max_turns=1, silent=True)

    expected = [
        {"content": "writer>prompt"},
        {"content": "handler>writer>prompt"},
    ]
    assert initiator.chat_messages[peer] == expected
    assert peer.chat_messages[initiator] == expected
    assert initiator.last_message(peer) == expected[-1]


def test_multimodal_agent_wraps_text_response(monkeypatch):
    monkeypatch.setattr(
        compat_module,
        "_invoke_vision_model",
        lambda **kwargs: "vision result",
    )

    user_proxy = AssistantAgent(name="user", max_consecutive_auto_reply=0)
    vision_agent = MultimodalConversableAgent(name="vision", llm_config={})

    user_proxy.initiate_chat(vision_agent, message="Describe <img https://example.test/a.png>", silent=True)

    assert vision_agent.chat_messages[user_proxy] == [{"content": [{"text": "vision result"}]}]
