"""Opt-in community actions of the forum agents: found a new community, cross-post to a related one."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from y_client import communities, prompt_store  # noqa: E402
from y_client.classes import base_agent  # noqa: E402
from y_client.classes.base_agent import Agent  # noqa: E402


@pytest.fixture(autouse=True)
def _off():
    yield
    communities.install({})


def _on(**sections):
    communities.install({"communities": sections})


def test_nothing_is_active_without_the_section():
    communities.install({})
    assert not communities.creation_enabled() and not communities.crosspost_enabled()
    assert not communities.draw_creation() and not communities.draw_crosspost()


def test_probabilities_and_limits_are_read_and_bounded():
    _on(creation={"enabled": True, "probability": 7, "max_topics": 12}, crosspost={"enabled": True, "probability": -1, "max_per_post": 3})
    assert communities.draw_creation()  # probability above 1 means always
    assert not communities.draw_crosspost()  # below 0 means never
    assert communities.max_topics() == 12 and communities.max_per_post() == 3


def test_parse_name():
    assert communities.parse_name("y/Home Brewing\n(because)") == "Home Brewing"
    assert communities.parse_name('"Retro Gaming".') == "Retro Gaming"
    assert communities.parse_name("NONE") is None and communities.parse_name("") is None
    assert len(communities.parse_name("A very long community name about everything")) <= 20


def test_parse_choice_keeps_only_existing_communities_in_order():
    names = ["Windows", "Winter", "Gaming"]
    assert communities.parse_choice("y/windows, Gaming, Cooking", names, limit=5) == ["Windows", "Gaming"]
    assert communities.parse_choice("Gaming, Windows", names, limit=1) == ["Gaming"]
    assert communities.parse_choice("NONE", names) == [] and communities.parse_choice(None, names) == []


def test_crosspost_text_keeps_the_title_and_fits_a_post():
    text = communities.crosspost_text("TITLE: kernel 7 is out\n\n" + "body " * 200, "Cross-posted from y/Linux")
    assert text.startswith("TITLE: kernel 7 is out") and text.endswith("Cross-posted from y/Linux")
    assert len(text) <= 500


class _Resp:
    def __init__(self, payload):
        self.__dict__["_content"] = json.dumps(payload).encode()


def _agent(reply, topics, monkeypatch, created=None, crossposted=None):
    """The community methods of an Agent on a fake server and a fake model."""
    calls = {"asked": []}

    def fake_get(url, **kw):
        assert url.endswith("/get_topics")
        return _Resp(topics)

    def fake_post(url, **kw):
        body = json.loads(kw["data"])
        if url.endswith("/create_topic"):
            calls.setdefault("create", []).append(body)
            return _Resp({"status": 200, "id": 99, "name": body["name"], "created": True})
        if url.endswith("/crosspost_thread"):
            calls.setdefault("crosspost", []).append(body)
            return _Resp({"status": 200, "id": 500 + len(calls["crosspost"])})
        raise AssertionError(url)

    monkeypatch.setattr(base_agent, "get", fake_get)
    monkeypatch.setattr(base_agent, "post", fake_post)
    me = SimpleNamespace(name="alice", user_id=7, base_url="http://srv", interests=["Linux", "Programming"],
                         age=30, gender="f", nationality="it", profession="dev", leaning="left", education_level="phd")

    def ask(prompt):
        calls["asked"].append(prompt)
        return reply

    me._community_ask = ask
    me._community_profile = lambda: "profile"
    me._community_list = lambda: Agent._community_list(me)
    return me, calls


def test_found_a_community_creates_it_and_uses_it_for_the_post(monkeypatch):
    _on(creation={"enabled": True, "probability": 1.0})
    me, calls = _agent("Home Brewing", [{"id": 1, "name": "Linux"}], monkeypatch)
    names, ids = Agent._community_found(me, 5, ["Linux"])
    assert (names, ids) == (["Home Brewing"], [99])
    assert calls["create"][0]["name"] == "Home Brewing" and calls["create"][0]["tid"] == 5
    assert "Linux" in calls["asked"][0]  # the model sees the existing communities


def test_found_a_community_does_nothing_when_off_none_duplicate_or_at_the_cap(monkeypatch):
    me, calls = _agent("Home Brewing", [{"id": 1, "name": "Linux"}], monkeypatch)
    assert Agent._community_found(me, 1, ["Linux"]) is None and calls["asked"] == []  # off
    _on(creation={"enabled": True, "probability": 1.0})
    me, calls = _agent("NONE", [{"id": 1, "name": "Linux"}], monkeypatch)
    assert Agent._community_found(me, 1, ["Linux"]) is None and "create" not in calls
    me, calls = _agent("linux", [{"id": 1, "name": "Linux"}], monkeypatch)
    assert Agent._community_found(me, 1, ["Linux"]) is None and "create" not in calls
    _on(creation={"enabled": True, "probability": 1.0, "max_topics": 1})
    me, calls = _agent("Home Brewing", [{"id": 1, "name": "Linux"}], monkeypatch)
    assert Agent._community_found(me, 1, ["Linux"]) is None and calls["asked"] == []


def test_crosspost_goes_only_to_the_communities_the_model_names(monkeypatch):
    _on(crosspost={"enabled": True, "probability": 1.0, "max_per_post": 2})
    topics = [{"id": 1, "name": "Linux"}, {"id": 2, "name": "Windows"}, {"id": 3, "name": "Winter"}]
    me, calls = _agent("Windows, Cooking", topics, monkeypatch)
    made = Agent._community_crosspost(me, 42, "TITLE: kernel 7\n\nbody", ["Linux"], 5)
    assert len(made) == 1
    sent = calls["crosspost"][0]
    assert (sent["post_id"], sent["topic_id"], sent["user_id"]) == (42, 2, 7)
    assert "Winter" in calls["asked"][0] and "- Linux" not in calls["asked"][0]  # the source is not a candidate
    assert "Cross-posted from y/Linux" in sent["text"]


def test_crosspost_does_nothing_when_off_unsure_or_without_model(monkeypatch):
    topics = [{"id": 1, "name": "Linux"}, {"id": 2, "name": "Windows"}]
    me, calls = _agent("Windows", topics, monkeypatch)
    assert Agent._community_crosspost(me, 42, "x", ["Linux"], 1) == [] and calls["asked"] == []  # off
    _on(crosspost={"enabled": True, "probability": 1.0})
    me, calls = _agent("NONE", topics, monkeypatch)
    assert Agent._community_crosspost(me, 42, "x", ["Linux"], 1) == [] and "crosspost" not in calls
    me, calls = _agent(None, topics, monkeypatch)  # no usable model: the reply is None
    assert Agent._community_crosspost(me, 42, "x", ["Linux"], 1) == []
    assert Agent._community_crosspost(me, None, "x", ["Linux"], 1) == []  # the server gave no post id


def test_prompts_exist_in_english_and_italian():
    for key in ("community_create", "community_crosspost", "community_crosspost_note"):
        assert prompt_store.get(key, "en") and prompt_store.get(key, "it")
