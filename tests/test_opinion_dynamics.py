import json
import importlib.util
import sqlite3
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stub_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _identity_decorator(func=None, **kwargs):
    if func is None:
        return lambda inner: inner
    return func


def _load_base_agent_module():
    _stub_module("y_client", __path__=[])
    _stub_module("y_client.recsys", __path__=[])
    _stub_module("y_client.news_feeds", __path__=[])
    _stub_module("y_client.classes", __path__=[])
    _stub_module("y_client.recsys.ContentRecSys", ContentRecSys=type("ContentRecSys", (), {}))
    _stub_module("y_client.recsys.FollowRecSys", FollowRecSys=type("FollowRecSys", (), {}))
    _stub_module(
        "y_client.news_feeds.client_modals",
        Websites=type("Websites", (), {}),
        Images=type("Images", (), {}),
        Articles=type("Articles", (), {}),
        session=None,
        Agent_Custom_Prompt=type("Agent_Custom_Prompt", (), {}),
        ImagePosts=type("ImagePosts", (), {}),
    )
    _stub_module("y_client.classes.annotator", Annotator=type("Annotator", (), {}))
    _stub_module("y_client.logger", log_execution_time=_identity_decorator)
    _stub_module("y_client.news_feeds.feed_reader", NewsFeed=type("NewsFeed", (), {}))
    _stub_module("y_client.classes.time", SimulationSlot=type("SimulationSlot", (), {}))
    _stub_module("y_client.memory_runtime", build_agent_memory_engine=lambda *args, **kwargs: None)
    _stub_module(
        "y_client.llm",
        AssistantAgent=type("AssistantAgent", (), {}),
        MultimodalConversableAgent=type("MultimodalConversableAgent", (), {}),
    )
    _stub_module(
        "yclient_memory.contracts",
        BrowseMemoryRequest=type("BrowseMemoryRequest", (), {}),
        CommentMemoryEvent=type("CommentMemoryEvent", (), {}),
        PostMemoryEvent=type("PostMemoryEvent", (), {}),
        PostStyleRequest=type("PostStyleRequest", (), {}),
        ReplyMemoryRequest=type("ReplyMemoryRequest", (), {}),
        VoteMemoryEvent=type("VoteMemoryEvent", (), {}),
    )
    _stub_module("sqlalchemy", text=lambda value: value)
    _stub_module("sqlalchemy.sql", __path__=[])
    _stub_module("sqlalchemy.sql.expression", func=object())

    module_name = "opinion_dynamics_base_agent_under_test"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(
        module_name,
        ROOT / "y_client" / "classes" / "base_agent.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


base_agent_module = _load_base_agent_module()
Agent = base_agent_module.Agent


def _make_db(path: Path):
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE rounds (
            id INTEGER PRIMARY KEY,
            day INTEGER,
            hour INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE interests (
            iid INTEGER PRIMARY KEY,
            interest TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE agent_opinion (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id INTEGER,
            tid INTEGER,
            topic_id INTEGER,
            id_interacted_with INTEGER,
            id_post INTEGER,
            opinion REAL
        )
        """
    )
    conn.commit()
    conn.close()


def _bare_agent(db_path: Path):
    ag = Agent.__new__(Agent)
    ag.user_id = 7
    ag.name = "tester"
    ag.is_page = 0
    ag.interests = ["topic a", "topic b"]
    ag.opinions = {}
    ag.experiment_db_path = str(db_path)
    ag.opinions_enabled = True
    ag.opinion_dynamics = {
        "enabled": True,
        "model_name": "bounded_confidence",
        "parameters": {
            "epsilon": 0.5,
            "mu": 0.5,
            "theta": 0.1,
            "cold_start": "neutral",
        },
    }
    ag.base_url = "http://example.test"
    return ag


def test_seed_initial_opinions_if_needed(tmp_path):
    db_path = tmp_path / "opinions.db"
    _make_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO rounds (id, day, hour) VALUES (1, 0, 0)")
    conn.execute("INSERT INTO interests (iid, interest) VALUES (1, 'topic a')")
    conn.execute("INSERT INTO interests (iid, interest) VALUES (2, 'topic b')")
    conn.commit()
    conn.close()

    ag = _bare_agent(db_path)
    ag.opinions = {"topic a": 0.8, "topic b": 0.2}

    ag._seed_initial_opinions_if_needed()

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT topic_id, opinion, tid FROM agent_opinion WHERE agent_id = ? ORDER BY topic_id",
        (ag.user_id,),
    ).fetchall()
    conn.close()

    assert rows == [(1, 0.8, 1), (2, 0.2, 1)]


def test_new_opinions_updates_with_bounded_confidence(tmp_path, monkeypatch):
    db_path = tmp_path / "opinions.db"
    _make_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO interests (iid, interest) VALUES (1, 'topic a')")
    conn.execute(
        """
        INSERT INTO agent_opinion (agent_id, tid, topic_id, id_interacted_with, id_post, opinion)
        VALUES (2, 4, 1, 2, 99, 0.6)
        """
    )
    conn.commit()
    conn.close()

    ag = _bare_agent(db_path)
    ag.opinions = {"topic a": 0.2}
    ag.get_user_from_post = lambda post_id: 2

    class _Resp:
        def __init__(self, payload):
            self.__dict__["_content"] = json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(base_agent_module, "get", lambda *args, **kwargs: _Resp([1]))

    ag.new_opinions(post_id=99, tid=5, text="reply")

    assert round(ag.opinions["topic a"], 3) == 0.5

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        """
        SELECT opinion, id_interacted_with, id_post
        FROM agent_opinion
        WHERE agent_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (ag.user_id,),
    ).fetchone()
    conn.close()

    assert row == (0.5, 2, 99)


def test_new_opinions_keeps_value_when_author_outside_confidence_bound(tmp_path, monkeypatch):
    db_path = tmp_path / "opinions.db"
    _make_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO interests (iid, interest) VALUES (1, 'topic a')")
    conn.execute(
        """
        INSERT INTO agent_opinion (agent_id, tid, topic_id, id_interacted_with, id_post, opinion)
        VALUES (2, 4, 1, 2, 99, 0.95)
        """
    )
    conn.commit()
    conn.close()

    ag = _bare_agent(db_path)
    ag.opinions = {"topic a": 0.2}
    ag.get_user_from_post = lambda post_id: 2

    class _Resp:
        def __init__(self, payload):
            self.__dict__["_content"] = json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(base_agent_module, "get", lambda *args, **kwargs: _Resp([1]))

    ag.new_opinions(post_id=99, tid=5, text="reply")

    assert round(ag.opinions["topic a"], 3) == 0.2

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        """
        SELECT opinion
        FROM agent_opinion
        WHERE agent_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (ag.user_id,),
    ).fetchone()
    conn.close()

    assert row == (0.2,)
