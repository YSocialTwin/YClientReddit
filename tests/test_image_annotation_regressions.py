from __future__ import annotations

from types import SimpleNamespace

from y_client.classes.annotator import Annotator
from y_client.classes.base_agent import Agent


def test_annotator_accepts_string_payload():
    ann = Annotator.__new__(Annotator)
    class _Proxy:
        def initiate_chat(self, *args, **kwargs):
            return None

    ann.user_proxy = _Proxy()
    ann.image_agent = SimpleNamespace(chat_messages={ann.user_proxy: [{"content": "plain description"}]})

    assert ann.annotate("http://example.test/image.png") == "plain description"


def test_select_image_uses_uppercase_news_action_key(monkeypatch):
    agent = Agent.__new__(Agent)
    agent.actions_likelihood = {"NEWS": 0.5}
    agent.llm_v_config = {
        "model": "minicpm-v:latest",
        "url": "http://127.0.0.1:11434/v1",
        "api_key": "EMPTY",
        "temperature": 0.2,
        "max_tokens": 128,
    }
    agent.user_id = 7
    agent.base_url = "http://example.test"
    image = SimpleNamespace(id=9, url="http://example.test/image.png", description=None, remote_article_id=None)
    article = SimpleNamespace(title="Test", summary="Summary", link="http://example.test/a")
    website = SimpleNamespace(
        name="Example",
        rss="http://example.test/rss",
        leaning="center",
        country="IT",
        language="en",
        category="news",
        last_fetched="2026-03-30",
    )

    monkeypatch.setattr(agent, "select_news", lambda: (article, website), raising=False)
    monkeypatch.setattr(
        agent,
        "news",
        lambda tid, article, website: SimpleNamespace(_content=b'{"article_id": 44}'),
        raising=False,
    )

    import y_client.classes.base_agent as base_agent_module

    monkeypatch.setattr(base_agent_module.content_store, "get_random_image", lambda: None)
    monkeypatch.setattr(base_agent_module.content_store, "get_image_by_article_id", lambda article_id: image)
    monkeypatch.setattr(
        base_agent_module.content_store,
        "save_image_remote_article",
        lambda image_id, remote_article_id: SimpleNamespace(
            id=image_id,
            url=image.url,
            description=image.description,
            remote_article_id=remote_article_id,
        ),
    )

    class FakeAnnotator:
        def __init__(self, config):
            self.config = config

        def annotate(self, image_url):
            return "described image"

    monkeypatch.setattr(base_agent_module, "Annotator", FakeAnnotator)
    monkeypatch.setattr(
        base_agent_module.content_store,
        "save_image_description",
        lambda image_id, description: SimpleNamespace(
            id=image_id,
            url=image.url,
            description=description,
            remote_article_id=44,
        ),
    )

    selected_image, article_id = agent.select_image(tid=5)

    assert article_id == 44
    assert selected_image.description == "described image"
    assert selected_image.remote_article_id == 44
