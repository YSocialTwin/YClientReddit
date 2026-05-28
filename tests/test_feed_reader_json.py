import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from y_client.news_feeds.feed_reader import parse_feed_with_retry


def test_parse_feed_with_retry_accepts_reddit_listing_json(monkeypatch):
    json_payload = """
    {
      "data": {
        "children": [
          {
            "data": {
              "title": "Policy article",
              "url_overridden_by_dest": "https://example.org/policy",
              "selftext": "context"
            }
          },
          {
            "data": {
              "title": "Thread-only post",
              "permalink": "/r/politics/comments/abc123/thread_only/"
            }
          }
        ]
      }
    }
    """

    response = SimpleNamespace(
        status_code=200,
        text=json_payload,
        headers={"Content-Type": "application/json"},
    )

    monkeypatch.setattr(
        "y_client.news_feeds.feed_reader.requests.get",
        lambda *args, **kwargs: response,
    )

    feed, used_url, error = parse_feed_with_retry(
        "https://www.reddit.com/r/politics/new.json?limit=100",
        require_entries=True,
    )

    assert error is None
    assert used_url == "https://www.reddit.com/r/politics/new.json?limit=100"
    assert getattr(feed, "bozo", False) is False
    assert len(feed.entries) == 2
    assert feed.entries[0].title == "Policy article"
    assert feed.entries[0].link == "https://example.org/policy"
    assert feed.entries[0].summary == "context"
    assert (
        feed.entries[1].link
        == "https://www.reddit.com/r/politics/comments/abc123/thread_only/"
    )
