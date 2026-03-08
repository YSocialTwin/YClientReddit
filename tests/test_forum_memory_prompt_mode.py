import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "external" / "YClientReddit") not in sys.path:
    sys.path.insert(0, str(ROOT / "external" / "YClientReddit"))

from y_client.classes.base_agent import Agent  # noqa: E402


def _bare_agent():
    ag = Agent.__new__(Agent)
    ag.user_id = 42
    ag.name = "tester"
    ag.memory_enabled = True
    ag.memory_run_id = "run-1"
    ag.memory_semantic_enabled = True
    ag.memory_prompt_mode = "subtle_forum"
    ag.memory_reply_context_max_chars = 280
    ag.memory_cross_thread_callback_min_score = 0.80
    ag.memory_vote_signal_only = True
    ag.post_max_chars = 420
    ag.post_max_sentences = 3
    ag._memory_cache_digest = None
    return ag


def test_build_reply_context_filters_vote_language_and_limits_cross_thread_callbacks():
    ag = _bare_agent()
    ag._memory_fetch_context = lambda **kwargs: {
        "social_card": {
            "affinity": 0.8,
            "conflict": 0.2,
            "humor": 0.1,
            "trust": 0.4,
            "summary_text": "usually argue about finales. I upvoted their post once.",
        },
        "thread_card": {
            "gist_text": "debating whether trauma excuses what Wanda did",
        },
        "recent_pair_events": [
            {
                "thread_root_id": 900,
                "event_type": "upvote",
                "relation_label": "upvote",
                "salient_claim": "upvote on @alice: strong point",
            },
            {
                "thread_root_id": 900,
                "event_type": "comment",
                "relation_label": "disagree",
                "salient_claim": "pushed back on the idea that grief excuses harm",
            },
        ],
    }
    ag._memory_search = lambda **kwargs: {
        "retrieval_meta": {
            "degraded_mode": False,
            "embedding_degraded": False,
            "no_ready_candidates": False,
        },
        "items": [
            {
                "item_id": 1,
                "thread_root_id": 777,
                "other_user_id": 55,
                "score": 0.92,
                "text_humanized": "last week you two argued about whether Wanda's ending was deserved",
            },
            {
                "item_id": 2,
                "thread_root_id": 666,
                "other_user_id": 55,
                "score": 0.99,
                "text_humanized": "you upvoted @alice after that meme thread",
            },
        ],
    }

    memory_text, meta = ag._memory_build_reply_context(
        query_text="compose a reply",
        other_user_id=55,
        thread_root_id=900,
        other_username="alice",
        round_id=12,
    )

    assert "affinity=0.80" in memory_text
    assert "upvoted" not in memory_text.lower()
    assert "trauma excuses" in meta["continuity_text"]
    assert "grief excuses harm" in meta["continuity_text"]
    assert "wanda's ending" in meta["continuity_text"].lower()
    assert meta["cross_thread_callback_candidate"] is True
    assert meta["top_score"] == 0.92


def test_memory_after_vote_hides_vote_events_from_prompt_visible_memory():
    ag = _bare_agent()
    ag._memory_get_author_id_and_username = lambda post_id: (55, "alice")
    ag._memory_get_thread_root_id = lambda post_id: 900

    recorded_events = []
    social_updates = []
    digest_updates = []
    reflection_updates = []

    ag._memory_record_event = lambda **kwargs: recorded_events.append(kwargs)
    ag._memory_upsert_social_card = lambda **kwargs: social_updates.append(kwargs)
    ag._memory_maybe_update_community_digest = lambda **kwargs: digest_updates.append(kwargs)
    ag._memory_maybe_generate_reflections = lambda **kwargs: reflection_updates.append(kwargs)

    ag._memory_after_vote(tid=12, post_id=99, vote_type="like")

    assert recorded_events == []
    assert digest_updates == []
    assert reflection_updates == []
    assert len(social_updates) == 1
    assert social_updates[0]["include_evidence"] is False
    assert social_updates[0]["count_as_event"] is False
    assert social_updates[0]["relation_label"] is None
    assert social_updates[0]["salient_claim"] is None


def test_validate_structured_post_text_requires_title_and_body():
    ag = _bare_agent()

    valid = ag._validate_structured_post_text(
        "TITLE: wanda was still wrong\n\ntrauma explains it, it does not excuse it."
    )
    missing_title = ag._validate_structured_post_text(
        "trauma explains it, it does not excuse it."
    )
    missing_body = ag._validate_structured_post_text("TITLE: wanda was still wrong")

    assert valid["valid"] is True
    assert valid["title"] == "wanda was still wrong"
    assert valid["body"] == "trauma explains it, it does not excuse it."
    assert missing_title["valid"] is False
    assert "missing_title" in missing_title["reasons"]
    assert missing_body["valid"] is False
    assert "missing_body" in missing_body["reasons"]


def test_post_style_context_uses_style_fields_only_when_digest_is_mature():
    ag = _bare_agent()
    ag._memory_cache_digest = {
        "digest_text": "the nanny dominates everything here",
        "norms": ["short snarky complaint posts", "question bait titles"],
        "memes": ["sleeping-on takes"],
        "top_topics": ["the nanny reboot"],
    }
    ag._memory_get_recent_root_posts = lambda **kwargs: [
        {"id": 1, "user_id": 1, "tweet": "TITLE: one\n\nbody", "round": 1},
        {"id": 2, "user_id": 2, "tweet": "TITLE: two\n\nbody", "round": 2},
        {"id": 3, "user_id": 3, "tweet": "TITLE: three\n\nbody", "round": 3},
        {"id": 4, "user_id": 4, "tweet": "TITLE: four\n\nbody", "round": 4},
        {"id": 5, "user_id": 1, "tweet": "TITLE: five\n\nbody", "round": 5},
        {"id": 6, "user_id": 2, "tweet": "TITLE: six\n\nbody", "round": 6},
    ]

    text_value, meta = ag._memory_build_post_style_context(tid=6)

    assert meta["mature"] is True
    assert meta["usage"] == "style_digest_only"
    assert "short snarky complaint posts" in text_value
    assert "question bait titles" in text_value
    assert "the nanny" not in text_value.lower()


def test_post_style_context_stays_off_before_digest_maturity():
    ag = _bare_agent()
    ag._memory_cache_digest = {
        "norms": ["short snarky complaint posts"],
        "memes": ["sleeping-on takes"],
    }
    ag._memory_get_recent_root_posts = lambda **kwargs: [
        {"id": 1, "user_id": 1, "tweet": "TITLE: one\n\nbody", "round": 1},
        {"id": 2, "user_id": 2, "tweet": "TITLE: two\n\nbody", "round": 2},
        {"id": 3, "user_id": 1, "tweet": "TITLE: three\n\nbody", "round": 3},
    ]

    text_value, meta = ag._memory_build_post_style_context(tid=3)

    assert text_value == ""
    assert meta["mature"] is False
    assert meta["usage"] == "none"


def test_thread_browse_context_ignores_community_digest():
    ag = _bare_agent()
    ag._memory_fetch_context = lambda **kwargs: {
        "community_digest": {"digest_text": "the nanny dominates everything here"},
        "thread_card": {
            "gist_text": "people are arguing about whether the merger will ruin discovery",
            "my_role": "skeptic",
        },
    }

    text_value, meta = ag._memory_build_thread_browse_context(thread_root_id=9, tid=5)

    assert meta["usage"] == "thread_local"
    assert "thread gist" in text_value
    assert "the nanny" not in text_value.lower()


def test_post_topic_freshness_matches_recent_text_roots():
    ag = _bare_agent()
    ag._memory_get_recent_root_posts = lambda **kwargs: [
        {
            "id": 7,
            "round": 4,
            "user_id": 101,
            "news_id": None,
            "image_post_id": None,
            "image_id": None,
            "tweet": "TITLE: The Nanny reboot is a hot mess\n\nThe reboot is a disaster waiting to happen.",
        },
        {
            "id": 8,
            "round": 4,
            "user_id": 102,
            "news_id": 55,
            "image_post_id": None,
            "image_id": None,
            "tweet": "TITLE: streaming merger news\n\nHuge content bundle coming.",
        },
    ]

    fingerprint, matches = ag._post_find_recent_topic_matches(
        text_value="TITLE: why the nanny reboot is a disaster\n\nThis reboot is still a hot mess.",
        tid=9,
    )

    assert fingerprint["title"] == "why the nanny reboot is a disaster"
    assert matches
    assert matches[0]["post_id"] == 7
