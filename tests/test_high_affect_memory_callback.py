import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "external" / "YClientReddit") not in sys.path:
    sys.path.insert(0, str(ROOT / "external" / "YClientReddit"))

from y_client.classes.base_agent import Agent  # noqa: E402


def _bare_agent():
    ag = Agent.__new__(Agent)
    ag.user_id = 101
    ag.name = "tester"
    ag.memory_enabled = True
    ag.memory_semantic_enabled = True
    ag.memory_high_affect_enabled = True
    ag.memory_high_affect_rule_threshold = 0.55
    ag.memory_high_affect_uncertain_low = 0.35
    ag.memory_high_affect_uncertain_high = 0.70
    ag.memory_high_affect_search_k = 12
    ag.memory_high_affect_max_items = 6
    ag.memory_high_affect_max_chars = 900
    ag.memory_high_affect_llm_fallback = False
    return ag


def test_extract_last_thread_message_parses_reddit_lines():
    ag = _bare_agent()
    txt, author = ag._extract_last_thread_message("@alice - first\n@bob - second\n")
    assert txt == "second"
    assert author == "bob"


def test_detect_high_affect_rules_for_criticism_and_conflict():
    ag = _bare_agent()
    ag.memory_high_affect_llm_fallback = False
    ag._memory_detect_prior_opinion_match = lambda **kwargs: (False, None)

    signal = ag._detect_high_affect_signal(
        incoming_text="you are wrong, source for that claim?",
        thread_context="thread",
        other_user_id=202,
        thread_root_id=303,
        round_id=9,
        target_username="alice",
    )

    assert signal["is_high_affect"] is True
    assert signal["triggers"]["criticism_or_challenge"] is True
    assert signal["triggers"]["conflict_or_argument"] is True


def test_detect_high_affect_defending_prior_opinion_with_memory_match():
    ag = _bare_agent()
    ag._memory_detect_prior_opinion_match = lambda **kwargs: (True, {"item_id": 77})

    signal = ag._detect_high_affect_signal(
        incoming_text="you said this last time too",
        thread_context="thread",
        other_user_id=202,
        thread_root_id=303,
        round_id=9,
        target_username="alice",
    )

    assert signal["triggers"]["defending_prior_opinion"] is True
    assert signal["prior_match"] is True


def test_detect_high_affect_uses_llm_fallback_in_uncertain_band():
    ag = _bare_agent()
    ag.memory_high_affect_llm_fallback = True
    ag._memory_detect_prior_opinion_match = lambda **kwargs: (False, None)
    ag._memory_llm_high_affect_classifier = lambda **kwargs: {
        "criticism_or_challenge": False,
        "conflict_or_argument": True,
        "incoming_anecdote": False,
        "defending_prior_opinion": False,
        "confidence": 0.82,
    }

    signal = ag._detect_high_affect_signal(
        incoming_text="you sure, source?",
        thread_context="thread",
        other_user_id=202,
        thread_root_id=303,
        round_id=9,
        target_username="alice",
    )

    assert signal["source"] == "hybrid"
    assert signal["used_llm_fallback"] is True
    assert signal["is_high_affect"] is True


def test_collect_high_affect_recall_caps_and_dedupes():
    ag = _bare_agent()

    def fake_search(**kwargs):
        q = kwargs.get("query_text", "")
        if "back and forth" in q:
            items = [
                {"item_id": 1, "text_humanized": "argued about benchmark results", "score": 0.91, "round_id": 4},
                {"item_id": 2, "text_humanized": "called out missing evidence", "score": 0.87, "round_id": 5},
            ]
        elif "previously stated opinion" in q:
            items = [
                {"item_id": 2, "text_humanized": "duplicate id should dedupe", "score": 0.86, "round_id": 5},
                {"item_id": 3, "text_humanized": "said nerf was unnecessary", "score": 0.80, "round_id": 3},
            ]
        elif "personal experience anecdote" in q:
            items = [
                {"item_id": 4, "text_humanized": "once tested this in ranked and lost", "score": 0.76, "round_id": 2},
            ]
        else:
            items = [
                {"item_id": 5, "text_humanized": "history with @alice is conflict heavy", "score": 0.70, "round_id": 6},
            ]
        return {"items": items}

    ag._memory_search = fake_search

    pack = ag._memory_collect_high_affect_recall(
        incoming_text="you are wrong",
        thread_context="thread",
        other_user_id=202,
        thread_root_id=303,
        round_id=9,
        target_username="alice",
    )

    assert pack["has_usable_memories"] is True
    assert len(pack["items"]) == 5
    assert pack["counts_by_bucket"]["interaction"] <= 2
    assert pack["counts_by_bucket"]["opinion"] <= 2
    assert "[RECALLED MEMORIES]" in pack["prompt_block"]


def test_memory_callback_check_marker_or_overlap():
    ag = _bare_agent()
    memories = [
        {"text": "argued about benchmark evidence quality"},
    ]

    ok_marker, _ = ag._memory_reply_references_recalled_item(
        "I remember that benchmark argument and your evidence was still weak.",
        memories,
    )
    assert ok_marker is True

    ok_overlap, _ = ag._memory_reply_references_recalled_item(
        "Your benchmark evidence quality is still weak.",
        memories,
    )
    assert ok_overlap is True

    ok_none, _ = ag._memory_reply_references_recalled_item(
        "cool story bro",
        memories,
    )
    assert ok_none is False
