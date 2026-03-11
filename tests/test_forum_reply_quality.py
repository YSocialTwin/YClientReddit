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
    ag.memory_run_id = "run-1"
    ag.style_elaborate_enabled = False
    ag.toxicity_val = 0.0
    return ag


def test_build_comment_style_options_blocks_quick_styles_below_root():
    ag = _bare_agent()

    root_styles = ag._build_comment_style_options(reply_depth=0)
    deep_styles = ag._build_comment_style_options(reply_depth=2)

    assert root_styles["quick_styles_allowed"] is True
    assert "QUICK_AFFIRM" in root_styles["style_names"]
    assert "QUICK_DISAGREE" in root_styles["style_names"]
    assert deep_styles["quick_styles_allowed"] is False
    assert "QUICK_AFFIRM" not in deep_styles["style_names"]
    assert "QUICK_DISAGREE" not in deep_styles["style_names"]


def test_filter_forum_browse_candidates_rejects_low_substance_predictions():
    ag = _bare_agent()
    posts = [
        {
            "post_id": 1,
            "comment_to": -1,
            "text": "Adin Ross buying the Breaking Bad house could work if the tours actually recreate scenes.",
        },
        {"post_id": 5, "comment_to": 1, "text": "fans will really like that"},
        {"post_id": 6, "comment_to": 1, "text": "Fans will not lose their shit, theyll be disappointed."},
        {
            "post_id": 7,
            "comment_to": 1,
            "text": "How do you figure moving the trailer up will make the story land better?",
        },
    ]
    thread_maps = ag._build_thread_analysis_maps(posts, 1)
    candidates = [
        {"post_id": 5, "node": posts[1], "depth": 1, "score": 1.0},
        {"post_id": 6, "node": posts[2], "depth": 1, "score": 1.0},
        {"post_id": 7, "node": posts[3], "depth": 1, "score": 1.0},
    ]

    filtered, stats = ag._filter_forum_browse_candidates(
        candidates,
        thread_root_id=1,
        node_by_id=thread_maps["node_by_id"],
        children_by_parent=thread_maps["children_by_parent"],
        ordered_nodes=thread_maps["ordered_nodes"],
        ordered_index=thread_maps["ordered_index"],
    )

    assert [c["post_id"] for c in filtered] == [7]
    assert stats["filtered_low_substance_candidates"] == 2
    assert stats["filtered_redundant_candidates"] == 0


def test_filter_forum_browse_candidates_marks_redundant_low_substance_branch():
    ag = _bare_agent()
    posts = [
        {
            "post_id": 1,
            "comment_to": -1,
            "text": "Adin Ross buying the Breaking Bad house could work if the tours actually recreate scenes.",
        },
        {"post_id": 5, "comment_to": 1, "text": "fans will really like that"},
        {"post_id": 6, "comment_to": 1, "text": "Fans will not lose their shit, theyll be disappointed."},
        {"post_id": 9, "comment_to": 6, "text": "fans might hate it"},
        {
            "post_id": 21,
            "comment_to": 6,
            "text": "fans could be disappointed if its just a photo op with no real setup",
        },
    ]
    thread_maps = ag._build_thread_analysis_maps(posts, 1)
    candidates = [
        {"post_id": 21, "node": posts[4], "depth": 2, "score": 1.0},
    ]

    filtered, stats = ag._filter_forum_browse_candidates(
        candidates,
        thread_root_id=1,
        node_by_id=thread_maps["node_by_id"],
        children_by_parent=thread_maps["children_by_parent"],
        ordered_nodes=thread_maps["ordered_nodes"],
        ordered_index=thread_maps["ordered_index"],
    )

    assert filtered == []
    assert stats["filtered_redundant_candidates"] == 1


def test_log_comment_style_selection_emits_quality_fields():
    ag = _bare_agent()
    events = []
    ag._decision_log = lambda payload: events.append(payload)

    ag._log_comment_style_selection(
        tid=7,
        post_id=21,
        thread_root_id=1,
        reply_depth=2,
        target_quality={
            "low_substance": True,
            "redundant_branch": True,
            "generic_crowd_reaction": True,
        },
        style_selection={
            "selected_style": "SKIP",
            "available_styles": ["QUESTION", "MEDIUM_ENGAGE"],
            "quick_styles_allowed": False,
            "forced_skip_reason": "redundant_low_substance_branch",
        },
    )

    assert len(events) == 1
    event = events[0]
    assert event["decision_type"] == "comment_style_selection"
    assert event["selected_style"] == "SKIP"
    assert event["quick_styles_allowed"] is False
    assert event["low_substance_parent"] is True
    assert event["redundant_branch"] is True
    assert event["generic_reaction_parent"] is True
    assert event["forced_skip_reason"] == "redundant_low_substance_branch"
