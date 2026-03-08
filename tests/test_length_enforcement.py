import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "external" / "YClientReddit") not in sys.path:
    sys.path.insert(0, str(ROOT / "external" / "YClientReddit"))

from y_client.classes.base_agent import Agent  # noqa: E402


def _bare_agent():
    ag = Agent.__new__(Agent)
    ag.user_id = 11
    ag.name = "tester"
    ag.reply_length_enforcement_enabled = True
    ag.comment_max_chars = 220
    ag.comment_max_sentences = 2
    ag.post_max_chars = 420
    ag.post_max_sentences = 3
    ag.reply_rewrite_max_attempts = 0
    ag.reply_trim_fallback_enabled = True
    ag.anti_repetition_enabled = True
    ag.anti_repetition_window_comments = 6
    ag.style_elaborate_enabled = False
    ag._recent_generated_comments = []
    ag._base_temperature = 0.6
    ag.writing_actions_this_round = 0
    ag._temperature_step = 0.05
    ag._temperature_cap = 1.5
    ag.llm_config = {
        "temperature": 0.6,
        "seed": 1,
        "max_tokens": 300,
        "config_list": [{"model": "test"}],
    }
    ag.prompts = {}
    ag._decision_log = lambda *args, **kwargs: None
    return ag


def test_trim_text_to_limits_enforces_chars_and_sentences():
    ag = _bare_agent()
    text = (
        "First sentence has detail. "
        "Second sentence is also here and somewhat long. "
        "Third sentence should be removed."
    )
    trimmed = ag._trim_text_to_limits(text, max_chars=60, max_sentences=2)
    assert len(trimmed) <= 60
    assert ag._sentence_count(trimmed) <= 2


def test_enforce_text_limits_comment_trims_when_over_limit():
    ag = _bare_agent()
    text = "A" * 260 + ". Another sentence that should not survive."
    out, meta = ag._enforce_text_limits(text=text, mode="comment", context_text="ctx")
    assert len(out) <= ag.comment_max_chars
    assert ag._sentence_count(out) <= ag.comment_max_sentences
    assert meta["trim_fallback_used"] is True


def test_enforce_text_limits_post_uses_post_caps():
    ag = _bare_agent()
    text = "Sentence one. Sentence two. Sentence three. Sentence four. " + ("B" * 600)
    out, meta = ag._enforce_text_limits(text=text, mode="post", context_text="ctx")
    assert len(out) <= ag.post_max_chars
    assert ag._sentence_count(out) <= ag.post_max_sentences
    assert meta["mode"] == "post"


def test_no_elaborate_style_in_random_fallback_when_disabled():
    ag = _bare_agent()
    # trigger fallback path that uses random choice from available styles
    ag.prompts = {}
    seen = set()
    for _ in range(120):
        seen.add(ag._select_comment_style("context", []))
    assert "ELABORATE" not in seen
    assert seen.issubset(
        {"QUICK_AFFIRM", "QUICK_DISAGREE", "QUESTION", "MEDIUM_ENGAGE", "PERSONAL_ANECDOTE"}
    )


def test_clean_text_preserves_period_spacing_and_hyphen():
    ag = _bare_agent()
    ag._strip_prompt_scaffold = lambda x: x
    cleaned = ag._Agent__clean_text("Hello. world and high-quality take")
    assert "Hello. world" in cleaned
    assert "high-quality" in cleaned


def test_repetition_detector_flags_reused_prefix():
    ag = _bare_agent()
    ag._record_generated_comment("I hear you but story still matters more here")
    assert ag._looks_repetitive_comment("I hear you but story still matters in this thread") is True
