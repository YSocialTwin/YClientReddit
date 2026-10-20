from __future__ import annotations

def _load_memory_package():
    try:
        from yclient_memory import build_memory_engine  # type: ignore
        from yclient_memory.config import MemoryConfig  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The agent memory integration requires the 'Agent Memory' plugin of YWeb "
            "(external/y_memory_subsystem, install it from the plugin page) or the pip "
            "package 'yclient-memory'. Install one of them before enabling agent memory."
        ) from exc
    return build_memory_engine, MemoryConfig


class YClientRedditMemoryRuntime:
    def __init__(self, agent):
        self.agent = agent

    def llm_json(self, prompt_key, variables, config=None):
        if prompt_key == "memory_high_affect_classifier":
            return self.agent._memory_llm_high_affect_classifier(
                incoming_text=variables.get("incoming_text", ""),
                thread_context=variables.get("thread_context", ""),
                target_username=variables.get("target_username", ""),
            ) or {}
        return {}

    def llm_text(self, prompt_key, variables, config=None):
        return ""

    def get_author_id_and_username(self, post_id):
        return self.agent._memory_get_author_id_and_username(int(post_id))

    def get_thread_root_id(self, post_id):
        return self.agent._memory_get_thread_root_id(int(post_id))

    def get_recent_root_posts(self, round_id, limit=24, rounds_back=18):
        return self.agent._memory_get_recent_root_posts(
            tid=int(round_id),
            limit=int(limit),
            rounds_back=int(rounds_back),
        )

    def get_post_text(self, post_id):
        getter = getattr(self.agent, f"_{self.agent.__class__.__name__}__get_post", None)
        if getter is None:
            return ""
        try:
            return getter(int(post_id))
        except Exception:
            return ""

    def persona_snapshot(self):
        return {
            "user_id": getattr(self.agent, "user_id", None),
            "name": getattr(self.agent, "name", None),
            "language": getattr(self.agent, "language", None),
            "leaning": getattr(self.agent, "leaning", None),
        }

    def decision_log(self, payload):
        if hasattr(self.agent, "_decision_log"):
            try:
                self.agent._decision_log(payload)
            except Exception:
                return None
        return None

_LANGUAGE_WARNED = False


def _warn_if_language_unsupported(MemoryConfig, language):
    """yclient-memory < 0.2.0 has no language support: it keeps rendering English."""
    global _LANGUAGE_WARNED
    if _LANGUAGE_WARNED or not language or language == "en":
        return
    if "language" not in getattr(MemoryConfig, "__dataclass_fields__", {}):
        _LANGUAGE_WARNED = True
        import logging

        logging.getLogger(__name__).warning(
            "yclient-memory is older than 0.2.0 and ignores the prompt language %r: "
            "memory blocks stay in English. Update the Agent Memory plugin.",
            language,
        )

_PACKAGE_LOGGED = False


def _log_memory_package_once(language):
    """One INFO line saying which yclient_memory is loaded (plugin checkout or pip)."""
    global _PACKAGE_LOGGED
    if _PACKAGE_LOGGED:
        return
    _PACKAGE_LOGGED = True
    try:
        import logging

        import yclient_memory

        logging.getLogger(__name__).info(
            "yclient_memory %s loaded from %s (memory language: %s)",
            getattr(yclient_memory, "__version__", "?"),
            getattr(yclient_memory, "__file__", "?"),
            language or "en",
        )
    except Exception:  # logging must never break the simulation
        pass


def build_agent_memory_engine(agent):
    from y_client import prompt_store

    build_memory_engine, MemoryConfig = _load_memory_package()
    raw = {
        # Language of the simulation prompts: the memory package renders its texts in it.
        "memory_language": prompt_store.current_language(),
        "memory_enabled": getattr(agent, "memory_enabled", True),
        "memory_backend": getattr(agent, "memory_backend", "hybrid_semantic"),
        "memory_prompt_mode": getattr(agent, "memory_prompt_mode", "legacy"),
        "memory_vote_signal_only": getattr(agent, "memory_vote_signal_only", False),
        "memory_reply_context_max_chars": getattr(agent, "memory_reply_context_max_chars", 280),
        "memory_cross_thread_callback_min_score": getattr(
            agent, "memory_cross_thread_callback_min_score", 0.80
        ),
        "memory_high_affect_enabled": getattr(agent, "memory_high_affect_enabled", True),
        "memory_high_affect_rule_threshold": getattr(agent, "memory_high_affect_rule_threshold", 0.55),
        "memory_high_affect_uncertain_low": getattr(agent, "memory_high_affect_uncertain_low", 0.35),
        "memory_high_affect_uncertain_high": getattr(agent, "memory_high_affect_uncertain_high", 0.70),
        "memory_high_affect_search_k": getattr(agent, "memory_high_affect_search_k", 12),
        "memory_high_affect_max_items": getattr(agent, "memory_high_affect_max_items", 6),
        "memory_high_affect_max_chars": getattr(agent, "memory_high_affect_max_chars", 900),
        "memory_high_affect_llm_fallback": getattr(agent, "memory_high_affect_llm_fallback", True),
        "memory_nuance_enabled": getattr(agent, "memory_nuance_enabled", True),
        "memory_nuance_min_score": getattr(agent, "memory_nuance_min_score", 0.35),
        "memory_nuance_callback_probability": getattr(agent, "memory_nuance_callback_probability", 0.55),
        "memory_nuance_cues_max_chars": getattr(agent, "memory_nuance_cues_max_chars", 900),
    }
    _warn_if_language_unsupported(MemoryConfig, raw["memory_language"])
    _log_memory_package_once(raw["memory_language"])
    config = MemoryConfig.from_mapping(raw)
    runtime = YClientRedditMemoryRuntime(agent)
    engine = build_memory_engine(
        backend=str(raw["memory_backend"] or "hybrid_semantic"),
        config=config,
        runtime=runtime,
    )
    return runtime, engine
