from y_client.recsys.ContentRecSys import ContentRecSys
from y_client.recsys.FollowRecSys import FollowRecSys
from y_client.news_feeds.client_modals import Websites, Images, Articles, session, Agent_Custom_Prompt, ImagePosts
from y_client.classes.annotator import Annotator
from y_client.logger import log_execution_time
from sqlalchemy.sql.expression import func
from sqlalchemy import text
from y_client.news_feeds.feed_reader import NewsFeed
from y_client.classes.time import SimulationSlot
import random
from requests import get, post
import json
from autogen import AssistantAgent
import numpy as np
import re
import logging
import uuid
import math
import hashlib

__all__ = ["Agent", "Agents"]

_MEMORY_RUN_ID = None
_MEMORY_RESET_DONE = False
_MEMORY_LAST_DIGEST_UPDATE_ROUND = None

DEFAULT_MAX_THREAD_CONTEXT_CHARS = 3200
MAX_MAX_THREAD_CONTEXT_CHARS = 4800
DEFAULT_OLLAMA_MAX_TOKENS = 768
DEFAULT_MEMORY_PROMPT_MAX_CHARS = 1600
MAX_MEMORY_PROMPT_MAX_CHARS = 3200
DEFAULT_MEMORY_SEARCH_MAX_CHARS = 900
MAX_MEMORY_SEARCH_MAX_CHARS = 1800
DEFAULT_MEMORY_TIER_A_MAX_CHARS = 350
MAX_MEMORY_TIER_A_MAX_CHARS = 2000
DEFAULT_MEMORY_TIER_B_MAX_CHARS = 900
DEFAULT_MEMORY_TIER_C_MAX_CHARS = 900
MAX_MEMORY_TIER_BC_MAX_CHARS = 3200
DEFAULT_MEMORY_TOTAL_MAX_CHARS = 2200
MAX_MEMORY_TOTAL_MAX_CHARS = 5000
_PROMPT_SCAFFOLD_PATTERNS = [
    re.compile(r"\bmemory tier [abc]\b", re.IGNORECASE),
    re.compile(r"\bmemory context\b", re.IGNORECASE),
    re.compile(r"\bmemory search brief\b", re.IGNORECASE),
    re.compile(r"\bmemory pack\b", re.IGNORECASE),
    re.compile(r"\bfacts pack\b", re.IGNORECASE),
    re.compile(r"\bi am the handler\b", re.IGNORECASE),
    re.compile(r"\bwrite a new caption\b", re.IGNORECASE),
    re.compile(r"\byour interests\s*\(pick one\)\b", re.IGNORECASE),
]
NO_EM_EN_DASH_PROMPT_RULE = (
    "STYLE RULE: Do not use any dash characters. This includes U+2014 em dash, "
    "U+2013 en dash, and ASCII minus sign. Use commas, periods, or parentheses instead."
)
_HIGH_AFFECT_CALLBACK_MARKERS = [
    "i remember",
    "you said",
    "last time",
    "this reminds me",
    "same thing happened",
    "again with",
    "like before",
    "as before",
]
_MEMORY_CALLBACK_STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "because",
    "before",
    "being",
    "between",
    "could",
    "every",
    "first",
    "from",
    "have",
    "just",
    "last",
    "make",
    "many",
    "much",
    "only",
    "other",
    "really",
    "same",
    "some",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "those",
    "very",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "your",
}


def _normalize_output_tokens(raw_value):
    """Normalize LLM output budget, allowing any positive integer."""
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_OLLAMA_MAX_TOKENS
    if parsed <= 0:
        return DEFAULT_OLLAMA_MAX_TOKENS
    return parsed


class Agent(object):
    def __init__(
        self,
        name: str,
        email: str,
        pwd: str = None,
        age: int = None,
        interests: list = None,
        leaning: str = None,
        ag_type="llama3",
        load: bool = False,
        recsys: ContentRecSys = None,
        frecsys: FollowRecSys = None,
        config: dict = None,
        big_five: dict = None,
        language: str = None,
        owner: str = None,
        education_level: str = None,
        joined_on: int = None,
        round_actions: int = 3,
        gender: str = None,
        nationality: str = None,
        profession: str = None,
        toxicity: str = "no",
        api_key: str = "NULL",
        is_page: int = 0,
        daily_activity_level: int = 1,
        activity_profile: str = "Always On",
        *args,
        **kwargs,
    ):
        """
        Initialize the Agent object.

        :param name: the name of the agent
        :param email: the email of the agent
        :param pwd: the password of the agent
        :param age: the age of the agent
        :param interests: the interests of the agent
        :param leaning: the leaning of the agent
        :param ag_type: the type of the agent
        :param load: whether to load the agent from file or not
        :param recsys: the content recommendation system
        :param frecsys: the follow recommendation system
        :param config: the configuration dictionary
        :param big_five: the big five personality traits
        :param language: the language of the agent
        :param owner: the owner of the agent
        :param education_level: the education level of the agent
        :param joined_on: the joined on date of the agent
        :param round_actions: the number of daily actions
        :param gender: the agent gender
        :param nationality: the agent nationality
        :param toxicity: the toxicity level of the agent, default is "no"
        :param api_key: the LLM server api key, default is NULL (self-hosted)
        """

        if "web" in kwargs:

            self.__web_init(name=name, email=email, pwd=pwd, interests=interests, leaning=leaning,
                            ag_type=ag_type, load=load, recsys=recsys, age=age,
                            frecsys=frecsys, config=config, big_five=big_five, language=language, owner=owner, education_level=education_level,
                            joined_on=joined_on, round_actions=round_actions, gender=gender, nationality=nationality,
                            profession=profession, toxicity=toxicity,
                            api_key=api_key, is_page=is_page, daily_activity_level=daily_activity_level,
                            activity_profile=activity_profile, *args, **kwargs)
        else:
            self.emotions = config["posts"]["emotions"]
            self.actions_likelihood = config["simulation"]["actions_likelihood"]
            self.base_url = config["servers"]["api"]
            self.llm_base = config["servers"]["llm"]
            self.content_rec_sys_name = None
            self.follow_rec_sys_name = None
            self.name = name
            self.email = email
            self.attention_window = int(config["agents"]["attention_window"])
            # Lightweight per-experiment "subreddit vibe" injected by YSocial from
            # Exps.exp_descr. Used in prompts to keep agents in the right tone/topic
            # without repeating long instructions everywhere.
            self.subreddit_vibe = (
                (config.get("simulation", {}).get("subreddit_vibe") or "").strip()
            )
            if not self.subreddit_vibe:
                self.subreddit_vibe = (
                    "casual fun subreddit about pop culture/gaming; short replies; "
                    "avoid real-world politics/history detours"
                )
            # Hard cap thread context to keep prompts Ollama-friendly.
            self.max_thread_context_chars = int(
                config.get("agents", {}).get(
                    "max_thread_context_chars", DEFAULT_MAX_THREAD_CONTEXT_CHARS
                )
            )
            self.llm_v_config = {
                "url": config["servers"]["llm_v"],
                "api_key": config["servers"]["llm_v_api_key"] if (config["servers"]["llm_v_api_key"] is not None and config["servers"]["llm_v_api_key"] != "") else "NULL",
                "model": config["agents"]["llm_v_agent"],
                "temperature": config["servers"]["llm_v_temperature"],
                "max_tokens": config["servers"]["llm_v_max_tokens"]
            }
            self.is_page = is_page
            # Keep constructor-provided defaults available for resume paths.
            self.daily_activity_level = daily_activity_level
            self.activity_profile = activity_profile
            self.profession = profession

            if not load:
                self.language = language
                self.type = ag_type
                self.age = age
                self.interests = interests
                self.leaning = leaning
                self.pwd = pwd
                self.oe = big_five["oe"]
                self.co = big_five["co"]
                self.ex = big_five["ex"]
                self.ag = big_five["ag"]
                self.ne = big_five["ne"]
                self.owner = owner
                self.education_level = education_level
                self.joined_on = joined_on
                sc = SimulationSlot(config)
                sc.get_current_slot()
                self.joined_on = sc.id
                self.round_actions = round_actions
                self.gender = gender
                self.nationality = nationality
                self.profession = profession
                self.toxicity = toxicity
                self.daily_activity_level = daily_activity_level
                self.replying_to = ""  # Will be set when commenting
                self.activity_profile = activity_profile

                uid = self.__register()
                if uid is None:
                    pass
                else:
                    self.user_id = uid

            else:
                us = json.loads(self.__get_user())
                self.user_id = us["id"]
                self.type = us["user_type"]
                self.age = us["age"]

                if us["is_page"] == 0:
                    self.interests = random.randint(config["agents"]["n_interests"]["min"],
                                                    config["agents"]["n_interests"]["max"])
                    self.interests = self.__get_interests(-1)[0]
                else:
                    self.interests = []

                self.leaning = us["leaning"]
                self.pwd = us["password"]
                self.oe = us["oe"]
                self.co = us["co"]
                self.ex = us["ex"]
                self.ag = us["ag"]
                self.ne = us["ne"]
                self.content_rec_sys_name = us["rec_sys"]
                self.follow_rec_sys_name = us["frec_sys"]
                self.language = us["language"]
                self.owner = us["owner"]
                self.education_level = us["education_level"]
                self.round_actions = us["round_actions"]
                self.joined_on = us["joined_on"]
                self.gender = us["gender"]
                self.toxicity = us["toxicity"]
                self.nationality = us["nationality"]
                self.is_page = us["is_page"]
                # Preserve profile values provided by the population file on resume.
                # YServer /get_user may not include activity_profile in some deployments.
                self.daily_activity_level = us.get(
                    "daily_activity_level", self.daily_activity_level
                )
                self.activity_profile = us.get(
                    "activity_profile", self.activity_profile or "Always On"
                )
                self.profession = us.get("profession", "")

            config_list = {
                "model": f"{self.type}",
                "base_url": self.llm_base,
                "timeout": 10000,
                "api_type": "open_ai",
                "api_key": api_key if (api_key is not None and api_key != "") else "NULL",
                "price": [0, 0],
            }

            self.llm_config = {
                "cache_seed": None,  # Disable AutoGen's caching
                "config_list": [config_list],
                "seed": np.random.randint(0, 100000),
                "max_tokens": _normalize_output_tokens(
                    config.get("servers", {}).get("llm_max_tokens")
                ),
                # max response length, -1 no limits. Imposing limits may lead to truncated responses
                "temperature": config['servers']['llm_temperature'],
            }

            # add and configure the content recsys
            self.content_rec_sys = recsys
            if self.content_rec_sys is not None:
                self.content_rec_sys.add_user_id(self.user_id)

            # add and configure the follow recsys
            self.follow_rec_sys = frecsys
            if self.follow_rec_sys is not None:
                self.follow_rec_sys.add_user_id(self.user_id)

            self.prompts = None

            # Track posts per round to prevent duplicates
            self.posts_this_round = 0
            self._base_temperature = config['servers']['llm_temperature']
            self.writing_actions_this_round = 0
            self._temperature_step = 0.05
            self._temperature_cap = 1.5
            # Track posted article links to prevent duplicates
            self._posted_links = set()
            # Track recent post texts for content deduplication
            self._recent_posts = []
            # Track per-round per-parent comment text to avoid exact duplicate replies.
            self._recent_comments_by_round_parent = {}
            # Track recent generated comments to reduce boilerplate repetition.
            self._recent_generated_comments = []

            # Mention-handling guardrails (reply/vote/ignore on notifications).
            self.max_replies_per_round = int(config["agents"].get("max_replies_per_round", 2))
            self.reply_cooldown_rounds = int(config["agents"].get("reply_cooldown_rounds", 2))
            self.max_reply_chain_depth = int(config["agents"].get("max_reply_chain_depth", 5))
            self.max_absolute_reply_depth = int(config["agents"].get("max_absolute_reply_depth", 5))
            self.max_comments_per_thread = int(config["agents"].get("max_comments_per_thread", 0))
            self.mention_decision_mode = config["agents"].get("mention_decision_mode", "llm")

            self.replies_this_round = 0
            self.last_replied_to = {}  # other_user_id -> last_round_replied
            self.reply_chain_depths = {}  # (thread_root_id, other_user_id) -> replies so far
            self._init_thread_browse_config(config)
            self._init_memory_config(config)
            self._init_decision_logging_config(config)

    def __web_init(self, name: str,
        email: str,
        pwd: str = None,
        age: int = None,
        interests: list = None,
        leaning: str = None,
        ag_type="llama3",
        load: bool = False,
        recsys: ContentRecSys = None,
        frecsys: FollowRecSys = None,
        config: dict = None,
        big_five: dict = None,
        language: str = None,
        owner: str = None,
        education_level: str = None,
        joined_on: int = None,
        round_actions: int = 3,
        gender: str = None,
        nationality: str = None,
        profession: str = None,
        toxicity: str = "no",
        api_key: str = "NULL",
        is_page: int = 0,
        daily_activity_level: int = 1,
        activity_profile: str = "Always On",
        *args,
        **kwargs,):

        self.emotions = config["posts"]["emotions"]
        self.actions_likelihood = config["simulation"]["actions_likelihood"]
        self.base_url = config["servers"]["api"]
        self.llm_base = config["servers"]["llm"]
        self.content_rec_sys_name = None
        self.follow_rec_sys_name = None
        self.content_rec_sys = None
        self.follow_rec_sys = None

        self.name = name
        self.email = email
        self.attention_window = int(config["agents"]["attention_window"])
        self.subreddit_vibe = (
            (config.get("simulation", {}).get("subreddit_vibe") or "").strip()
        )
        if not self.subreddit_vibe:
            self.subreddit_vibe = (
                "casual fun subreddit about pop culture/gaming; short replies; "
                "avoid real-world politics/history detours"
            )
        self.max_thread_context_chars = int(
            config.get("agents", {}).get(
                "max_thread_context_chars", DEFAULT_MAX_THREAD_CONTEXT_CHARS
            )
        )

        if "prompts" in kwargs:
            self.prompts = kwargs["prompts"]
            # save on agent custom prompt
            if self.prompts is not None:
                aprompt = Agent_Custom_Prompt(name=self.name, prompt=self.prompts)
                session.add(aprompt)
                session.commit()

        self.llm_v_config = {
            "url": config["servers"]["llm_v"],
            "api_key": config["servers"]["llm_v_api_key"] if (config["servers"]["llm_v_api_key"] is not None and config["servers"]["llm_v_api_key"] != "") else "NULL",
            "temperature": config["servers"]["llm_v_temperature"],
            "max_tokens": int(config["servers"]["llm_v_max_tokens"])
        }
        try:
            self.llm_v_config["model"] = config["servers"]["llm_v_agent"]
        except:
            self.llm_v_config["model"] = 'minicpm-v'

        self.is_page = is_page
        # Keep constructor-provided defaults available for resume paths.
        self.daily_activity_level = daily_activity_level
        self.activity_profile = activity_profile
        self.profession = profession

        if not load:
            self.language = language
            self.type = ag_type
            self.age = age
            self.interests = interests
            self.leaning = leaning
            self.pwd = pwd
            try:
                self.oe = big_five["oe"]
                self.co = big_five["co"]
                self.ex = big_five["ex"]
                self.ag = big_five["ag"]
                self.ne = big_five["ne"]

            except:
                self.oe = kwargs["oe"]
                self.co = kwargs["co"]
                self.ex = kwargs["ex"]
                self.ag = kwargs["ag"]
                self.ne = kwargs["ne"]

            self.toxicity = toxicity
            self.owner = owner
            self.education_level = education_level
            self.joined_on = joined_on
            sc = SimulationSlot(config)
            sc.get_current_slot()
            self.joined_on = sc.id
            self.round_actions = round_actions
            self.gender = gender
            self.nationality = nationality
            self.profession = profession
            self.daily_activity_level = daily_activity_level
            self.replying_to = ""  # Will be set when commenting
            self.activity_profile = activity_profile

            uid = self.__register()
            if uid is None:
                pass
            else:
                self.user_id = uid

        else:
            us = json.loads(self.__get_user())
            self.user_id = us["id"]
            self.type = us["user_type"]
            self.age = us["age"]

            if us["is_page"] == 0:
                try:
                    self.interests = random.randint(config["agents"]["n_interests"]["min"],
                                                    config["agents"]["n_interests"]["max"])
                    self.interests = self.__get_interests(-1)[0]
                except:
                    self.interests = interests
                    self.interests = self.__get_interests(-1)[0]
            else:
                self.interests = []

            self.leaning = us["leaning"]
            self.pwd = us["password"]
            self.oe = us["oe"]
            self.co = us["co"]
            self.ex = us["ex"]
            self.ag = us["ag"]
            self.ne = us["ne"]
            self.content_rec_sys_name = us["rec_sys"]
            self.follow_rec_sys_name = us["frec_sys"]
            self.language = us["language"]
            self.owner = us["owner"]
            self.education_level = us["education_level"]
            self.round_actions = us["round_actions"]
            self.joined_on = us["joined_on"]
            self.gender = us["gender"]
            self.toxicity = us["toxicity"]
            self.nationality = us["nationality"]
            self.is_page = us["is_page"]
            # Preserve profile values provided by the population file on resume.
            # YServer /get_user may not include activity_profile in some deployments.
            self.daily_activity_level = us.get(
                "daily_activity_level", self.daily_activity_level
            )
            self.activity_profile = us.get(
                "activity_profile", self.activity_profile or "Always On"
            )
            self.profession = us.get("profession", "")

        config_list = {
            "model": f"{self.type}",
            "base_url": self.llm_base,
            "timeout": 10000,
            "api_type": "open_ai",
            "api_key": api_key if (api_key is not None and api_key != "") else "NULL",
            "price": [0, 0],
        }

        self.llm_config = {
            "cache_seed": None,  # Disable AutoGen's caching
            "config_list": [config_list],
            "seed": np.random.randint(0, 100000),
            "max_tokens": _normalize_output_tokens(
                config.get("servers", {}).get("llm_max_tokens")
            ),
            # max response length, -1 no limits. Imposing limits may lead to truncated responses
            "temperature": float(config['servers']['llm_temperature']),
        }

        self.set_rec_sys(recsys, frecsys)

        # add and configure the content recsys
        if self.content_rec_sys is not None:
            self.content_rec_sys.add_user_id(self.user_id)

        # add and configure the follow recsys
        if self.follow_rec_sys is not None:
            self.follow_rec_sys.add_user_id(self.user_id)

        self.prompts = None

        # Track posts per round to prevent duplicates
        self.posts_this_round = 0
        self._base_temperature = float(config['servers']['llm_temperature'])
        self.writing_actions_this_round = 0
        self._temperature_step = 0.05
        self._temperature_cap = 1.5
        # Track recent post texts for content deduplication
        self._recent_posts = []
        # Track per-round per-parent comment text to avoid exact duplicate replies.
        self._recent_comments_by_round_parent = {}
        # Track recent generated comments to reduce boilerplate repetition.
        self._recent_generated_comments = []

        # Mention-handling guardrails (reply/vote/ignore on notifications).
        self.max_replies_per_round = int(config["agents"].get("max_replies_per_round", 2))
        self.reply_cooldown_rounds = int(config["agents"].get("reply_cooldown_rounds", 2))
        self.max_reply_chain_depth = int(config["agents"].get("max_reply_chain_depth", 5))
        self.max_absolute_reply_depth = int(config["agents"].get("max_absolute_reply_depth", 5))
        self.max_comments_per_thread = int(config["agents"].get("max_comments_per_thread", 0))
        self.mention_decision_mode = config["agents"].get("mention_decision_mode", "llm")

        self.replies_this_round = 0
        self.last_replied_to = {}  # other_user_id -> last_round_replied
        self.reply_chain_depths = {}  # (thread_root_id, other_user_id) -> replies so far
        self._init_thread_browse_config(config)
        self._init_memory_config(config)
        self._init_decision_logging_config(config)

    def _init_thread_browse_config(self, config: dict):
        """
        Initialize per-agent thread browsing configuration used when selecting a comment target
        inside an existing thread (human-like sequential reading).
        """
        agents_cfg = {}
        if isinstance(config, dict):
            agents_cfg = config.get("agents", {}) or {}

        def _to_int(val, default):
            try:
                return int(val)
            except Exception:
                return default

        def _to_str(val, default):
            try:
                s = str(val).strip()
                return s if s else default
            except Exception:
                return default

        self.thread_browse_mode = _to_str(agents_cfg.get("thread_browse_mode"), "llm")
        self.thread_browse_order = _to_str(agents_cfg.get("thread_browse_order"), "tree_dfs")

        self.thread_browse_max_nodes = _to_int(agents_cfg.get("thread_browse_max_nodes"), 400)
        self.thread_browse_chunk_size = _to_int(agents_cfg.get("thread_browse_chunk_size"), 20)
        self.thread_browse_top_k = _to_int(agents_cfg.get("thread_browse_top_k"), 6)
        self.thread_browse_max_llm_steps = _to_int(agents_cfg.get("thread_browse_max_llm_steps"), 3)
        self.thread_browse_snippet_chars = _to_int(agents_cfg.get("thread_browse_snippet_chars"), 220)
        self.thread_browse_context_window = _to_int(agents_cfg.get("thread_browse_context_window"), 30)

        # Safety bounds to prevent runaway prompt sizes / server load.
        if self.thread_browse_max_nodes <= 0:
            self.thread_browse_max_nodes = 400
        if self.thread_browse_max_nodes > 2000:
            self.thread_browse_max_nodes = 2000

        if self.thread_browse_chunk_size <= 0:
            self.thread_browse_chunk_size = 20
        if self.thread_browse_chunk_size > 200:
            self.thread_browse_chunk_size = 200

        if self.thread_browse_top_k <= 0:
            self.thread_browse_top_k = 6
        if self.thread_browse_top_k > 20:
            self.thread_browse_top_k = 20

        if self.thread_browse_max_llm_steps <= 0:
            self.thread_browse_max_llm_steps = 1
        if self.thread_browse_max_llm_steps > 10:
            self.thread_browse_max_llm_steps = 10

        if self.thread_browse_snippet_chars <= 0:
            self.thread_browse_snippet_chars = 220
        if self.thread_browse_snippet_chars > 800:
            self.thread_browse_snippet_chars = 800

        if self.thread_browse_context_window <= 0:
            self.thread_browse_context_window = 30
        if self.thread_browse_context_window > 200:
            self.thread_browse_context_window = 200

    # ------------------------------------------------------------------
    # Run-scoped agent memory (hybrid storage, LLM-on-write + decay).
    #
    # Design goals:
    # - No persistence across separate runs: use a random run_id per process.
    # - Hybrid storage: server DB is the shared source of truth; each agent also keeps
    #   a small in-process cache for prompt injection and update cadence tracking.
    # - LLM-on-write: update memory only after the agent writes (comment/post/vote).
    # - Forgetting: numeric decay + light text corruption + periodic resummarization.
    # ------------------------------------------------------------------

    def _init_memory_config(self, config: dict):
        agents_cfg = {}
        if isinstance(config, dict):
            agents_cfg = config.get("agents", {}) or {}

        def _to_int(val, default):
            try:
                return int(val)
            except Exception:
                return default

        def _to_float(val, default):
            try:
                return float(val)
            except Exception:
                return default

        def _to_bool(val, default):
            if val is None:
                return default
            if isinstance(val, bool):
                return val
            return str(val).strip().lower() in {"1", "true", "on", "yes"}

        def _normalize_run_id(raw_value):
            raw_run_id = str(raw_value).strip() if raw_value else uuid.uuid4().hex
            if len(raw_run_id) > 64:
                digest = hashlib.sha1(raw_run_id.encode("utf-8")).hexdigest()[:24]
                raw_run_id = f"run_{digest}"
            return raw_run_id

        self.memory_enabled = _to_bool(agents_cfg.get("memory_enabled"), True)

        self.memory_pair_limit = _to_int(agents_cfg.get("memory_pair_limit"), 5)
        self.memory_prompt_max_chars = _to_int(
            agents_cfg.get("memory_prompt_max_chars"), DEFAULT_MEMORY_PROMPT_MAX_CHARS
        )

        # Decay (per round): value *= exp(-lambda * delta_rounds)
        self.memory_social_decay_lambda = _to_float(agents_cfg.get("memory_social_decay_lambda"), 0.05)
        self.memory_thread_decay_lambda = _to_float(agents_cfg.get("memory_thread_decay_lambda"), 0.03)

        # Forgetting + resummarization cadence
        self.memory_social_corruption_rate = _to_float(agents_cfg.get("memory_social_corruption_rate"), 0.02)
        self.memory_thread_corruption_rate = _to_float(agents_cfg.get("memory_thread_corruption_rate"), 0.01)
        self.memory_social_resummarize_every_events = _to_int(
            agents_cfg.get("memory_social_resummarize_every_events"), 4
        )
        self.memory_thread_resummarize_every_events = _to_int(
            agents_cfg.get("memory_thread_resummarize_every_events"), 4
        )
        self.memory_evidence_tail_max = _to_int(agents_cfg.get("memory_evidence_tail_max"), 8)

        # Shared digest update cadence (one agent per process updates periodically)
        self.memory_digest_update_cadence_rounds = _to_int(
            agents_cfg.get("memory_digest_update_cadence_rounds"), 3
        )
        self.memory_digest_events_limit = _to_int(agents_cfg.get("memory_digest_events_limit"), 80)

        # Semantic retrieval + prompt budgeting (tiered context injection).
        self.memory_semantic_enabled = _to_bool(agents_cfg.get("memory_semantic_enabled"), True)
        self.memory_search_k = _to_int(agents_cfg.get("memory_search_k"), 8)
        self.memory_search_max_chars = _to_int(
            agents_cfg.get("memory_search_max_chars"), DEFAULT_MEMORY_SEARCH_MAX_CHARS
        )
        self.memory_search_time_window_rounds = _to_int(
            agents_cfg.get("memory_search_time_window_rounds"), 40
        )
        self.memory_tier_a_max_chars = _to_int(
            agents_cfg.get("memory_tier_a_max_chars"), DEFAULT_MEMORY_TIER_A_MAX_CHARS
        )
        self.memory_tier_b_max_chars = _to_int(
            agents_cfg.get("memory_tier_b_max_chars"), DEFAULT_MEMORY_TIER_B_MAX_CHARS
        )
        self.memory_tier_c_max_chars = _to_int(
            agents_cfg.get("memory_tier_c_max_chars"), DEFAULT_MEMORY_TIER_C_MAX_CHARS
        )
        self.memory_total_max_chars = _to_int(
            agents_cfg.get("memory_total_max_chars"), DEFAULT_MEMORY_TOTAL_MAX_CHARS
        )
        self.memory_tier_c_uncertainty_threshold = _to_float(
            agents_cfg.get("memory_tier_c_uncertainty_threshold"), 0.45
        )

        # Reflection/consolidation cadence.
        self.memory_reflection_cadence_rounds = _to_int(
            agents_cfg.get("memory_reflection_cadence_rounds"), 3
        )
        self.memory_reflection_min_events = _to_int(
            agents_cfg.get("memory_reflection_min_events"), 12
        )
        self.memory_reflection_trigger_importance_sum = _to_float(
            agents_cfg.get("memory_reflection_trigger_importance_sum"), 3.5
        )
        self.memory_reflection_max_items_per_run = _to_int(
            agents_cfg.get("memory_reflection_max_items_per_run"), 60
        )
        self.memory_embedding_model = str(
            agents_cfg.get("memory_embedding_model") or "BAAI/bge-m3"
        ).strip()
        self.memory_embedding_async = _to_bool(agents_cfg.get("memory_embedding_async"), True)
        self.memory_importance_mode = str(
            agents_cfg.get("memory_importance_mode") or "heuristic_then_batch_llm"
        ).strip()
        self.memory_nuance_enabled = _to_bool(agents_cfg.get("memory_nuance_enabled"), True)
        self.memory_nuance_callback_probability = _to_float(
            agents_cfg.get("memory_nuance_callback_probability"), 0.55
        )
        self.memory_nuance_min_score = _to_float(
            agents_cfg.get("memory_nuance_min_score"), 0.35
        )
        self.memory_nuance_cues_max_chars = _to_int(
            agents_cfg.get("memory_nuance_cues_max_chars"), 900
        )
        self.memory_nuance_planner_enabled = _to_bool(
            agents_cfg.get("memory_nuance_planner_enabled"), True
        )
        self.memory_nuance_planner_max_tokens = _to_int(
            agents_cfg.get("memory_nuance_planner_max_tokens"), 120
        )
        self.memory_nuance_planner_temperature = _to_float(
            agents_cfg.get("memory_nuance_planner_temperature"), 0.25
        )
        self.memory_relationship_priority_enabled = _to_bool(
            agents_cfg.get("memory_relationship_priority_enabled"), True
        )
        self.memory_relationship_priority_mode = str(
            agents_cfg.get("memory_relationship_priority_mode") or "soft_bias"
        ).strip().lower()
        self.memory_relationship_priority_window_rounds = _to_int(
            agents_cfg.get("memory_relationship_priority_window_rounds"), 120
        )
        self.memory_relationship_priority_max_targets = _to_int(
            agents_cfg.get("memory_relationship_priority_max_targets"), 3
        )
        self.memory_trust_gate_threshold = _to_float(
            agents_cfg.get("memory_trust_gate_threshold"), 0.15
        )
        self.memory_alignment_gate_threshold = _to_float(
            agents_cfg.get("memory_alignment_gate_threshold"), 0.0
        )
        self.memory_option_shuffle_temperature = _to_float(
            agents_cfg.get("memory_option_shuffle_temperature"), 0.35
        )
        self.memory_high_affect_enabled = _to_bool(
            agents_cfg.get("memory_high_affect_enabled"), True
        )
        self.memory_high_affect_rule_threshold = _to_float(
            agents_cfg.get("memory_high_affect_rule_threshold"), 0.55
        )
        self.memory_high_affect_uncertain_low = _to_float(
            agents_cfg.get("memory_high_affect_uncertain_low"), 0.35
        )
        self.memory_high_affect_uncertain_high = _to_float(
            agents_cfg.get("memory_high_affect_uncertain_high"), 0.70
        )
        self.memory_high_affect_search_k = _to_int(
            agents_cfg.get("memory_high_affect_search_k"), 12
        )
        self.memory_high_affect_max_items = _to_int(
            agents_cfg.get("memory_high_affect_max_items"), 6
        )
        self.memory_high_affect_max_chars = _to_int(
            agents_cfg.get("memory_high_affect_max_chars"), 900
        )
        self.memory_high_affect_callback_retry_count = _to_int(
            agents_cfg.get("memory_high_affect_callback_retry_count"), 1
        )
        self.memory_high_affect_llm_fallback = _to_bool(
            agents_cfg.get("memory_high_affect_llm_fallback"), True
        )
        self.memory_prompt_mode = str(
            agents_cfg.get("memory_prompt_mode") or "legacy"
        ).strip().lower()
        self.memory_reply_context_max_chars = _to_int(
            agents_cfg.get("memory_reply_context_max_chars"), 280
        )
        self.memory_vote_signal_only = _to_bool(
            agents_cfg.get("memory_vote_signal_only"), False
        )
        self.forum_post_structure_strict = _to_bool(
            agents_cfg.get("forum_post_structure_strict"), False
        )
        self.memory_cross_thread_callback_min_score = _to_float(
            agents_cfg.get("memory_cross_thread_callback_min_score"), 0.80
        )
        self.reply_length_enforcement_enabled = _to_bool(
            agents_cfg.get("reply_length_enforcement_enabled"), True
        )
        self.comment_max_chars = _to_int(agents_cfg.get("comment_max_chars"), 220)
        self.comment_max_sentences = _to_int(agents_cfg.get("comment_max_sentences"), 2)
        self.post_max_chars = _to_int(agents_cfg.get("post_max_chars"), 420)
        self.post_max_sentences = _to_int(agents_cfg.get("post_max_sentences"), 3)
        self.reply_rewrite_max_attempts = _to_int(
            agents_cfg.get("reply_rewrite_max_attempts"), 1
        )
        self.reply_trim_fallback_enabled = _to_bool(
            agents_cfg.get("reply_trim_fallback_enabled"), True
        )
        self.style_elaborate_enabled = _to_bool(
            agents_cfg.get("style_elaborate_enabled"), False
        )
        self.anti_repetition_enabled = _to_bool(
            agents_cfg.get("anti_repetition_enabled"), True
        )
        self.anti_repetition_window_comments = _to_int(
            agents_cfg.get("anti_repetition_window_comments"), 6
        )

        # --- Proactive high-affect initiation engine ---
        self.proactive_affect_enabled = _to_bool(
            agents_cfg.get("proactive_affect_enabled"), True
        )
        self.proactive_affect_cap_per_round = _to_int(
            agents_cfg.get("proactive_affect_cap_per_round"), 2
        )
        self.proactive_affect_probability_scale = _to_float(
            agents_cfg.get("proactive_affect_probability_scale"), 0.7
        )
        self.proactive_affect_contrarian_threshold = _to_float(
            agents_cfg.get("proactive_affect_contrarian_threshold"), 0.75
        )
        self._proactive_affect_this_round = 0
        self._persona_affect_profile_cache = None

        # --- Memory cold-start ---
        self.memory_cold_start_window = _to_int(
            agents_cfg.get("memory_cold_start_window"), 5
        )
        self._memory_global_interaction_index = 0
        self._memory_cold_start_decay_level = 0

        if self.memory_search_k <= 0:
            self.memory_search_k = 8
        if self.memory_search_k > 40:
            self.memory_search_k = 40
        if self.max_thread_context_chars < 200:
            self.max_thread_context_chars = DEFAULT_MAX_THREAD_CONTEXT_CHARS
        if self.max_thread_context_chars > MAX_MAX_THREAD_CONTEXT_CHARS:
            self.max_thread_context_chars = MAX_MAX_THREAD_CONTEXT_CHARS
        if self.memory_prompt_max_chars < 200:
            self.memory_prompt_max_chars = DEFAULT_MEMORY_PROMPT_MAX_CHARS
        if self.memory_prompt_max_chars > MAX_MEMORY_PROMPT_MAX_CHARS:
            self.memory_prompt_max_chars = MAX_MEMORY_PROMPT_MAX_CHARS
        if self.memory_search_max_chars < 300:
            self.memory_search_max_chars = DEFAULT_MEMORY_SEARCH_MAX_CHARS
        if self.memory_search_max_chars > MAX_MEMORY_SEARCH_MAX_CHARS:
            self.memory_search_max_chars = MAX_MEMORY_SEARCH_MAX_CHARS
        if self.memory_search_time_window_rounds < 0:
            self.memory_search_time_window_rounds = 40
        if self.memory_tier_a_max_chars < 100:
            self.memory_tier_a_max_chars = DEFAULT_MEMORY_TIER_A_MAX_CHARS
        if self.memory_tier_a_max_chars > MAX_MEMORY_TIER_A_MAX_CHARS:
            self.memory_tier_a_max_chars = MAX_MEMORY_TIER_A_MAX_CHARS
        if self.memory_tier_b_max_chars < 400:
            self.memory_tier_b_max_chars = DEFAULT_MEMORY_TIER_B_MAX_CHARS
        if self.memory_tier_b_max_chars > MAX_MEMORY_TIER_BC_MAX_CHARS:
            self.memory_tier_b_max_chars = MAX_MEMORY_TIER_BC_MAX_CHARS
        if self.memory_tier_c_max_chars < 400:
            self.memory_tier_c_max_chars = DEFAULT_MEMORY_TIER_C_MAX_CHARS
        if self.memory_tier_c_max_chars > MAX_MEMORY_TIER_BC_MAX_CHARS:
            self.memory_tier_c_max_chars = MAX_MEMORY_TIER_BC_MAX_CHARS
        if self.memory_total_max_chars < 800:
            self.memory_total_max_chars = (
                self.memory_tier_a_max_chars
                + self.memory_tier_b_max_chars
                + self.memory_tier_c_max_chars
            )
        if self.memory_total_max_chars > MAX_MEMORY_TOTAL_MAX_CHARS:
            self.memory_total_max_chars = MAX_MEMORY_TOTAL_MAX_CHARS
        tiers_total = (
            self.memory_tier_a_max_chars
            + self.memory_tier_b_max_chars
            + self.memory_tier_c_max_chars
        )
        if tiers_total > self.memory_total_max_chars:
            # Keep tier priority A -> B -> C while forcing tier sum to stay within total cap.
            remaining = max(0, int(self.memory_total_max_chars))
            self.memory_tier_a_max_chars = min(self.memory_tier_a_max_chars, remaining)
            remaining -= self.memory_tier_a_max_chars
            self.memory_tier_b_max_chars = min(self.memory_tier_b_max_chars, remaining)
            remaining -= self.memory_tier_b_max_chars
            self.memory_tier_c_max_chars = min(self.memory_tier_c_max_chars, remaining)
        if self.memory_tier_c_uncertainty_threshold < 0:
            self.memory_tier_c_uncertainty_threshold = 0.45
        if self.memory_tier_c_uncertainty_threshold > 1:
            self.memory_tier_c_uncertainty_threshold = 1.0
        if self.memory_reflection_cadence_rounds < 1:
            self.memory_reflection_cadence_rounds = 3
        if self.memory_reflection_min_events < 1:
            self.memory_reflection_min_events = 12
        if self.memory_reflection_max_items_per_run < 1:
            self.memory_reflection_max_items_per_run = 60
        if self.memory_nuance_callback_probability < 0:
            self.memory_nuance_callback_probability = 0.0
        if self.memory_nuance_callback_probability > 1:
            self.memory_nuance_callback_probability = 1.0
        if self.memory_nuance_min_score < 0:
            self.memory_nuance_min_score = 0.0
        if self.memory_nuance_min_score > 1:
            self.memory_nuance_min_score = 1.0
        if self.memory_nuance_cues_max_chars < 180:
            self.memory_nuance_cues_max_chars = 900
        if self.memory_nuance_cues_max_chars > 3000:
            self.memory_nuance_cues_max_chars = 3000
        if self.memory_nuance_planner_max_tokens < 32:
            self.memory_nuance_planner_max_tokens = 120
        if self.memory_nuance_planner_max_tokens > 512:
            self.memory_nuance_planner_max_tokens = 512
        if self.memory_nuance_planner_temperature < 0:
            self.memory_nuance_planner_temperature = 0.0
        if self.memory_nuance_planner_temperature > 1:
            self.memory_nuance_planner_temperature = 1.0
        if self.memory_relationship_priority_mode not in {"soft_bias"}:
            self.memory_relationship_priority_mode = "soft_bias"
        if self.memory_relationship_priority_window_rounds < 1:
            self.memory_relationship_priority_window_rounds = 120
        if self.memory_relationship_priority_window_rounds > 2000:
            self.memory_relationship_priority_window_rounds = 2000
        if self.memory_relationship_priority_max_targets < 1:
            self.memory_relationship_priority_max_targets = 3
        if self.memory_relationship_priority_max_targets > 10:
            self.memory_relationship_priority_max_targets = 10
        if self.memory_trust_gate_threshold < -1:
            self.memory_trust_gate_threshold = -1.0
        if self.memory_trust_gate_threshold > 1:
            self.memory_trust_gate_threshold = 1.0
        if self.memory_alignment_gate_threshold < -1:
            self.memory_alignment_gate_threshold = -1.0
        if self.memory_alignment_gate_threshold > 1:
            self.memory_alignment_gate_threshold = 1.0
        if self.memory_option_shuffle_temperature < 0:
            self.memory_option_shuffle_temperature = 0.0
        if self.memory_option_shuffle_temperature > 1:
            self.memory_option_shuffle_temperature = 1.0
        if self.memory_high_affect_rule_threshold < 0:
            self.memory_high_affect_rule_threshold = 0.55
        if self.memory_high_affect_rule_threshold > 1:
            self.memory_high_affect_rule_threshold = 1.0
        if self.memory_high_affect_uncertain_low < 0:
            self.memory_high_affect_uncertain_low = 0.35
        if self.memory_high_affect_uncertain_low > 1:
            self.memory_high_affect_uncertain_low = 1.0
        if self.memory_high_affect_uncertain_high < 0:
            self.memory_high_affect_uncertain_high = 0.70
        if self.memory_high_affect_uncertain_high > 1:
            self.memory_high_affect_uncertain_high = 1.0
        if self.memory_high_affect_uncertain_high < self.memory_high_affect_uncertain_low:
            self.memory_high_affect_uncertain_high = self.memory_high_affect_uncertain_low
        if self.memory_high_affect_search_k < 1:
            self.memory_high_affect_search_k = 12
        if self.memory_high_affect_search_k > 40:
            self.memory_high_affect_search_k = 40
        if self.memory_high_affect_max_items < 1:
            self.memory_high_affect_max_items = 6
        if self.memory_high_affect_max_items > 16:
            self.memory_high_affect_max_items = 16
        if self.memory_high_affect_max_chars < 180:
            self.memory_high_affect_max_chars = 900
        if self.memory_high_affect_max_chars > 3000:
            self.memory_high_affect_max_chars = 3000
        if self.memory_high_affect_callback_retry_count < 0:
            self.memory_high_affect_callback_retry_count = 0
        if self.memory_high_affect_callback_retry_count > 2:
            self.memory_high_affect_callback_retry_count = 2
        if self.memory_prompt_mode not in {"legacy", "subtle_forum"}:
            self.memory_prompt_mode = "legacy"
        if self.memory_reply_context_max_chars < 120:
            self.memory_reply_context_max_chars = 120
        if self.memory_reply_context_max_chars > 800:
            self.memory_reply_context_max_chars = 800
        if self.memory_cross_thread_callback_min_score < 0:
            self.memory_cross_thread_callback_min_score = 0.0
        if self.memory_cross_thread_callback_min_score > 1:
            self.memory_cross_thread_callback_min_score = 1.0
        if self.comment_max_chars < 80:
            self.comment_max_chars = 220
        if self.comment_max_chars > 1200:
            self.comment_max_chars = 1200
        if self.comment_max_sentences < 1:
            self.comment_max_sentences = 2
        if self.comment_max_sentences > 6:
            self.comment_max_sentences = 6
        if self.post_max_chars < 120:
            self.post_max_chars = 420
        if self.post_max_chars > 2200:
            self.post_max_chars = 2200
        if self.post_max_sentences < 1:
            self.post_max_sentences = 3
        if self.post_max_sentences > 8:
            self.post_max_sentences = 8
        if self.reply_rewrite_max_attempts < 0:
            self.reply_rewrite_max_attempts = 0
        if self.reply_rewrite_max_attempts > 2:
            self.reply_rewrite_max_attempts = 2
        if self.anti_repetition_window_comments < 1:
            self.anti_repetition_window_comments = 6
        if self.anti_repetition_window_comments > 30:
            self.anti_repetition_window_comments = 30

        # Local caches (hybrid storage)
        self._memory_cache_social = {}  # other_user_id -> social_card dict
        self._memory_cache_thread = {}  # thread_root_id -> thread_card dict
        self._memory_cache_digest = None  # community digest dict
        self._memory_thread_event_counts = {}  # thread_root_id -> count (client-only)
        self._memory_last_reflection_round = None
        self._memory_reflection_count = 0

        if not self.memory_enabled:
            self.memory_run_id = None
            return

        global _MEMORY_RUN_ID, _MEMORY_RESET_DONE, _MEMORY_LAST_DIGEST_UPDATE_ROUND
        cfg_run_id = agents_cfg.get("memory_run_id")
        normalized_cfg_run_id = _normalize_run_id(cfg_run_id) if cfg_run_id else None
        if normalized_cfg_run_id and normalized_cfg_run_id != _MEMORY_RUN_ID:
            _MEMORY_RUN_ID = normalized_cfg_run_id
            _MEMORY_RESET_DONE = False
            _MEMORY_LAST_DIGEST_UPDATE_ROUND = None
        if _MEMORY_RUN_ID is None:
            _MEMORY_RUN_ID = normalized_cfg_run_id or _normalize_run_id(None)
            _MEMORY_LAST_DIGEST_UPDATE_ROUND = None

        self.memory_run_id = _MEMORY_RUN_ID

        # Ensure run-scoped memory is empty for this run_id.
        if not _MEMORY_RESET_DONE:
            try:
                self._memory_api_post("/memory/reset", {"run_id": self.memory_run_id})
            except Exception as exc:
                logging.warning(
                    "Memory reset failed for run %s; will retry on next agent init: %s",
                    self.memory_run_id,
                    exc,
                )
            else:
                _MEMORY_RESET_DONE = True

    # ------------------------------------------------------------------
    # Structured decision logging (server-side JSON logs).
    #
    # Motivation: explain agent behavior in live experiments by reading server logs.
    # This must be lightweight and never crash the simulation if the logging endpoint is down.
    # ------------------------------------------------------------------

    def _init_decision_logging_config(self, config: dict):
        agents_cfg = {}
        if isinstance(config, dict):
            agents_cfg = config.get("agents", {}) or {}

        def _to_int(val, default):
            try:
                return int(val)
            except Exception:
                return default

        def _to_bool(val, default):
            if val is None:
                return default
            if isinstance(val, bool):
                return val
            return str(val).strip().lower() in {"1", "true", "on", "yes"}

        self.decision_logging_enabled = _to_bool(
            agents_cfg.get("decision_logging_enabled"), True
        )
        self.decision_logging_max_chars = _to_int(agents_cfg.get("decision_logging_max_chars"), 900)
        self.decision_logging_endpoint = (
            str(agents_cfg.get("decision_logging_endpoint") or "/log/agent_decision").strip()
        )

        if self.decision_logging_max_chars < 100:
            self.decision_logging_max_chars = 100
        if self.decision_logging_max_chars > 20000:
            self.decision_logging_max_chars = 20000

    def _decision_log(self, payload: dict, timeout_s: float = 2.0):
        if not getattr(self, "decision_logging_enabled", True):
            return None

        if not isinstance(payload, dict):
            payload = {"value": str(payload)}

        max_chars = int(getattr(self, "decision_logging_max_chars", 900) or 900)

        def _trunc(s: str) -> str:
            if not s:
                return ""
            s = str(s)
            if len(s) <= max_chars:
                return s
            return s[: max_chars - 3].rstrip() + "..."

        for k in ["llm_raw", "prompt_preview", "context_preview", "scan_snippets", "mention_text"]:
            if k in payload and isinstance(payload.get(k), str):
                payload[k] = _trunc(payload.get(k))

        for k in ["path", "duration", "day", "hour", "time"]:
            payload.pop(k, None)

        try:
            api_url = f"{self.base_url}{self.decision_logging_endpoint}"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            post(f"{api_url}", headers=headers, data=json.dumps(payload), timeout=timeout_s)
        except Exception:
            return None

        return None

    def _decision_compact_text(self, value, max_chars: int = 180) -> str:
        """Normalize and truncate text fields for structured decision logs."""
        if value is None:
            return ""
        try:
            txt = str(value)
        except Exception:
            txt = ""
        txt = re.sub(r"\s+", " ", txt).strip()
        if len(txt) <= max_chars:
            return txt
        return txt[: max_chars - 3].rstrip() + "..."

    def _decision_persona_snapshot(self) -> dict:
        """Small persona snapshot to contextualize content-sharing decisions."""
        return {
            "leaning": getattr(self, "leaning", None),
            "language": getattr(self, "language", None),
            "toxicity": getattr(self, "toxicity", None),
            "activity_profile": getattr(self, "activity_profile", None),
        }

    def _memory_api_post(self, path: str, payload: dict, timeout_s: float = 4.0):
        if not getattr(self, "memory_enabled", False):
            return None

        api_url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        try:
            resp = post(f"{api_url}", headers=headers, data=json.dumps(payload), timeout=timeout_s)
        except Exception:
            return None

        try:
            return json.loads(resp.__dict__["_content"].decode("utf-8"))
        except Exception:
            return None

    def _memory_decay_value(self, value: float, delta_rounds: int, lam: float) -> float:
        try:
            v = float(value)
        except Exception:
            v = 0.0
        try:
            d = int(delta_rounds)
        except Exception:
            d = 0
        if d <= 0:
            return v
        try:
            lam = float(lam)
        except Exception:
            lam = 0.0
        if lam <= 0:
            return v
        try:
            return v * math.exp(-lam * d)
        except Exception:
            return v

    def _memory_corrupt_text(self, text: str, rate: float) -> str:
        if not text:
            return text
        try:
            rate = float(rate)
        except Exception:
            rate = 0.0
        if rate <= 0:
            return text

        words = str(text).split()
        if len(words) < 6:
            return text

        kept = []
        dropped = 0
        for w in words:
            if random.random() < rate:
                dropped += 1
                continue
            kept.append(w)

        if not kept or dropped <= 0:
            return text

        out = " ".join(kept).strip()
        return out if out else text

    def _memory_warn(self, message: str):
        try:
            logging.warning(f"[{self.name}][memory] {message}")
        except Exception:
            pass

    def _memory_extract_json(self, raw: str):
        if not raw:
            return None
        s = str(raw).strip()
        if not s:
            return None

        candidates = [s]

        # Prefer fenced JSON payload if present.
        fence = re.search(r"```(?:json)?\s*(.*?)```", s, flags=re.IGNORECASE | re.DOTALL)
        if fence:
            fenced = fence.group(1).strip()
            if fenced:
                candidates.insert(0, fenced)

        # Try best-effort bracket extraction for object or array payloads.
        for open_ch, close_ch in [("{", "}"), ("[", "]")]:
            start = s.find(open_ch)
            end = s.rfind(close_ch)
            if 0 <= start < end:
                candidates.append(s[start : end + 1])

        seen = set()
        for cand in candidates:
            cand = cand.strip()
            if not cand or cand in seen:
                continue
            seen.add(cand)
            for attempt in [cand, re.sub(r",\s*([}\]])", r"\1", cand)]:
                try:
                    return json.loads(attempt)
                except Exception:
                    continue
        return None

    def _memory_extract_topics_from_text(self, text: str):
        if not isinstance(text, str) or not text.strip():
            return []
        match = re.search(r"topics\s*:\s*([^|\n]+)", text, flags=re.IGNORECASE)
        if not match:
            return []
        raw = match.group(1).strip()
        if not raw:
            return []

        out = []
        seen = set()
        for token in raw.split(","):
            t = re.sub(r"\s+", " ", str(token).strip().lower())
            t = re.sub(r"[^a-z0-9 _/+\\-]", "", t).strip()
            if not t or t in seen:
                continue
            seen.add(t)
            out.append(t[:32])
            if len(out) >= 8:
                break
        return out

    def _memory_extract_keywords_from_texts(self, texts, max_tokens: int = 8):
        if not isinstance(texts, list) or not texts:
            return []

        stop = {
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "from",
            "have",
            "been",
            "were",
            "they",
            "them",
            "their",
            "about",
            "just",
            "what",
            "when",
            "where",
            "who",
            "your",
            "you",
            "its",
            "into",
            "also",
            "because",
            "while",
            "would",
            "could",
            "should",
            "really",
            "there",
            "here",
            "like",
            "dont",
            "didnt",
            "cant",
            "wont",
        }

        counts = {}
        for txt in texts:
            if not isinstance(txt, str):
                continue
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_+\-]{2,28}", txt.lower()):
                if token in stop:
                    continue
                counts[token] = int(counts.get(token, 0) or 0) + 1

        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [k for k, _ in ranked[: max(1, int(max_tokens))]]

    def _memory_normalize_toxicity_label(self):
        val = getattr(self, "toxicity", None)
        try:
            s = str(val or "").strip().lower()
        except Exception:
            s = ""
        s = re.sub(r"\s+", " ", s)
        return s

    def _memory_is_low_toxicity_profile(self):
        label = self._memory_normalize_toxicity_label()
        return label in {"no", "absolutely no"}

    def _memory_sanitize_summary_text(self, text: str, max_chars: int = 280):
        if not isinstance(text, str):
            return ""
        out = re.sub(r"\s+", " ", text).strip()
        if not out:
            return ""

        # Keep behavior memory but avoid repeating explicit abusive terms verbatim.
        blocked = [
            "faggot",
            "cocksucker",
            "cunt",
            "shithead",
            "motherfucker",
            "nigger",
        ]
        for term in blocked:
            out = re.sub(rf"\b{re.escape(term)}\b", "[abusive language]", out, flags=re.IGNORECASE)

        return self._memory_truncate(out, max_chars)

    def _memory_extract_behavior_labels(
        self,
        text: str = "",
        *,
        relation_label: str = "",
        tone_label: str = "",
        event_type: str = "",
    ):
        joined = " ".join(
            [
                str(text or ""),
                str(relation_label or ""),
                str(tone_label or ""),
                str(event_type or ""),
            ]
        ).lower()
        if not joined.strip():
            return []

        labels = set()
        if any(t in joined for t in ["spam", "bot repost", "copy paste", "flood"]):
            labels.add("spam")
        if any(t in joined for t in ["bait", "troll", "flamebait", "ragebait"]):
            labels.add("baiting")
        if any(t in joined for t in ["slur", "racist", "homophobic", "hate speech", "bigot"]):
            labels.add("slur")
        if any(t in joined for t in ["dogpile", "brigade", "pile-on", "pile on"]):
            labels.add("dogpile")
        if any(t in joined for t in ["harass", "threat", "abuse", "insult", "bully", "hostile"]):
            labels.add("harassment")
        if (
            str(event_type or "").strip().lower() == "downvote"
            or str(relation_label or "").strip().lower() in {"hostile", "disagree"}
            or str(tone_label or "").strip().lower() in {"angry", "snarky"}
        ):
            labels.add("toxicity_escalation")

        ordered = [
            "harassment",
            "spam",
            "slur",
            "dogpile",
            "baiting",
            "toxicity_escalation",
        ]
        return [lb for lb in ordered if lb in labels][:4]

    def _memory_get_relationship_signal_for_user(self, other_user_id: int, tid: int):
        signal = {
            "user_id": None,
            "username": "",
            "trust_score": 0.0,
            "affinity_score": 0.0,
            "conflict_score": 0.0,
            "interaction_count": 0,
            "interaction_norm": 0.0,
            "recency_score": 0.0,
            "recent_conflict_count": 0,
            "behavior_labels": [],
            "has_social_card": False,
        }
        try:
            uid = int(other_user_id)
        except Exception:
            return signal
        signal["user_id"] = uid

        ctx = None
        try:
            ctx = self._memory_fetch_context(other_user_id=uid, thread_root_id=None)
        except Exception:
            ctx = None
        if not isinstance(ctx, dict):
            ctx = {}

        sc = ctx.get("social_card")
        if not isinstance(sc, dict):
            sc = {}

        def _stat(name: str):
            try:
                return float(sc.get(name) or 0.0)
            except Exception:
                return 0.0

        trust = _stat("trust")
        affinity = _stat("affinity")
        conflict = _stat("conflict")

        def _norm(v):
            try:
                v = float(v) / 5.0
            except Exception:
                v = 0.0
            return max(-1.0, min(1.0, v))

        signal["trust_score"] = _norm(trust)
        signal["affinity_score"] = _norm(affinity)
        signal["conflict_score"] = _norm(conflict)
        signal["has_social_card"] = bool(sc)

        try:
            signal["username"] = str(
                sc.get("other_username") or ctx.get("other_username") or ""
            ).strip().lstrip("@")
        except Exception:
            signal["username"] = ""

        event_count = 0
        try:
            event_count = int(sc.get("event_count") or 0)
        except Exception:
            event_count = 0

        recent_pair_events = ctx.get("recent_pair_events")
        if not isinstance(recent_pair_events, list):
            recent_pair_events = []
        window_rounds = int(
            getattr(self, "memory_relationship_priority_window_rounds", 120) or 120
        )
        if window_rounds <= 0:
            window_rounds = 120

        if recent_pair_events:
            filtered_events = []
            for ev in recent_pair_events:
                if not isinstance(ev, dict):
                    continue
                rid = ev.get("round_id")
                try:
                    rid = int(rid)
                except Exception:
                    rid = None
                if rid is None:
                    filtered_events.append(ev)
                    continue
                try:
                    delta = int(tid) - rid
                except Exception:
                    delta = 0
                if delta <= window_rounds:
                    filtered_events.append(ev)
            recent_pair_events = filtered_events

        recent_conflict = 0
        behavior_counts = {}
        for ev in recent_pair_events:
            if not isinstance(ev, dict):
                continue
            rel = str(ev.get("relation_label") or "").strip().lower()
            tone = str(ev.get("tone_label") or "").strip().lower()
            evt = str(ev.get("event_type") or "").strip().lower()
            claim = str(ev.get("salient_claim") or "")
            if rel in {"hostile", "disagree"} or tone in {"angry", "snarky"} or evt == "downvote":
                recent_conflict += 1
            for lb in self._memory_extract_behavior_labels(
                claim, relation_label=rel, tone_label=tone, event_type=evt
            ):
                behavior_counts[lb] = int(behavior_counts.get(lb, 0) or 0) + 1

        if recent_pair_events:
            interaction_count = len(recent_pair_events)
        else:
            fallback_cap = max(1, int(window_rounds / 24))
            interaction_count = min(max(0, event_count), fallback_cap)
        signal["interaction_count"] = int(max(0, interaction_count))
        signal["interaction_norm"] = min(1.0, float(signal["interaction_count"]) / 8.0)
        signal["recent_conflict_count"] = int(recent_conflict)
        signal["behavior_labels"] = [
            k
            for k, _ in sorted(behavior_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        ]

        last_round = sc.get("last_round_id")
        if last_round is None:
            last_round = sc.get("last_updated_round")
        try:
            delta = max(0, int(tid) - int(last_round)) if last_round is not None else None
        except Exception:
            delta = None
        if delta is None:
            recency = 0.0
        else:
            lam = float(getattr(self, "memory_social_decay_lambda", 0.05) or 0.05)
            recency = float(math.exp(-max(0.0, lam) * float(delta)))
        signal["recency_score"] = max(0.0, min(1.0, recency))

        return signal

    def _memory_json_loads_maybe(self, value):
        if value is None:
            return None
        if isinstance(value, (list, dict)):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return json.loads(stripped)
            except Exception:
                return stripped
        return value

    def _memory_parse_root_post_text(self, text_value: str):
        raw_text = str(text_value or "").strip()
        if not raw_text:
            return {"title": "", "body": "", "combined": ""}
        validation = self._validate_structured_post_text(raw_text)
        if validation.get("valid"):
            title = (validation.get("title") or "").strip()
            body = (validation.get("body") or "").strip()
            combined = " ".join([part for part in [title, body] if part]).strip()
            return {"title": title, "body": body, "combined": combined}
        if raw_text.lower().startswith("title:"):
            body_parts = raw_text.split("\n", 1)
            title = body_parts[0][6:].strip()
            body = body_parts[1].strip() if len(body_parts) > 1 else ""
            combined = " ".join([part for part in [title, body] if part]).strip()
            return {"title": title, "body": body, "combined": combined}
        raw = re.sub(r"\s+", " ", raw_text)
        parts = raw_text.split("\n\n", 1)
        title = parts[0].replace("TITLE:", "").strip() if parts else raw[:120]
        body = parts[1].strip() if len(parts) > 1 else raw
        combined = " ".join([part for part in [title, body] if part]).strip()
        return {"title": title, "body": body, "combined": combined}

    def _memory_get_recent_root_posts(self, *, tid: int, limit: int = 24, rounds_back: int = 12):
        try:
            current_round = int(tid)
        except Exception:
            current_round = 0
        lower_bound = max(0, current_round - max(1, int(rounds_back or 12)) + 1)
        try:
            result = session.execute(
                text(
                    """
                    SELECT id, round, user_id, news_id, image_post_id, image_id, tweet
                    FROM post
                    WHERE id = thread_id
                      AND round <= :tid
                      AND round >= :lower_bound
                    ORDER BY round DESC, id DESC
                    LIMIT :limit
                    """
                ),
                {"tid": current_round, "lower_bound": lower_bound, "limit": int(limit or 24)},
            )
        except Exception:
            return []

        rows = []
        for row in result.fetchall():
            rows.append(
                {
                    "id": row[0],
                    "round": row[1],
                    "user_id": row[2],
                    "news_id": row[3],
                    "image_post_id": row[4],
                    "image_id": row[5],
                    "tweet": row[6],
                }
            )
        return rows

    def _memory_digest_maturity_meta(self, root_posts):
        if not isinstance(root_posts, list):
            root_posts = []
        distinct_authors = set()
        for row in root_posts:
            if not isinstance(row, dict):
                continue
            try:
                distinct_authors.add(int(row.get("user_id")))
            except Exception:
                continue
        root_count = len([row for row in root_posts if isinstance(row, dict)])
        return {
            "root_count": int(root_count),
            "distinct_author_count": int(len(distinct_authors)),
            "mature": bool(root_count >= 6 and len(distinct_authors) >= 4),
        }

    def _memory_root_post_style_signals(self, root_post: dict):
        if not isinstance(root_post, dict):
            return {"norms": [], "memes": [], "topics": []}

        parsed = self._memory_parse_root_post_text(root_post.get("tweet") or "")
        title = (parsed.get("title") or "").strip()
        body = (parsed.get("body") or "").strip()
        combined = " ".join([title, body]).strip().lower()
        norms = []
        memes = []

        is_news_root = bool((root_post.get("news_id") or -1) not in (-1, 0, None))
        is_image_root = bool(
            (root_post.get("image_post_id") or -1) not in (-1, 0, None)
            or (root_post.get("image_id") or -1) not in (-1, 0, None)
        )

        if is_news_root or is_image_root:
            norms.append("reaction-driven share posts")
        else:
            norms.append("freeform opinion posts")

        title_lower = title.lower()
        if title_lower.startswith(("why ", "how ", "what ", "should ", "can ", "is ", "are ")):
            norms.append("question-bait titles")
            memes.append("why-is-this-happening threads")
        if "?" in title or "?" in body:
            norms.append("question-led hooks")
        if any(
            token in combined
            for token in ["reboot", "original", "better than", "instead of", "vs ", "versus", "compared to"]
        ):
            norms.append("comparison-heavy posts")
        if any(token in combined for token in ["90s", "nostalgia", "used to", "back then", "original"]):
            norms.append("nostalgia framing")
        if any(
            token in combined
            for token in ["hot mess", "disaster", "garbage", "trash", "flop", "sleeping on", "needs to", "still bothers me"]
        ):
            norms.append("blunt hot takes")
        if any(token in combined for token in ["why everyone", "sleeping on"]):
            memes.append("sleeping-on takes")
        if any(token in combined for token in ["hot mess", "disaster waiting to happen"]):
            memes.append("hot-mess framing")
        if "instead of" in combined:
            memes.append("instead-of comparisons")
        if any(token in combined for token in ["fuck", "shit", "wtf", "damn"]):
            norms.append("casual profanity")
        if len(body) <= 180:
            norms.append("short punchy bodies")

        topics = self._memory_extract_topics_from_text(parsed.get("combined") or "")
        return {
            "norms": norms,
            "memes": memes,
            "topics": topics[:4],
        }

    def _memory_style_phrase_ok(self, phrase: str) -> bool:
        text_value = re.sub(r"\s+", " ", str(phrase or "").strip().lower())
        if not text_value:
            return False
        if any(ch in text_value for ch in ['"', "'", "@", "http", "/", "\\"]):
            return False
        tokens = [tok for tok in re.findall(r"[a-z]+", text_value) if tok]
        if not tokens or len(tokens) > 8:
            return False
        return True

    def _memory_build_community_digest_fallback(self, root_posts, prev_digest_text: str = ""):
        if not isinstance(root_posts, list):
            root_posts = []

        def _pick_top(counter: dict, n: int):
            ranked = sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))
            out = []
            for key, _ in ranked:
                skey = str(key).strip()
                if not skey:
                    continue
                out.append(skey)
                if len(out) >= n:
                    break
            return out

        def _dedupe(values, n: int):
            out = []
            seen = set()
            for val in values:
                s = re.sub(r"\s+", " ", str(val or "").strip().lower())
                if not s:
                    continue
                if s in seen:
                    continue
                seen.add(s)
                out.append(s)
                if len(out) >= n:
                    break
            return out

        topic_counts = {}
        norm_counts = {}
        meme_counts = {}
        style_markers = []

        for row in root_posts[:48]:
            signals = self._memory_root_post_style_signals(row)
            for norm in signals.get("norms") or []:
                norm_counts[norm] = int(norm_counts.get(norm, 0) or 0) + 1
                style_markers.append(norm)
            for meme in signals.get("memes") or []:
                meme_counts[meme] = int(meme_counts.get(meme, 0) or 0) + 1
            for topic in signals.get("topics") or []:
                topic_counts[topic] = int(topic_counts.get(topic, 0) or 0) + 1

        norms = _pick_top(norm_counts, 6)
        memes = [
            key
            for key, count in sorted(meme_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            if count >= 2
        ][:6]
        top_topics = _pick_top(topic_counts, 8)

        digest_text = ""
        if norms:
            digest_text = "posting style lately leans toward " + ", ".join(norms[:3]) + "."
        elif prev_digest_text:
            digest_text = prev_digest_text.strip()
        else:
            digest_text = "posting style lately leans toward short opinion posts and conversational hooks."

        polarizing_issues = []
        if "comparison-heavy posts" in norms:
            polarizing_issues.append("comparison-first arguments")
        if "blunt hot takes" in norms:
            polarizing_issues.append("sweeping verdict threads")
        if "question-bait titles" in norms:
            polarizing_issues.append("prompting everyone to pick a side")

        return {
            "digest_text": self._memory_truncate(digest_text.lower(), 700),
            "top_topics": top_topics[:8],
            "norms": _dedupe(norms, 6),
            "memes": _dedupe(memes, 6),
            "polarizing_issues": _dedupe(polarizing_issues, 6),
        }

    def _memory_build_reflections_fallback(
        self,
        *,
        items,
        other_user_id=None,
        other_username=None,
        community_digest_text: str = "",
    ):
        if not isinstance(items, list) or not items:
            return []

        topic_counts = {}
        other_user_counts = {}
        other_username_counts = {}
        post_counts = {}
        support_event_ids = []
        conflict_score = 0
        support_score = 0
        memorable_claims = []
        behavior_counts = {}

        for it in items[:24]:
            if isinstance(it, str):
                it = {"text": it}
            if not isinstance(it, dict):
                continue

            text_value = str(it.get("text") or "")
            text_l = text_value.lower()
            for token in ["disagree", "hostile", "downvote", "snarky", "angry", "skeptic"]:
                if token in text_l:
                    conflict_score += 1
            for token in ["agree", "helpful", "upvote", "supportive", "funny"]:
                if token in text_l:
                    support_score += 1
            for lb in self._memory_extract_behavior_labels(text_value):
                behavior_counts[lb] = int(behavior_counts.get(lb, 0) or 0) + 1

            topic_tags = it.get("topic_tags")
            if isinstance(topic_tags, str):
                try:
                    topic_tags = json.loads(topic_tags)
                except Exception:
                    topic_tags = []
            if not isinstance(topic_tags, list):
                topic_tags = []

            for tag in topic_tags:
                t = re.sub(r"\s+", " ", str(tag).strip().lower())
                if not t:
                    continue
                topic_counts[t] = int(topic_counts.get(t, 0) or 0) + 1

            for topic in self._memory_extract_topics_from_text(text_value):
                topic_counts[topic] = int(topic_counts.get(topic, 0) or 0) + 1

            if isinstance(text_value, str) and text_value.strip():
                snippet = re.sub(r"\s+", " ", text_value).strip()
                if len(snippet) > 120:
                    snippet = snippet[:117].rstrip() + "..."
                if snippet:
                    memorable_claims.append(snippet)

            uid = it.get("other_user_id")
            try:
                if uid is not None:
                    uid = int(uid)
                    other_user_counts[uid] = int(other_user_counts.get(uid, 0) or 0) + 1
            except Exception:
                pass
            uname = it.get("target_username") or it.get("other_username")
            try:
                uname = str(uname).strip().lstrip("@")
            except Exception:
                uname = ""
            if uname:
                other_username_counts[uname] = int(other_username_counts.get(uname, 0) or 0) + 1

            for pid_key in ["target_post_id", "thread_root_id", "actor_post_id"]:
                pid = it.get(pid_key)
                try:
                    if pid is None:
                        continue
                    pid = int(pid)
                    post_counts[pid] = int(post_counts.get(pid, 0) or 0) + 1
                except Exception:
                    continue

            sid_values = it.get("supporting_event_ids")
            if isinstance(sid_values, list):
                for sid in sid_values:
                    try:
                        support_event_ids.append(int(sid))
                    except Exception:
                        continue
            else:
                sid = it.get("source_event_id")
                try:
                    if sid is not None:
                        support_event_ids.append(int(sid))
                except Exception:
                    pass

        top_topics = [
            k
            for k, _ in sorted(topic_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
            if str(k).strip()
        ]

        if not top_topics:
            texts = [str(it.get("text") or "") for it in items[:15] if isinstance(it, dict)]
            top_topics = self._memory_extract_keywords_from_texts(texts, max_tokens=8)

        top_behavior_labels = [
            k
            for k, _ in sorted(behavior_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        ]

        dedup_event_ids = []
        seen_ids = set()
        for sid in support_event_ids:
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            dedup_event_ids.append(int(sid))
            if len(dedup_event_ids) >= 16:
                break

        relationship_uid = None
        try:
            if other_user_id is not None:
                relationship_uid = int(other_user_id)
        except Exception:
            relationship_uid = None
        if relationship_uid is None and other_user_counts:
            relationship_uid = sorted(other_user_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

        relationship_uname = ""
        try:
            relationship_uname = str(other_username or "").strip().lstrip("@")
        except Exception:
            relationship_uname = ""
        if not relationship_uname and other_username_counts:
            relationship_uname = sorted(other_username_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

        stance = "contested"
        if support_score > conflict_score:
            stance = "mostly cooperative"
        elif conflict_score == support_score:
            stance = "mixed"

        reflections = []

        memorable_topic = ", ".join(top_topics[:3]) if top_topics else "recurring debate cycles"
        if community_digest_text:
            community_txt = (
                "Community bridge: keep replies anchored to this forum vibe: "
                + self._memory_truncate(community_digest_text, 170)
            )
        else:
            community_txt = (
                "Community bridge: recurring topical experience centers on "
                + memorable_topic
                + "; keep future replies grounded there."
            )
        if top_behavior_labels:
            community_txt += " Watch for " + ", ".join(top_behavior_labels[:2]) + "."
        reflections.append(
            {
                "text": self._memory_sanitize_summary_text(community_txt, 280),
                "importance": 0.72,
                "supporting_event_ids": dedup_event_ids[:8],
                "topic_tags": top_topics[:4],
                "link_kind": "community",
                "facets": {
                    "topical_experience": self._memory_truncate(memorable_topic, 120),
                    "behavior_labels": top_behavior_labels[:3],
                },
            }
        )

        memorable_line = ""
        if memorable_claims:
            memorable_line = memorable_claims[0]
        core_txt = (
            f"Core memory: discussions have been {stance} around {memorable_topic}; "
            "keep continuity while adapting claim-by-claim."
        )
        if top_behavior_labels:
            core_txt += " Avoid escalating " + ", ".join(top_behavior_labels[:1]) + "."
        reflections.append(
            {
                "text": self._memory_sanitize_summary_text(core_txt, 280),
                "importance": 0.64,
                "supporting_event_ids": dedup_event_ids[:8],
                "topic_tags": top_topics[:4],
                "link_kind": "core",
                "facets": {
                    "topical_experience": self._memory_truncate(memorable_topic, 120),
                    "negative_experience": self._memory_truncate(
                        "conflict spikes" if conflict_score >= support_score else "occasional friction",
                        120,
                    ),
                    "behavior_labels": top_behavior_labels[:3],
                },
            }
        )

        if top_topics:
            memorable_post = ""
            if post_counts:
                memorable_post = str(sorted(post_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0])
            topic_txt = (
                "Topic memory: repeated experiences on "
                + ", ".join(top_topics[:3])
                + " keep returning in heated comment chains."
            )
            if memorable_post:
                topic_txt += f" Memorable anchor: post_id={memorable_post}."
            reflections.append(
                {
                    "text": self._memory_sanitize_summary_text(topic_txt, 280),
                    "importance": 0.69,
                    "supporting_event_ids": dedup_event_ids[:10],
                    "topic_tags": top_topics[:4],
                    "link_kind": "topic",
                    "facets": {
                        "topical_experience": self._memory_truncate(", ".join(top_topics[:4]), 120),
                        "memorable_entities": (
                            [f"post_id={memorable_post}"] if memorable_post else []
                        ),
                        "behavior_labels": top_behavior_labels[:3],
                    },
                }
            )

        if relationship_uid is not None or relationship_uname:
            rel_mode = "friction" if conflict_score >= support_score else "alignment"
            memorable_post = ""
            if post_counts:
                memorable_post = str(sorted(post_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0])
            if relationship_uname:
                rel_txt = (
                    f"Relationship memory: repeated exchanges with @{relationship_uname} show {rel_mode};"
                    " preserve that continuity when relevant."
                )
            else:
                rel_txt = (
                    f"Relationship memory: repeated exchanges with a recurring user show {rel_mode};"
                    " preserve that continuity when relevant."
                )
            if memorable_post:
                rel_txt += f" Memorable touchpoint: post_id={memorable_post}."
            reflections.append(
                {
                    "text": self._memory_sanitize_summary_text(rel_txt, 280),
                    "importance": 0.67,
                    "supporting_event_ids": dedup_event_ids[:10],
                    "topic_tags": top_topics[:4],
                    "link_kind": "relationship",
                    "facets": {
                        "memorable_entities": (
                            ([f"@{relationship_uname}"] if relationship_uname else [])
                            + ([f"post_id={memorable_post}"] if memorable_post else [])
                        )[:4],
                        "negative_experience": self._memory_truncate(
                            "repeated disagreement patterns" if rel_mode == "friction" else "",
                            120,
                        ),
                        "behavior_labels": top_behavior_labels[:3],
                    },
                }
            )

        return reflections[:4]

    def _memory_get_thread_root_id(self, post_id: int):
        try:
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            api_url = f"{self.base_url}/get_thread_root"
            response = get(
                f"{api_url}",
                headers=headers,
                data=json.dumps({"post_id": int(post_id)}),
            )
            root_id = json.loads(response.__dict__["_content"].decode("utf-8"))
            return int(root_id)
        except Exception:
            return None

    def _memory_get_author_id_and_username(self, post_id: int):
        """
        Return (user_id, username) for a post/comment. Best-effort.
        """
        try:
            uid, uname = self.get_username_from_post(int(post_id))
            try:
                uid = int(uid) if uid is not None else None
            except Exception:
                uid = None
            if uname is not None:
                try:
                    uname = str(uname).strip().lstrip("@")
                except Exception:
                    uname = None
            return uid, uname
        except Exception:
            pass

        try:
            uname = self.get_user_from_post(int(post_id))
            if uname is not None:
                uname = str(uname).strip().lstrip("@")
        except Exception:
            uname = None

        return None, uname

    def _memory_fetch_context(self, other_user_id=None, thread_root_id=None):
        if not getattr(self, "memory_enabled", False):
            return None
        if not getattr(self, "memory_run_id", None):
            return None

        payload = {"run_id": self.memory_run_id, "agent_user_id": int(self.user_id)}
        if other_user_id is not None:
            try:
                payload["other_user_id"] = int(other_user_id)
            except Exception:
                pass
        if thread_root_id is not None:
            try:
                payload["thread_root_id"] = int(thread_root_id)
            except Exception:
                pass
        payload["pair_limit"] = int(getattr(self, "memory_pair_limit", 5) or 5)

        res = self._memory_api_post("/memory/get_context", payload)
        if not isinstance(res, dict) or res.get("status") != 200:
            return None

        try:
            sc = res.get("social_card")
            if isinstance(sc, dict) and other_user_id is not None:
                self._memory_cache_social[int(other_user_id)] = sc
        except Exception:
            pass
        try:
            tc = res.get("thread_card")
            if isinstance(tc, dict) and thread_root_id is not None:
                self._memory_cache_thread[int(thread_root_id)] = tc
        except Exception:
            pass
        try:
            cd = res.get("community_digest")
            if isinstance(cd, dict):
                self._memory_cache_digest = cd
        except Exception:
            pass

        return res

    def _memory_truncate(self, text: str, max_chars: int):
        if not isinstance(text, str):
            return ""
        if max_chars is None or max_chars <= 0:
            return text.strip()
        out = text.strip()
        if len(out) <= max_chars:
            return out
        return out[: max_chars - 3].rstrip() + "..."

    def _memory_search(
        self,
        *,
        query_text: str,
        other_user_id=None,
        thread_root_id=None,
        topic_tags=None,
        time_window_rounds=None,
        types=None,
        k=None,
        max_chars=None,
        include_evidence_tail=False,
        round_id=None,
    ):
        if not getattr(self, "memory_enabled", False):
            return None
        if not getattr(self, "memory_run_id", None):
            return None
        if not getattr(self, "memory_semantic_enabled", True):
            return None
        if not isinstance(query_text, str) or not query_text.strip():
            return None

        payload = {
            "run_id": self.memory_run_id,
            "agent_user_id": int(self.user_id),
            "query_text": query_text.strip(),
            "k": int(k if k is not None else getattr(self, "memory_search_k", 8)),
            "max_chars": int(
                max_chars
                if max_chars is not None
                else getattr(
                    self,
                    "memory_search_max_chars",
                    DEFAULT_MEMORY_SEARCH_MAX_CHARS,
                )
            ),
            "include_evidence_tail": bool(include_evidence_tail),
        }
        if other_user_id is not None:
            try:
                payload["other_user_id"] = int(other_user_id)
            except Exception:
                pass
        if thread_root_id is not None:
            try:
                payload["thread_root_id"] = int(thread_root_id)
            except Exception:
                pass
        if time_window_rounds is not None:
            try:
                payload["time_window_rounds"] = int(time_window_rounds)
            except Exception:
                pass
        elif getattr(self, "memory_search_time_window_rounds", None) is not None:
            payload["time_window_rounds"] = int(self.memory_search_time_window_rounds)
        if round_id is not None:
            try:
                payload["round_id"] = int(round_id)
            except Exception:
                pass
        if isinstance(topic_tags, list) and topic_tags:
            payload["topic_tags"] = topic_tags
        if isinstance(types, list) and types:
            payload["types"] = types

        res = self._memory_api_post("/memory/search", payload)
        if not isinstance(res, dict) or res.get("status") != 200:
            return None
        return res

    def _memory_format_search_brief(self, search_res: dict, max_chars: int):
        if not isinstance(search_res, dict):
            return ""
        items = search_res.get("items")
        if isinstance(items, list):
            lines = ["[MEMORY SEARCH BRIEF]"]
            for it in items[:10]:
                if not isinstance(it, dict):
                    continue
                if self._memory_is_vote_artifact(
                    text_value=(it.get("text_humanized") or it.get("text") or ""),
                    relation_label=it.get("relation_label") or "",
                    metadata=it.get("metadata") if isinstance(it.get("metadata"), dict) else None,
                ):
                    continue
                txt = (it.get("text_humanized") or it.get("text") or "").strip()
                if not txt:
                    continue
                if len(txt) > 220:
                    txt = txt[:217].rstrip() + "..."
                rid = it.get("round_id")
                score = it.get("score")
                try:
                    score = f"{float(score):.2f}"
                except Exception:
                    score = "?"
                label_bits = []
                target_uname = str(it.get("target_username") or it.get("other_username") or "").strip().lstrip("@")
                if target_uname:
                    label_bits.append(f"target=@{target_uname}")
                target_post_id = it.get("target_post_id")
                thread_root_id = it.get("thread_root_id")
                if target_post_id is not None:
                    label_bits.append(f"target_post_id={target_post_id}")
                if thread_root_id is not None:
                    label_bits.append(f"thread_root_id={thread_root_id}")
                label_suffix = (" " + " ".join(label_bits)) if label_bits else ""
                lines.append(f"- ({it.get('item_type')}, r{rid}, s={score}){label_suffix} {txt}")
            if len(lines) > 1:
                return self._memory_truncate("\n".join(lines), max_chars)

        brief = search_res.get("memory_brief")
        if isinstance(brief, str) and brief.strip():
            return self._memory_truncate(brief.strip(), max_chars)
        return ""

    def _memory_build_query_text(self, *parts):
        tokens = []
        for p in parts:
            if p is None:
                continue
            s = str(p).strip()
            if not s:
                continue
            tokens.append(s)
        if not tokens:
            return "recent social memory context"
        return "\n".join(tokens)[-4000:]

    def _memory_use_subtle_prompt_mode(self):
        return str(getattr(self, "memory_prompt_mode", "legacy") or "").strip().lower() == "subtle_forum"

    @staticmethod
    def _memory_loads_maybe(value):
        if value is None:
            return None
        if isinstance(value, (list, dict)):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return json.loads(stripped)
            except Exception:
                return stripped
        return value

    def _memory_is_vote_artifact(
        self,
        *,
        text_value: str = "",
        event_type: str = "",
        relation_label: str = "",
        metadata=None,
    ) -> bool:
        parts = [
            str(event_type or ""),
            str(relation_label or ""),
            str(text_value or ""),
        ]
        if isinstance(metadata, dict):
            for key in (
                "event_type",
                "source_event_type",
                "relation_label",
                "source_relation_label",
                "text",
            ):
                val = metadata.get(key)
                if val is not None:
                    parts.append(str(val))
        blob = " ".join([p for p in parts if p]).strip().lower()
        if not blob:
            return False
        return bool(re.search(r"\b(upvot\w*|downvot\w*)\b", blob))

    def _memory_sanitize_prompt_memory_text(self, text: str, max_chars: int = 140):
        if not isinstance(text, str):
            return ""
        raw = text.strip()
        if not raw:
            return ""

        chunks = re.split(r"(?:\n+|(?<=[.!?])\s+)", raw)
        kept = []
        for chunk in chunks:
            clean = re.sub(r"\s+", " ", str(chunk or "")).strip(" -")
            if not clean:
                continue
            if self._memory_is_vote_artifact(text_value=clean):
                continue
            kept.append(clean)

        if not kept:
            return ""

        cleaned = " ".join(kept)
        cleaned = re.sub(r"^\[[^\]]+\]\s*", "", cleaned).strip()
        return self._memory_truncate(cleaned, max_chars)

    def _memory_build_reply_context(
        self,
        *,
        query_text: str,
        other_user_id=None,
        thread_root_id=None,
        other_username=None,
        round_id=None,
    ):
        meta = {
            "search_used": False,
            "degraded_mode": False,
            "embedding_degraded": False,
            "no_ready_candidates": False,
            "tier_c_used": False,
            "retrieved_item_count": 0,
            "top_score": None,
            "fallback_used": False,
            "continuity_text": "",
            "cross_thread_callback_candidate": False,
            "cross_thread_callback_score": None,
        }
        if not getattr(self, "memory_enabled", False):
            return "", meta
        if not getattr(self, "memory_run_id", None):
            return "", meta

        try:
            context_cap = int(getattr(self, "memory_reply_context_max_chars", 280) or 280)
        except Exception:
            context_cap = 280

        ctx = self._memory_fetch_context(
            other_user_id=other_user_id,
            thread_root_id=thread_root_id,
        ) or {}

        username = str(other_username or "").strip().lstrip("@")
        if not username:
            username = "this user"

        metrics_line = ""
        continuity_parts = []
        local_signal_count = 0

        sc = ctx.get("social_card") if isinstance(ctx, dict) else None
        if isinstance(sc, dict):
            try:
                metrics_line = (
                    f"History with @{username}: "
                    f"affinity={float(sc.get('affinity') or 0.0):.2f}, "
                    f"conflict={float(sc.get('conflict') or 0.0):.2f}, "
                    f"humor={float(sc.get('humor') or 0.0):.2f}, "
                    f"trust={float(sc.get('trust') or 0.0):.2f}"
                )
            except Exception:
                metrics_line = ""

            summary_text = self._memory_sanitize_prompt_memory_text(
                sc.get("summary_text") or "",
                max_chars=120,
            )
            if summary_text:
                continuity_parts.append(f"with @{username}: {summary_text}")
                local_signal_count += 1

        tc = ctx.get("thread_card") if isinstance(ctx, dict) else None
        if isinstance(tc, dict):
            gist_text = self._memory_sanitize_prompt_memory_text(
                tc.get("gist_text") or "",
                max_chars=120,
            )
            if gist_text:
                continuity_parts.append(f"this thread: {gist_text}")
                local_signal_count += 1

        recent_pair_events = ctx.get("recent_pair_events") if isinstance(ctx, dict) else None
        if isinstance(recent_pair_events, list):
            for ev in reversed(recent_pair_events):
                if not isinstance(ev, dict):
                    continue
                ev_thread_root = ev.get("thread_root_id")
                try:
                    if (
                        thread_root_id is not None
                        and ev_thread_root is not None
                        and int(ev_thread_root) != int(thread_root_id)
                    ):
                        continue
                except Exception:
                    pass
                if self._memory_is_vote_artifact(
                    text_value=ev.get("salient_claim") or "",
                    event_type=ev.get("event_type") or "",
                    relation_label=ev.get("relation_label") or "",
                ):
                    continue
                claim = self._memory_sanitize_prompt_memory_text(
                    ev.get("salient_claim") or "",
                    max_chars=100,
                )
                if not claim:
                    continue
                continuity_parts.append(f"earlier here: {claim}")
                local_signal_count += 1
                break

        cross_thread_min_score = float(
            getattr(self, "memory_cross_thread_callback_min_score", 0.80) or 0.80
        )
        callback_score = None
        callback_text = ""
        if other_user_id is not None:
            search_res = self._memory_search(
                query_text=query_text,
                other_user_id=other_user_id,
                thread_root_id=None,
                types=["summary", "reflection"],
                round_id=round_id,
                k=4,
                max_chars=420,
            )
            if isinstance(search_res, dict):
                meta["search_used"] = True
                retrieval_meta = (
                    search_res.get("retrieval_meta")
                    if isinstance(search_res.get("retrieval_meta"), dict)
                    else {}
                )
                meta["degraded_mode"] = bool(retrieval_meta.get("degraded_mode", False))
                meta["embedding_degraded"] = bool(
                    retrieval_meta.get("embedding_degraded", False)
                )
                meta["no_ready_candidates"] = bool(
                    retrieval_meta.get("no_ready_candidates", False)
                )
                items = search_res.get("items")
                if isinstance(items, list):
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        try:
                            item_score = float(item.get("score") or 0.0)
                        except Exception:
                            item_score = 0.0
                        if meta["top_score"] is None:
                            meta["top_score"] = item_score
                        item_thread_root = item.get("thread_root_id")
                        try:
                            if (
                                thread_root_id is not None
                                and item_thread_root is not None
                                and int(item_thread_root) == int(thread_root_id)
                            ):
                                continue
                        except Exception:
                            pass
                        item_other = item.get("other_user_id")
                        try:
                            if (
                                other_user_id is not None
                                and item_other is not None
                                and int(item_other) != int(other_user_id)
                            ):
                                continue
                        except Exception:
                            pass
                        if item_score < cross_thread_min_score:
                            continue
                        metadata = (
                            item.get("metadata")
                            if isinstance(item.get("metadata"), dict)
                            else None
                        )
                        if self._memory_is_vote_artifact(
                            text_value=(item.get("text_humanized") or item.get("text") or ""),
                            relation_label=item.get("relation_label") or "",
                            metadata=metadata,
                        ):
                            continue
                        clean = self._memory_sanitize_prompt_memory_text(
                            item.get("text_humanized") or item.get("text") or "",
                            max_chars=110,
                        )
                        if not clean:
                            continue
                        callback_text = clean
                        callback_score = item_score
                        break

        if callback_text:
            continuity_parts.append(f"if clearly relevant: {callback_text}")
            meta["cross_thread_callback_candidate"] = True
            meta["cross_thread_callback_score"] = callback_score
            meta["top_score"] = callback_score

        continuity_text = self._memory_truncate(
            " | ".join(continuity_parts),
            context_cap,
        )

        if meta["top_score"] is None and local_signal_count > 0:
            meta["top_score"] = 0.72 if local_signal_count >= 2 else 0.58

        meta["retrieved_item_count"] = int(local_signal_count) + (1 if callback_text else 0)
        meta["continuity_text"] = continuity_text

        memory_lines = []
        if metrics_line:
            memory_lines.append(metrics_line)
        if continuity_text:
            memory_lines.append("Continuity: " + continuity_text)
        return "\n".join(memory_lines).strip(), meta

    def _memory_build_post_style_context(self, *, tid: int):
        meta = {
            "usage": "none",
            "root_count": 0,
            "distinct_author_count": 0,
            "mature": False,
        }
        if not getattr(self, "memory_enabled", False):
            return "", meta
        if not getattr(self, "memory_run_id", None):
            return "", meta

        root_posts = self._memory_get_recent_root_posts(tid=int(tid))
        maturity = self._memory_digest_maturity_meta(root_posts)
        meta.update(maturity)
        if not maturity.get("mature"):
            return "", meta

        digest = None
        if isinstance(getattr(self, "_memory_cache_digest", None), dict):
            digest = self._memory_cache_digest
        else:
            ctx = self._memory_fetch_context(other_user_id=None, thread_root_id=None) or {}
            digest = ctx.get("community_digest") if isinstance(ctx, dict) else None
        if not isinstance(digest, dict):
            return "", meta

        norms = self._memory_json_loads_maybe(
            digest.get("norms") if digest.get("norms") is not None else digest.get("norms_json")
        )
        memes = self._memory_json_loads_maybe(
            digest.get("memes") if digest.get("memes") is not None else digest.get("memes_json")
        )
        norm_lines = []
        meme_lines = []
        if isinstance(norms, list):
            norm_lines = [
                re.sub(r"\s+", " ", str(item or "").strip().lower())
                for item in norms
                if self._memory_style_phrase_ok(item)
            ][:4]
        if isinstance(memes, list):
            meme_lines = [
                re.sub(r"\s+", " ", str(item or "").strip().lower())
                for item in memes
                if self._memory_style_phrase_ok(item)
            ][:3]

        if not norm_lines and not meme_lines:
            return "", meta

        lines = ["[COMMUNITY STYLE GUIDE]"]
        if norm_lines:
            lines.append("posting style lately: " + ", ".join(norm_lines[:3]))
        if meme_lines:
            lines.append("recurring formats: " + ", ".join(meme_lines[:2]))
        lines.append("use this only for tone and structure, not as a topic list")
        meta["usage"] = "style_digest_only"
        return self._memory_truncate("\n".join(lines), 280), meta

    def _memory_build_thread_browse_context(self, *, thread_root_id: int, tid: int):
        meta = {"usage": "none"}
        if not getattr(self, "memory_enabled", False):
            return "", meta
        if thread_root_id is None:
            return "", meta

        ctx = self._memory_fetch_context(
            other_user_id=None,
            thread_root_id=int(thread_root_id),
        ) or {}
        thread_card = ctx.get("thread_card") if isinstance(ctx, dict) else None
        if not isinstance(thread_card, dict):
            return "", meta

        lines = []
        gist = self._memory_sanitize_prompt_memory_text(
            thread_card.get("gist_text") or "",
            max_chars=160,
        )
        if gist:
            lines.append("[THREAD MEMORY]")
            lines.append("thread gist: " + gist)
        role = str(thread_card.get("my_role") or "").strip().lower()
        if role:
            lines.append("my role here: " + role)
        if not lines:
            return "", meta
        meta["usage"] = "thread_local"
        return self._memory_truncate("\n".join(lines), 220), meta

    def _post_root_origin_kind(self, row: dict) -> str:
        if not isinstance(row, dict):
            return "text_post"
        if (row.get("news_id") or -1) not in (-1, 0, None):
            return "share_link"
        if (row.get("image_post_id") or -1) not in (-1, 0, None) or (
            row.get("image_id") or -1
        ) not in (-1, 0, None):
            return "share_image"
        return "text_post"

    def _post_topic_fingerprint(self, text_value: str):
        parsed = self._memory_parse_root_post_text(text_value or "")
        combined = (parsed.get("combined") or "").lower()
        entity_matches = re.findall(r"'([^']+)'|\"([^\"]+)\"", text_value or "")
        entities = []
        for left, right in entity_matches:
            value = re.sub(r"\s+", " ", (left or right or "").strip().lower())
            if value:
                entities.append(value)
        title = (parsed.get("title") or "").lower()
        if title.startswith("title:"):
            title = title[6:].strip()
        tokens = [
            tok
            for tok in re.findall(r"[a-z0-9]+", combined)
            if len(tok) >= 3 and tok not in _MEMORY_CALLBACK_STOPWORDS
        ]
        return {
            "title": title,
            "tokens": tokens[:24],
            "entities": entities[:4],
            "combined": combined,
        }

    def _post_topic_overlap_score(self, left: dict, right: dict) -> float:
        if not isinstance(left, dict) or not isinstance(right, dict):
            return 0.0
        if left.get("title") and right.get("title") and left.get("title") == right.get("title"):
            return 1.0
        left_entities = set(left.get("entities") or [])
        right_entities = set(right.get("entities") or [])
        if left_entities and right_entities and left_entities.intersection(right_entities):
            return 0.95
        left_tokens = set(left.get("tokens") or [])
        right_tokens = set(right.get("tokens") or [])
        if not left_tokens or not right_tokens:
            return 0.0
        return float(
            len(left_tokens.intersection(right_tokens))
            / max(1, min(len(left_tokens), len(right_tokens)))
        )

    def _post_find_recent_topic_matches(self, *, text_value: str, tid: int):
        matches = []
        fingerprint = self._post_topic_fingerprint(text_value)
        recent_roots = self._memory_get_recent_root_posts(
            tid=int(tid),
            limit=30,
            rounds_back=12,
        )
        for row in recent_roots:
            if self._post_root_origin_kind(row) != "text_post":
                continue
            recent_fp = self._post_topic_fingerprint(row.get("tweet") or "")
            score = self._post_topic_overlap_score(fingerprint, recent_fp)
            if score < 0.58:
                continue
            matches.append(
                {
                    "post_id": row.get("id"),
                    "round": row.get("round"),
                    "score": round(score, 3),
                    "title": recent_fp.get("title") or "",
                }
            )
        matches = sorted(
            matches,
            key=lambda item: (-float(item.get("score") or 0.0), -int(item.get("round") or 0)),
        )
        return fingerprint, matches[:3]

    def _memory_build_tiered_context(
        self,
        *,
        query_text: str,
        other_user_id=None,
        thread_root_id=None,
        other_username=None,
        round_id=None,
        uncertainty_score=0.0,
        allow_tier_c=True,
    ):
        if not getattr(self, "memory_enabled", False):
            return "", {
                "search_used": False,
                "degraded_mode": False,
                "tier_c_used": False,
                "retrieved_item_count": 0,
                "top_score": None,
                "fallback_used": False,
            }

        meta = {
            "search_used": False,
            "degraded_mode": False,
            "embedding_degraded": False,
            "no_ready_candidates": False,
            "tier_c_used": False,
            "retrieved_item_count": 0,
            "top_score": None,
            "fallback_used": False,
            "general_opinion_fallback_used": False,
        }

        ctx = self._memory_fetch_context(other_user_id=other_user_id, thread_root_id=thread_root_id) or {}

        digest_ctx = {}
        if isinstance(ctx, dict) and isinstance(ctx.get("community_digest"), dict):
            digest_ctx["community_digest"] = ctx.get("community_digest")
        tier_a = self._memory_format_context_for_prompt(digest_ctx, other_username=None)
        tier_a = self._memory_truncate(
            tier_a,
            int(
                getattr(self, "memory_tier_a_max_chars", DEFAULT_MEMORY_TIER_A_MAX_CHARS)
                or DEFAULT_MEMORY_TIER_A_MAX_CHARS
            ),
        )

        card_ctx = {}
        if isinstance(ctx, dict):
            if isinstance(ctx.get("social_card"), dict):
                card_ctx["social_card"] = ctx.get("social_card")
            if isinstance(ctx.get("thread_card"), dict):
                card_ctx["thread_card"] = ctx.get("thread_card")
            if isinstance(ctx.get("recent_pair_events"), list):
                card_ctx["recent_pair_events"] = ctx.get("recent_pair_events")
        card_text = self._memory_format_context_for_prompt(card_ctx, other_username=other_username)

        tier_b_chunks = []
        search_res = self._memory_search(
            query_text=query_text,
            other_user_id=other_user_id,
            thread_root_id=thread_root_id,
            types=["event", "reflection", "summary"],
            round_id=round_id,
            k=int(getattr(self, "memory_search_k", 8) or 8),
        )
        if isinstance(search_res, dict):
            meta["search_used"] = True
            rmeta = search_res.get("retrieval_meta") if isinstance(search_res.get("retrieval_meta"), dict) else {}
            meta["degraded_mode"] = bool(rmeta.get("degraded_mode", False))
            meta["embedding_degraded"] = bool(rmeta.get("embedding_degraded", False))
            meta["no_ready_candidates"] = bool(rmeta.get("no_ready_candidates", False))
            items = search_res.get("items")
            if isinstance(items, list):
                meta["retrieved_item_count"] = len(items)
                if items:
                    try:
                        meta["top_score"] = float(items[0].get("score"))
                    except Exception:
                        meta["top_score"] = None
            search_text = self._memory_format_search_brief(
                search_res,
                int(
                    getattr(self, "memory_search_max_chars", DEFAULT_MEMORY_SEARCH_MAX_CHARS)
                    or DEFAULT_MEMORY_SEARCH_MAX_CHARS
                ),
            )
            if search_text:
                tier_b_chunks.append(search_text)

        if card_text:
            tier_b_chunks.append(card_text)

        if not tier_b_chunks:
            # General opinion fallback: pull agent's own reflections + community digest
            if not meta.get("embedding_degraded"):
                try:
                    fallback_search = self._memory_search(
                        query_text=query_text,
                        other_user_id=None,
                        thread_root_id=None,
                        types=["reflection"],
                        round_id=round_id,
                        k=int(getattr(self, "memory_search_k", 8) or 8),
                    )
                    if isinstance(fallback_search, dict):
                        fb_text = self._memory_format_search_brief(
                            fallback_search,
                            int(
                                getattr(self, "memory_search_max_chars", DEFAULT_MEMORY_SEARCH_MAX_CHARS)
                                or DEFAULT_MEMORY_SEARCH_MAX_CHARS
                            ),
                        )
                        if fb_text:
                            tier_b_chunks.append(fb_text)
                            meta["general_opinion_fallback_used"] = True
                except Exception:
                    pass

            if not tier_b_chunks:
                # Phase fallback to prior context path.
                fallback = self._memory_format_context_for_prompt(ctx, other_username=other_username)
                meta["fallback_used"] = True
                return self._memory_truncate(
                    fallback,
                    int(
                        getattr(self, "memory_total_max_chars", DEFAULT_MEMORY_TOTAL_MAX_CHARS)
                        or DEFAULT_MEMORY_TOTAL_MAX_CHARS
                    ),
                ), meta

        tier_b = "\n\n".join([x for x in tier_b_chunks if isinstance(x, str) and x.strip()])
        tier_b = self._memory_truncate(
            tier_b,
            int(
                getattr(self, "memory_tier_b_max_chars", DEFAULT_MEMORY_TIER_B_MAX_CHARS)
                or DEFAULT_MEMORY_TIER_B_MAX_CHARS
            ),
        )

        tier_c = ""
        threshold = float(getattr(self, "memory_tier_c_uncertainty_threshold", 0.45) or 0.45)
        trigger_tier_c = float(uncertainty_score or 0.0) >= threshold
        if not trigger_tier_c:
            try:
                top_score = float(meta.get("top_score")) if meta.get("top_score") is not None else None
            except Exception:
                top_score = None
            if top_score is None or top_score < threshold:
                trigger_tier_c = True

        if allow_tier_c and trigger_tier_c:
            expanded = self._memory_search(
                query_text=query_text,
                other_user_id=None,
                thread_root_id=None,
                types=["event", "reflection", "summary"],
                round_id=round_id,
                k=int(max(12, int(getattr(self, "memory_search_k", 8) or 8) * 2)),
                time_window_rounds=int(
                    max(40, int(getattr(self, "memory_search_time_window_rounds", 40) or 40) * 2)
                ),
            )
            if isinstance(expanded, dict):
                tier_c = self._memory_format_search_brief(
                    expanded,
                    int(
                        getattr(self, "memory_tier_c_max_chars", DEFAULT_MEMORY_TIER_C_MAX_CHARS)
                        or DEFAULT_MEMORY_TIER_C_MAX_CHARS
                    ),
                )
                if tier_c:
                    meta["tier_c_used"] = True

        blocks = []
        if tier_a:
            blocks.append("[MEMORY TIER A]\n" + tier_a)
        if tier_b:
            blocks.append("[MEMORY TIER B]\n" + tier_b)
        if tier_c:
            blocks.append("[MEMORY TIER C]\n" + tier_c)
        merged = "\n\n".join(blocks).strip()
        merged = self._memory_truncate(
            merged,
            int(
                getattr(self, "memory_total_max_chars", DEFAULT_MEMORY_TOTAL_MAX_CHARS)
                or DEFAULT_MEMORY_TOTAL_MAX_CHARS
            ),
        )
        return merged, meta

    def _memory_build_conversation_cues(
        self,
        *,
        memory_text: str = "",
        memory_meta: dict = None,
        target_username: str = None,
        mode: str = "comment",
    ):
        cues = {
            "scope": "none",
            "callback_hint": "",
            "continuity_hint": "",
            "argument_hint": "",
            "tone_hint": "",
            "anecdote_hint": "",
            "should_callback": False,
            "retrieved_item_count": 0,
            "top_score": None,
            "degraded_mode": False,
            "mode": str(mode or "comment"),
        }
        if not getattr(self, "memory_nuance_enabled", True):
            return cues

        meta = memory_meta if isinstance(memory_meta, dict) else {}
        try:
            retrieved = int(meta.get("retrieved_item_count", 0) or 0)
        except Exception:
            retrieved = 0
        cues["retrieved_item_count"] = max(0, retrieved)

        top_score = None
        try:
            val = meta.get("top_score")
            if val is not None:
                top_score = float(val)
        except Exception:
            top_score = None
        cues["top_score"] = top_score

        degraded_mode = bool(meta.get("degraded_mode", False))
        embedding_degraded = bool(meta.get("embedding_degraded", False))
        no_ready_candidates = bool(meta.get("no_ready_candidates", False))
        continuity_text = str(meta.get("continuity_text") or "").strip()
        cross_thread_callback_candidate = bool(meta.get("cross_thread_callback_candidate"))
        cues["degraded_mode"] = degraded_mode
        cues["continuity_hint"] = continuity_text

        min_score = float(getattr(self, "memory_nuance_min_score", 0.35) or 0.35)
        callback_probability = float(
            getattr(self, "memory_nuance_callback_probability", 0.55) or 0.55
        )
        subtle_mode = self._memory_use_subtle_prompt_mode()
        if subtle_mode:
            callback_probability = min(callback_probability, 0.12)
            min_score = max(min_score, float(
                getattr(self, "memory_cross_thread_callback_min_score", 0.80) or 0.80
            ))

        has_partial_signal = retrieved > 0 and not degraded_mode
        has_strong_signal = (
            has_partial_signal and top_score is not None and float(top_score) >= float(min_score)
        )

        if embedding_degraded and retrieved <= 0:
            cues["scope"] = "degraded"
        elif no_ready_candidates and retrieved <= 0 and not embedding_degraded:
            cues["scope"] = "cold_start"
        elif has_strong_signal:
            cues["scope"] = "strong"
        elif has_partial_signal:
            cues["scope"] = "partial"

        target = "this user"
        if isinstance(target_username, str) and target_username.strip():
            target = "@" + target_username.strip().lstrip("@")

        def _extract_metric(name: str, default: float = 0.0):
            if not isinstance(memory_text, str) or not memory_text:
                return float(default)
            m = re.search(rf"{name}\s*=\s*([-+]?\d+(?:\.\d+)?)", memory_text, flags=re.IGNORECASE)
            if not m:
                return float(default)
            try:
                return float(m.group(1))
            except Exception:
                return float(default)

        affinity = _extract_metric("affinity", 0.0)
        conflict = _extract_metric("conflict", 0.0)
        humor = _extract_metric("humor", 0.0)
        trust = _extract_metric("trust", 0.0)

        should_callback = False
        if has_strong_signal and cross_thread_callback_candidate:
            should_callback = random.random() <= callback_probability
        elif has_partial_signal and not subtle_mode:
            should_callback = random.random() <= min(0.25, callback_probability * 0.4)
        cues["should_callback"] = bool(should_callback)

        if embedding_degraded and retrieved <= 0:
            callback_hint = (
                "Memory retrieval is degraded right now. Do not claim specific past interactions."
            )
            cues["callback_skip_reason"] = "degraded"
        elif cues["scope"] == "cold_start":
            callback_hint = (
                "You have limited interaction history. Focus on the current content "
                "and express your genuine opinions based on your personality and values."
            )
            cues["callback_skip_reason"] = "cold_start"
        elif subtle_mode and cross_thread_callback_candidate:
            callback_hint = (
                f"If clearly relevant, you may use one brief continuity cue with {target}. "
                "Do not force it or import unrelated stories."
            )
        elif should_callback:
            callback_hint = (
                f"If natural, reference one prior exchange with {target} in one short clause."
            )
        elif has_partial_signal:
            if subtle_mode:
                callback_hint = (
                    "Use continuity only if it sharpens the point. Stay anchored to this thread."
                )
            else:
                callback_hint = (
                    "Use only soft continuity (e.g., recurring disagreement) and avoid specific claims."
                )
            cues["callback_skip_reason"] = "below_min_score"
        else:
            callback_hint = "No memory callback needed; focus on the current message."
            cues["callback_skip_reason"] = "no_signal"

        if conflict - affinity >= 0.2:
            argument_hint = (
                "You often clash here. It is fine to continue the argument, but stay concrete."
            )
        elif affinity - conflict >= 0.2:
            argument_hint = (
                "You often align here. Start from common ground, then add a fresh point."
            )
        else:
            argument_hint = "Balance agreement and pushback based on the latest claim."

        if degraded_mode:
            tone_hint = "Keep tone grounded in the current thread; avoid certainty from memory."
        elif trust <= -0.2 or conflict >= 0.55:
            tone_hint = "Keep a skeptical tone; ask for specifics or challenge weak claims."
        elif humor >= 0.25 and trust >= 0.0:
            tone_hint = "Light banter is okay if it helps the point land."
        else:
            tone_hint = "Keep it conversational and direct."

        if subtle_mode:
            anecdote_hint = "Do not import unrelated anecdotes from other threads."
        elif should_callback and (trust >= 0.15 or humor >= 0.25):
            anecdote_hint = (
                "A tiny shared-memory anecdote is fine, but only if it is clearly supported."
            )
        elif conflict >= 0.5:
            anecdote_hint = "Prefer argument continuity over personal anecdotes."
        else:
            anecdote_hint = "Use anecdotes sparingly; only when they clarify your point."

        cues["callback_hint"] = callback_hint
        cues["argument_hint"] = argument_hint
        cues["tone_hint"] = tone_hint
        cues["anecdote_hint"] = anecdote_hint
        return cues

    def _memory_format_conversation_cues(self, cues: dict):
        if not isinstance(cues, dict):
            return ""
        if not getattr(self, "memory_nuance_enabled", True):
            return ""

        callback_hint = (cues.get("callback_hint") or "").strip()
        continuity_hint = (cues.get("continuity_hint") or "").strip()
        argument_hint = (cues.get("argument_hint") or "").strip()
        tone_hint = (cues.get("tone_hint") or "").strip()
        anecdote_hint = (cues.get("anecdote_hint") or "").strip()
        scope = (cues.get("scope") or "none").strip()
        try:
            retrieved_item_count = int(cues.get("retrieved_item_count", 0) or 0)
        except Exception:
            retrieved_item_count = 0
        degraded_mode = bool(cues.get("degraded_mode", False))

        # Keep prompts lean when memory signal is absent.
        if scope == "none" and retrieved_item_count <= 0 and not degraded_mode:
            return ""

        lines = ["[MEMORY CONVERSATION CUES]"]
        lines.append(f"- scope: {scope}")
        if callback_hint:
            lines.append(f"- callback: {callback_hint}")
        if continuity_hint:
            lines.append(f"- continuity: {continuity_hint}")
        if argument_hint:
            lines.append(f"- argument: {argument_hint}")
        if tone_hint:
            lines.append(f"- tone: {tone_hint}")
        if anecdote_hint:
            lines.append(f"- anecdote: {anecdote_hint}")
        lines.append("- Never invent people, comments, or events not present in memory context.")

        max_chars = int(getattr(self, "memory_nuance_cues_max_chars", 900) or 900)
        return self._memory_truncate("\n".join(lines), max_chars)

    def _memory_plan_reply_strategy(
        self,
        *,
        mode: str,
        mention_author: str,
        mention_text: str,
        thread_context: str,
        memory_cues_block: str,
        interests=None,
        proactive_affect_block: str = "",
    ):
        if not getattr(self, "memory_nuance_enabled", True):
            return ""
        if not getattr(self, "memory_nuance_planner_enabled", True):
            return ""
        if not isinstance(self.prompts, dict):
            return ""

        planner_prompt = self.prompts.get("handler_memory_reply_planner")
        if not isinstance(planner_prompt, str) or not planner_prompt.strip():
            return ""

        interests = interests if isinstance(interests, list) else []
        interests_str = ", ".join([str(x) for x in interests if str(x).strip()]) or "none"

        planner_cfg = self.__get_fresh_llm_config()
        planner_cfg["temperature"] = float(
            getattr(self, "memory_nuance_planner_temperature", 0.25) or 0.25
        )
        planner_cfg["max_tokens"] = int(
            getattr(self, "memory_nuance_planner_max_tokens", 120) or 120
        )

        u1 = AssistantAgent(
            name=f"{self.name}",
            llm_config=planner_cfg,
            system_message=self.__effify(
                self.prompts.get(
                    "agent_roleplay_comments_share",
                    self.prompts.get("agent_roleplay_simple", ""),
                ),
                interests=interests,
            ),
            max_consecutive_auto_reply=1,
        )

        u2 = AssistantAgent(
            name="MemoryPlanner",
            llm_config=planner_cfg,
            system_message=self.__effify(
                self.prompts.get(
                    "handler_instructions_simple",
                    "You are the Handler that specifies the actions to be taken.",
                )
            ),
            max_consecutive_auto_reply=0,
        )

        raw = ""
        try:
            u2.initiate_chat(
                u1,
                message=self.__effify(
                    planner_prompt,
                    mode=str(mode or "comment"),
                    mention_author=mention_author or "",
                    mention_text=mention_text or "",
                    thread_context=self._memory_truncate(thread_context or "", 1800),
                    memory_cues_block=self._memory_truncate(memory_cues_block or "", 700),
                    interests_str=interests_str,
                    proactive_affect_block=proactive_affect_block or "",
                ),
                silent=True,
                max_turns=1,
            )
            raw = u1.chat_messages[u2][-1]["content"]
        except Exception:
            raw = ""
        finally:
            try:
                u1.reset()
                u2.reset()
            except Exception:
                pass

        if not isinstance(raw, str) or not raw.strip():
            return ""

        parsed = self._memory_extract_json(raw)
        if not isinstance(parsed, dict):
            plain = re.sub(r"\s+", " ", raw).strip()
            if not plain:
                return ""
            return self._memory_truncate(
                "[MEMORY REPLY PLAN]\n- " + plain,
                int(getattr(self, "memory_nuance_cues_max_chars", 900) or 900),
            )

        lines = ["[MEMORY REPLY PLAN]"]
        opening_move = str(parsed.get("opening_move") or "").strip()
        callback_line = str(parsed.get("callback_line") or "").strip()
        stance = str(parsed.get("stance") or "").strip()
        tone = str(parsed.get("tone") or "").strip()
        avoid = str(parsed.get("avoid") or "").strip()
        if opening_move:
            lines.append(f"- opening: {opening_move}")
        if callback_line:
            lines.append(f"- callback line: {callback_line}")
        if stance:
            lines.append(f"- stance: {stance}")
        if tone:
            lines.append(f"- tone: {tone}")
        if avoid:
            lines.append(f"- avoid: {avoid}")

        if len(lines) == 1:
            return ""
        return self._memory_truncate(
            "\n".join(lines),
            int(getattr(self, "memory_nuance_cues_max_chars", 900) or 900),
        )

    def _extract_last_thread_message(self, thread_text: str):
        if not isinstance(thread_text, str) or not thread_text.strip():
            return "", ""
        lines = [ln.strip() for ln in thread_text.splitlines() if str(ln).strip()]
        for line in reversed(lines):
            parsed = re.match(r"^@?([^\s]+)\s*-\s*(.+)$", line)
            if parsed:
                return parsed.group(2).strip(), parsed.group(1).strip().lstrip("@")
        return lines[-1].strip(), ""

    def _memory_detect_prior_opinion_match(
        self,
        *,
        incoming_text: str,
        other_user_id=None,
        thread_root_id=None,
        round_id=None,
    ):
        if not getattr(self, "memory_enabled", False):
            return False, None
        incoming = str(incoming_text or "").strip()
        if not incoming:
            return False, None

        res = self._memory_search(
            query_text=self._memory_build_query_text(
                "my previous opinion or claim on this argument",
                incoming,
            ),
            other_user_id=other_user_id,
            thread_root_id=thread_root_id,
            types=["event", "reflection", "summary"],
            round_id=round_id,
            k=int(getattr(self, "memory_high_affect_search_k", 12) or 12),
        )
        if not isinstance(res, dict):
            return False, None
        items = res.get("items")
        if not isinstance(items, list) or not items:
            return False, None

        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                score = float(item.get("score") or 0.0)
            except Exception:
                score = 0.0
            if score < 0.35:
                continue

            actor_id = item.get("actor_user_id")
            try:
                actor_match = int(actor_id) == int(self.user_id)
            except Exception:
                actor_match = False

            txt = str(item.get("text_humanized") or item.get("text") or "").strip().lower()
            if actor_match or re.search(r"\b(i said|my take|my view|my point|i argued)\b", txt):
                return True, item
        return False, None

    def _memory_llm_high_affect_classifier(
        self,
        *,
        incoming_text: str,
        thread_context: str,
        target_username: str = "",
    ):
        if not getattr(self, "memory_high_affect_llm_fallback", True):
            return None
        if not isinstance(self.prompts, dict):
            return None

        prompt = self.prompts.get("handler_high_affect_classifier")
        if not isinstance(prompt, str) or not prompt.strip():
            return None

        cfg = self.__get_fresh_llm_config()
        cfg["temperature"] = 0.1
        cfg["max_tokens"] = 200

        u1 = AssistantAgent(
            name=f"{self.name}",
            llm_config=cfg,
            system_message=self.__effify(
                self.prompts.get(
                    "agent_roleplay_comments_share",
                    self.prompts.get("agent_roleplay_simple", ""),
                ),
                interests=[],
            ),
            max_consecutive_auto_reply=1,
        )
        u2 = AssistantAgent(
            name="HighAffectClassifier",
            llm_config=cfg,
            system_message=self.__effify(
                self.prompts.get(
                    "handler_instructions_simple",
                    "You are the Handler that specifies the actions to be taken.",
                )
            ),
            max_consecutive_auto_reply=0,
        )

        raw = ""
        try:
            u2.initiate_chat(
                u1,
                message=self.__effify(
                    prompt,
                    incoming_text=(incoming_text or "").strip(),
                    thread_context=self._memory_truncate(thread_context or "", 1800),
                    target_username=str(target_username or "").strip().lstrip("@"),
                ),
                silent=True,
                max_turns=1,
            )
            raw = u1.chat_messages[u2][-1]["content"]
        except Exception:
            raw = ""
        finally:
            try:
                u1.reset()
                u2.reset()
            except Exception:
                pass

        parsed = self._memory_extract_json(raw)
        if not isinstance(parsed, dict):
            return None

        return {
            "criticism_or_challenge": bool(parsed.get("criticism_or_challenge", False)),
            "conflict_or_argument": bool(parsed.get("conflict_or_argument", False)),
            "incoming_anecdote": bool(parsed.get("incoming_anecdote", False)),
            "defending_prior_opinion": bool(parsed.get("defending_prior_opinion", False)),
            "confidence": float(parsed.get("confidence", 0.0) or 0.0),
        }

    def _detect_high_affect_signal(
        self,
        *,
        incoming_text: str,
        thread_context: str,
        other_user_id=None,
        thread_root_id=None,
        round_id=None,
        target_username: str = "",
    ):
        text = str(incoming_text or "").strip()
        norm = re.sub(r"\s+", " ", text).strip().lower()
        if not norm:
            return {
                "is_high_affect": False,
                "confidence": 0.0,
                "source": "rules",
                "triggers": {
                    "criticism_or_challenge": False,
                    "conflict_or_argument": False,
                    "incoming_anecdote": False,
                    "defending_prior_opinion": False,
                },
                "used_llm_fallback": False,
                "prior_match": False,
            }

        challenge_terms = (
            "wrong",
            "prove",
            "source",
            "evidence",
            "citation",
            "explain",
            "how do you",
            "you are clueless",
            "nonsense",
            "cope",
            "l take",
            "bad take",
        )
        conflict_terms = (
            "i disagree",
            "nope",
            "bullshit",
            "ridiculous",
            "delusional",
            "you are",
            "wtf",
            "shut up",
            "terrible take",
            "this is dumb",
        )
        anecdote_terms = (
            "when i",
            "i remember",
            "my experience",
            "this happened to me",
            "back when i",
            "once i",
            "i had",
            "i was",
            "my friend and i",
        )
        defense_terms = (
            "you said",
            "last time",
            "again",
            "changed your mind",
            "you always",
            "same claim",
            "as before",
            "you keep saying",
        )

        has_you = bool(re.search(r"\b(you|your|u)\b", norm))
        has_question = "?" in text
        criticism_or_challenge = has_you and (
            has_question or any(term in norm for term in challenge_terms)
        )
        conflict_or_argument = any(term in norm for term in conflict_terms) or (
            bool(re.search(r"\b(disagree|nah|no way|not true)\b", norm)) and has_you
        )
        incoming_anecdote = any(term in norm for term in anecdote_terms) and bool(
            re.search(r"\b(i|my|me)\b", norm)
        )
        defense_lexical = any(term in norm for term in defense_terms)

        prior_match = False
        prior_item = None
        if defense_lexical or criticism_or_challenge or conflict_or_argument:
            try:
                prior_match, prior_item = self._memory_detect_prior_opinion_match(
                    incoming_text=text,
                    other_user_id=other_user_id,
                    thread_root_id=thread_root_id,
                    round_id=round_id,
                )
            except Exception:
                prior_match, prior_item = (False, None)
        defending_prior_opinion = bool(defense_lexical and prior_match)

        score = 0.0
        score += 0.36 if criticism_or_challenge else 0.0
        score += 0.36 if conflict_or_argument else 0.0
        score += 0.22 if incoming_anecdote else 0.0
        score += 0.30 if defending_prior_opinion else 0.0
        score = min(1.0, score)

        low = float(getattr(self, "memory_high_affect_uncertain_low", 0.35) or 0.35)
        high = float(getattr(self, "memory_high_affect_uncertain_high", 0.70) or 0.70)
        use_llm = bool(low <= score <= high)

        used_llm_fallback = False
        llm_signals = None
        if use_llm and getattr(self, "memory_high_affect_llm_fallback", True):
            llm_signals = self._memory_llm_high_affect_classifier(
                incoming_text=text,
                thread_context=thread_context,
                target_username=target_username,
            )
            used_llm_fallback = isinstance(llm_signals, dict)

        if isinstance(llm_signals, dict):
            criticism_or_challenge = bool(
                criticism_or_challenge or llm_signals.get("criticism_or_challenge", False)
            )
            conflict_or_argument = bool(
                conflict_or_argument or llm_signals.get("conflict_or_argument", False)
            )
            incoming_anecdote = bool(
                incoming_anecdote or llm_signals.get("incoming_anecdote", False)
            )
            defending_prior_opinion = bool(
                defending_prior_opinion or llm_signals.get("defending_prior_opinion", False)
            )
            try:
                score = max(score, float(llm_signals.get("confidence", 0.0) or 0.0))
            except Exception:
                pass

        triggers = {
            "criticism_or_challenge": bool(criticism_or_challenge),
            "conflict_or_argument": bool(conflict_or_argument),
            "incoming_anecdote": bool(incoming_anecdote),
            "defending_prior_opinion": bool(defending_prior_opinion),
        }
        any_trigger = any(triggers.values())
        threshold = float(getattr(self, "memory_high_affect_rule_threshold", 0.55) or 0.55)

        return {
            "is_high_affect": bool(any_trigger or score >= threshold),
            "confidence": float(max(0.0, min(1.0, score))),
            "source": "hybrid" if used_llm_fallback else "rules",
            "triggers": triggers,
            "used_llm_fallback": bool(used_llm_fallback),
            "prior_match": bool(prior_match),
            "prior_match_item_id": (prior_item or {}).get("item_id")
            if isinstance(prior_item, dict)
            else None,
        }

    # ------------------------------------------------------------------
    # Proactive High-Affect Initiation Engine (Sections 1A–1E)
    # ------------------------------------------------------------------

    @staticmethod
    def _trait_to_score(trait_text: str) -> float:
        """Map Big Five trait description to numeric 0.0–1.0 score."""
        mapping = {
            "very low": 0.1,
            "low": 0.3,
            "average": 0.5,
            "high": 0.7,
            "very high": 0.9,
        }
        if not isinstance(trait_text, str):
            return 0.5
        key = trait_text.strip().lower()
        return mapping.get(key, 0.5)

    def _build_persona_affect_profile(self) -> dict:
        """Derive static propensity dict from Big Five traits + toxicity. Cached."""
        if self._persona_affect_profile_cache is not None:
            return self._persona_affect_profile_cache

        ag_score = self._trait_to_score(getattr(self, "ag", "average"))
        ne_score = self._trait_to_score(getattr(self, "ne", "average"))
        oe_score = self._trait_to_score(getattr(self, "oe", "average"))
        co_score = self._trait_to_score(getattr(self, "co", "average"))
        ex_score = self._trait_to_score(getattr(self, "ex", "average"))
        tox = float(getattr(self, "toxicity_val", 0) or 0)
        tox = min(tox, 1.0)

        profile = {
            "disagree_propensity": (1.0 - ag_score) * 0.4 + ne_score * 0.3 + tox * 0.3,
            "probe_propensity": oe_score * 0.5 + (1.0 - ag_score) * 0.3 + ex_score * 0.2,
            "troll_propensity": tox * 0.5 + ne_score * 0.3 + (1.0 - co_score) * 0.2,
            "challenge_propensity": (1.0 - ag_score) * 0.4 + oe_score * 0.3 + (1.0 - co_score) * 0.3,
        }
        self._persona_affect_profile_cache = profile
        return profile

    def _detect_dominant_view_pressure(self, thread_context: str) -> dict:
        """Detect whether the thread exhibits dominant view consensus using lexical analysis + LLM fallback."""
        result = {
            "dominant_view_detected": False,
            "dominant_view_summary": "",
            "confidence": 0.0,
        }
        if not isinstance(thread_context, str) or not thread_context.strip():
            return result

        # Lexical pass: count agreement vs disagreement among replies
        lines = [l.strip() for l in thread_context.split("\n") if l.strip()]
        if len(lines) < 3:
            return result

        agreement_phrases = (
            "i agree", "exactly", "this", "100%", "same", "well said",
            "facts", "nailed it", "based", "true", "right", "yes",
            "absolutely", "for sure", "totally", "correct", "spot on",
            "underrated", "preach",
        )
        disagreement_phrases = (
            "i disagree", "wrong", "nope", "no way", "cope", "hard pass",
            "bullshit", "ridiculous", "doubt", "not true", "nah",
            "terrible take", "bad take", "l take",
        )

        agree_count = 0
        disagree_count = 0
        # Skip first line (likely OP)
        for line in lines[1:]:
            norm = line.lower()
            has_agree = any(p in norm for p in agreement_phrases)
            has_disagree = any(p in norm for p in disagreement_phrases)
            if has_agree and not has_disagree:
                agree_count += 1
            elif has_disagree:
                disagree_count += 1

        total_classified = agree_count + disagree_count
        reply_count = len(lines) - 1

        if reply_count >= 3 and total_classified > 0:
            agreement_ratio = agree_count / total_classified
            threshold = float(
                getattr(self, "proactive_affect_contrarian_threshold", 0.75) or 0.75
            )
            if agreement_ratio >= threshold:
                result["dominant_view_detected"] = True
                result["confidence"] = min(1.0, 0.5 + agreement_ratio * 0.3)
                result["dominant_view_summary"] = (
                    f"{agree_count}/{total_classified} classified replies show agreement"
                )
                return result

        # LLM fallback if lexical result is uncertain
        if (
            reply_count >= 3
            and total_classified >= 2
            and 0.4 <= (agree_count / max(1, total_classified)) <= 0.85
            and getattr(self, "proactive_affect_enabled", True)
            and isinstance(getattr(self, "prompts", None), dict)
            and "handler_proactive_high_affect_classifier" in self.prompts
        ):
            try:
                profile = self._build_persona_affect_profile()
                personality_str = (
                    f"disagree_propensity={profile['disagree_propensity']:.2f}, "
                    f"probe_propensity={profile['probe_propensity']:.2f}, "
                    f"challenge_propensity={profile['challenge_propensity']:.2f}, "
                    f"troll_propensity={profile['troll_propensity']:.2f}"
                )
                classifier_prompt = self.__effify(
                    self.prompts["handler_proactive_high_affect_classifier"],
                    agent_personality=personality_str,
                    thread_context=thread_context[-2000:],
                )
                cfg = self.llm_config.copy()
                cfg["temperature"] = 0.15
                u1 = AssistantAgent(
                    name="classifier",
                    llm_config=cfg,
                    system_message="You classify thread dynamics. Output only JSON.",
                    max_consecutive_auto_reply=0,
                )
                u2 = AssistantAgent(
                    name="driver",
                    llm_config=cfg,
                    system_message="Driver.",
                    max_consecutive_auto_reply=1,
                )
                u1.initiate_chat(u2, message=classifier_prompt, silent=True, max_turns=1)
                raw = u2.chat_messages[u1][-1]["content"].strip()
                u1.reset()
                u2.reset()
                parsed = self._memory_extract_json(raw)
                if isinstance(parsed, dict):
                    result["dominant_view_detected"] = bool(parsed.get("dominant_view_detected", False))
                    result["dominant_view_summary"] = str(
                        parsed.get("dominant_view_summary", "")
                    )[:120]
                    result["confidence"] = float(parsed.get("confidence", 0.0) or 0.0)
            except Exception:
                pass

        return result

    def _detect_proactive_high_affect_signal(
        self,
        *,
        thread_context: str,
        round_id=None,
    ) -> dict:
        """Proactive counterpart to _detect_high_affect_signal. Fires based on personality, not incoming aggression."""
        null_result = {
            "is_proactive_high_affect": False,
            "mode": None,
            "confidence": 0.0,
            "reasons": [],
            "dominant_view_detected": False,
        }
        if not getattr(self, "proactive_affect_enabled", True):
            return null_result

        cap = int(getattr(self, "proactive_affect_cap_per_round", 2) or 2)
        if int(getattr(self, "_proactive_affect_this_round", 0) or 0) >= cap:
            return null_result

        profile = self._build_persona_affect_profile()
        dominant = self._detect_dominant_view_pressure(thread_context)
        scale = float(getattr(self, "proactive_affect_probability_scale", 0.7) or 0.7)
        tox_val = float(getattr(self, "toxicity_val", 0) or 0)

        reasons = []
        probability = 0.0

        if dominant.get("dominant_view_detected") and profile["disagree_propensity"] > 0.4:
            probability += 0.5 * scale
            reasons.append("dominant_view+high_disagree_propensity")

        if profile["challenge_propensity"] > 0.6:
            probability += 0.3 * scale
            reasons.append("high_challenge_propensity")

        if profile["troll_propensity"] > 0.5:
            probability += 0.15 * scale
            reasons.append("troll_propensity")

        if profile["probe_propensity"] > 0.6 and not dominant.get("dominant_view_detected"):
            probability += 0.2 * scale
            reasons.append("high_probe_propensity")

        probability = min(1.0, probability)

        if probability < 0.15 or random.random() > probability:
            return null_result

        # Mode selection based on propensity scores and toxicity level
        mode = None
        if tox_val == 0:
            # Zero toxicity: only probe or challenge
            if profile["probe_propensity"] >= profile["challenge_propensity"]:
                mode = "probe"
            else:
                mode = "challenge"
        elif tox_val <= 0.3:
            # Low toxicity: probe, challenge, or troll_soft
            candidates = [
                ("probe", profile["probe_propensity"]),
                ("challenge", profile["challenge_propensity"]),
                ("troll_soft", profile["troll_propensity"] * 0.7),
            ]
            mode = max(candidates, key=lambda x: x[1])[0]
        else:
            # Higher toxicity: all modes including troll_hard
            candidates = [
                ("probe", profile["probe_propensity"]),
                ("challenge", profile["challenge_propensity"]),
                ("troll_soft", profile["troll_propensity"] * 0.6),
                ("troll_hard", profile["troll_propensity"]),
            ]
            mode = max(candidates, key=lambda x: x[1])[0]

        return {
            "is_proactive_high_affect": True,
            "mode": mode,
            "confidence": float(min(1.0, probability)),
            "reasons": reasons,
            "dominant_view_detected": bool(dominant.get("dominant_view_detected")),
        }

    @staticmethod
    def _build_effective_affect_signal(reactive: dict, proactive: dict) -> dict:
        """Merge reactive + proactive affect signals into one unified signal."""
        reactive = reactive if isinstance(reactive, dict) else {}
        proactive = proactive if isinstance(proactive, dict) else {}
        is_reactive = bool(reactive.get("is_high_affect"))
        is_proactive = bool(proactive.get("is_proactive_high_affect"))
        mode = proactive.get("mode") if is_proactive else ("reactive" if is_reactive else None)
        return {
            "is_high_affect": is_reactive or is_proactive,
            "reactive": reactive,
            "proactive": proactive,
            "mode": mode,
            "dominant_view_detected": bool(proactive.get("dominant_view_detected")),
        }

    def _format_proactive_affect_block(self, proactive_signal: dict) -> str:
        """Format proactive affect context for prompt injection."""
        if not isinstance(proactive_signal, dict):
            return ""
        if not proactive_signal.get("is_proactive_high_affect"):
            return ""
        mode = proactive_signal.get("mode") or "none"
        reasons = proactive_signal.get("reasons") or []
        dominant = proactive_signal.get("dominant_view_detected", False)
        lines = ["[PROACTIVE AFFECT CONTEXT]"]
        lines.append(f"proactive_mode={mode}")
        lines.append(f"dominant_view_detected={'true' if dominant else 'false'}")
        if reasons:
            lines.append(f"reasons={', '.join(reasons)}")
        if mode == "challenge":
            lines.append(
                "Your personality inclines you to push back. "
                "Consider a counterargument or devil's advocate position."
            )
        elif mode == "probe":
            lines.append(
                "Your personality inclines you to ask probing questions. "
                "Challenge assumptions or ask for evidence."
            )
        elif mode == "troll_soft":
            lines.append(
                "Your personality inclines you toward witty provocation. "
                "Use sarcasm or devil's advocate framing — be clever, not cruel."
            )
        elif mode == "troll_hard":
            lines.append(
                "Your personality inclines you toward provocative trolling. "
                "Stir the pot aggressively."
            )
        return "\n".join(lines)

    @property
    def _is_memory_cold_start(self):
        """True during the first N interactions (cold-start window)."""
        return int(getattr(self, "_memory_global_interaction_index", 0) or 0) <= int(
            getattr(self, "memory_cold_start_window", 5) or 5
        )

    def _memory_collect_high_affect_recall(
        self,
        *,
        incoming_text: str,
        thread_context: str,
        other_user_id=None,
        thread_root_id=None,
        round_id=None,
        target_username: str = "",
    ):
        pack = {
            "items": [],
            "counts_by_bucket": {},
            "has_usable_memories": False,
            "prompt_block": "",
        }
        if not getattr(self, "memory_enabled", False):
            return pack
        if not getattr(self, "memory_semantic_enabled", True):
            return pack

        max_items = int(getattr(self, "memory_high_affect_max_items", 6) or 6)
        max_chars = int(getattr(self, "memory_high_affect_max_chars", 900) or 900)
        k = int(getattr(self, "memory_high_affect_search_k", 12) or 12)

        buckets = [
            {
                "name": "interaction",
                "cap": 2,
                "search_kwargs": {
                    "query_text": self._memory_build_query_text(
                        "recent back and forth argument with this user",
                        incoming_text,
                        thread_context,
                    ),
                    "other_user_id": other_user_id,
                    "thread_root_id": thread_root_id,
                    "types": ["event", "summary"],
                    "round_id": round_id,
                    "k": k,
                },
            },
            {
                "name": "opinion",
                "cap": 2,
                "search_kwargs": {
                    "query_text": self._memory_build_query_text(
                        "my previously stated opinion or claim",
                        incoming_text,
                    ),
                    "thread_root_id": thread_root_id,
                    "types": ["event", "reflection", "summary"],
                    "round_id": round_id,
                    "k": k,
                },
            },
            {
                "name": "personal_experience",
                "cap": 1,
                "search_kwargs": {
                    "query_text": self._memory_build_query_text(
                        "my personal experience anecdote related to this topic",
                        incoming_text,
                    ),
                    "types": ["event", "reflection"],
                    "round_id": round_id,
                    "k": k,
                },
            },
            {
                "name": "relationship",
                "cap": 1,
                "search_kwargs": {
                    "query_text": self._memory_build_query_text(
                        "relationship history trust conflict humor with this user",
                        target_username or "",
                    ),
                    "other_user_id": other_user_id,
                    "types": ["summary", "reflection"],
                    "round_id": round_id,
                    "k": k,
                },
            },
        ]

        seen_ids = set()
        ordered_items = []
        counts = {}
        for bucket in buckets:
            bucket_name = bucket["name"]
            cap = int(bucket.get("cap", 1) or 1)
            counts[bucket_name] = 0
            try:
                res = self._memory_search(**bucket["search_kwargs"])
            except Exception:
                res = None
            if not isinstance(res, dict):
                continue
            items = res.get("items")
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                if counts[bucket_name] >= cap:
                    break
                if len(ordered_items) >= max_items:
                    break
                item_id = item.get("item_id")
                if item_id is None:
                    continue
                try:
                    item_id = int(item_id)
                except Exception:
                    continue
                if item_id in seen_ids:
                    continue
                seen_ids.add(item_id)
                txt = str(item.get("text_humanized") or item.get("text") or "").strip()
                if not txt:
                    continue
                if len(txt) > 180:
                    txt = txt[:177].rstrip() + "..."
                try:
                    score = float(item.get("score") or 0.0)
                except Exception:
                    score = 0.0
                ordered_items.append(
                    {
                        "item_id": item_id,
                        "bucket": bucket_name,
                        "score": score,
                        "round_id": item.get("round_id"),
                        "text": txt,
                    }
                )
                counts[bucket_name] += 1

        if not ordered_items:
            # General opinion fallback: pull reflections with no pair/thread filter
            try:
                fallback_res = self._memory_search(
                    query_text=self._memory_build_query_text(
                        "my general opinions and reflections on this topic",
                        incoming_text,
                    ),
                    other_user_id=None,
                    thread_root_id=None,
                    types=["reflection"],
                    round_id=round_id,
                    k=k,
                )
                if isinstance(fallback_res, dict):
                    fb_items = fallback_res.get("items") or []
                    for item in (fb_items if isinstance(fb_items, list) else []):
                        if not isinstance(item, dict):
                            continue
                        if len(ordered_items) >= max_items:
                            break
                        item_id = item.get("item_id")
                        if item_id is None:
                            continue
                        try:
                            item_id = int(item_id)
                        except Exception:
                            continue
                        if item_id in seen_ids:
                            continue
                        seen_ids.add(item_id)
                        txt = str(item.get("text_humanized") or item.get("text") or "").strip()
                        if not txt:
                            continue
                        if len(txt) > 180:
                            txt = txt[:177].rstrip() + "..."
                        try:
                            score = float(item.get("score") or 0.0)
                        except Exception:
                            score = 0.0
                        ordered_items.append({
                            "item_id": item_id,
                            "bucket": "general_opinion_fallback",
                            "score": score,
                            "round_id": item.get("round_id"),
                            "text": txt,
                        })
                    if ordered_items:
                        pack["general_opinion_fallback_used"] = True
            except Exception:
                pass

        if not ordered_items:
            return pack

        lines = ["[RECALLED MEMORIES]"]
        for idx, item in enumerate(ordered_items[:max_items], start=1):
            rid = item.get("round_id")
            score = item.get("score")
            try:
                score_txt = f"{float(score):.2f}"
            except Exception:
                score_txt = "?"
            lines.append(
                f"- [M{idx}] ({item.get('bucket')}, r{rid}, s={score_txt}) {item.get('text')}"
            )
        lines.append("- Reference memory naturally. Never invent events not listed above.")

        pack["items"] = ordered_items[:max_items]
        pack["counts_by_bucket"] = counts
        pack["has_usable_memories"] = bool(pack["items"])
        pack["prompt_block"] = self._memory_truncate("\n".join(lines), max_chars)
        return pack

    def _memory_format_high_affect_flags(self, high_affect_signal: dict):
        if not isinstance(high_affect_signal, dict):
            return ""
        triggers = high_affect_signal.get("triggers")
        if not isinstance(triggers, dict):
            triggers = {}
        lines = ["[HIGH AFFECT FLAGS]"]
        lines.append(
            "is_high_affect=" + ("true" if bool(high_affect_signal.get("is_high_affect")) else "false")
        )
        lines.append(
            "confidence=" + str(round(float(high_affect_signal.get("confidence", 0.0) or 0.0), 2))
        )
        lines.append(
            "criticism_or_challenge="
            + ("true" if bool(triggers.get("criticism_or_challenge")) else "false")
        )
        lines.append(
            "conflict_or_argument="
            + ("true" if bool(triggers.get("conflict_or_argument")) else "false")
        )
        lines.append(
            "incoming_anecdote="
            + ("true" if bool(triggers.get("incoming_anecdote")) else "false")
        )
        lines.append(
            "defending_prior_opinion="
            + ("true" if bool(triggers.get("defending_prior_opinion")) else "false")
        )
        return "\n".join(lines)

    def _memory_reply_references_recalled_item(self, reply_text: str, recalled_items):
        if not isinstance(reply_text, str) or not reply_text.strip():
            return False, "empty_reply"
        if not isinstance(recalled_items, list) or not recalled_items:
            return False, "no_recalled_items"

        norm_reply = re.sub(r"\s+", " ", reply_text).strip().lower()
        for marker in _HIGH_AFFECT_CALLBACK_MARKERS:
            if marker in norm_reply:
                return True, "callback_marker"

        reply_tokens = {
            tok
            for tok in re.findall(r"[a-z0-9']+", norm_reply)
            if len(tok) >= 4 and tok not in _MEMORY_CALLBACK_STOPWORDS
        }
        if not reply_tokens:
            return False, "no_reply_tokens"

        for item in recalled_items:
            if not isinstance(item, dict):
                continue
            mem_text = str(item.get("text") or "").strip().lower()
            if not mem_text:
                continue
            mem_tokens = {
                tok
                for tok in re.findall(r"[a-z0-9']+", mem_text)
                if len(tok) >= 4 and tok not in _MEMORY_CALLBACK_STOPWORDS
            }
            if len(mem_tokens) < 2:
                continue
            overlap = reply_tokens.intersection(mem_tokens)
            if len(overlap) >= 2:
                return True, "keyword_overlap"

        return False, "missing_callback"

    def _memory_rewrite_reply_with_callback(
        self,
        *,
        draft_text: str,
        thread_context: str,
        recalled_memories_block: str,
        memory_usage_requirement: str,
        high_affect_flags: str,
        interests=None,
    ):
        if not isinstance(self.prompts, dict):
            return draft_text or ""
        prompt = self.prompts.get("handler_memory_callback_rewrite")
        if not isinstance(prompt, str) or not prompt.strip():
            return draft_text or ""

        cfg = self._get_llm_config_for_write_action()
        cfg["temperature"] = min(0.9, max(0.1, float(cfg.get("temperature", 0.6) or 0.6)))

        interests = interests if isinstance(interests, list) else []

        u1 = AssistantAgent(
            name=f"{self.name}",
            llm_config=cfg,
            system_message=self.__effify(
                self.prompts.get(
                    "agent_roleplay_comments_share",
                    self.prompts.get("agent_roleplay_simple", ""),
                ),
                interests=interests,
            ),
            max_consecutive_auto_reply=1,
        )
        u2 = AssistantAgent(
            name="MemoryRewrite",
            llm_config=cfg,
            system_message=self.__effify(self.prompts.get("handler_instructions", "")),
            max_consecutive_auto_reply=0,
        )
        rewritten = ""
        try:
            u2.initiate_chat(
                u1,
                message=self.__effify(
                    prompt,
                    draft_text=(draft_text or "").strip(),
                    conv=self._memory_truncate(thread_context or "", 2200),
                    recalled_memories_block=recalled_memories_block or "",
                    memory_usage_requirement=memory_usage_requirement or "",
                    high_affect_flags=high_affect_flags or "",
                ),
                silent=True,
                max_turns=1,
            )
            rewritten = self._extract_generated_chat_content(
                u2,
                u1,
                prompt_hint=prompt,
                skip_emotion_like=True,
            )
        except Exception:
            rewritten = ""
        finally:
            try:
                u1.reset()
                u2.reset()
            except Exception:
                pass

        rewritten = self.__clean_text(rewritten)
        if len(rewritten) < 3:
            return draft_text or ""
        return rewritten

    def _memory_maybe_generate_reflections(
        self,
        *,
        tid: int,
        other_user_id=None,
        thread_root_id=None,
        reason: str = "periodic",
        query_hint: str = "",
    ):
        if not getattr(self, "memory_enabled", False):
            return
        if not getattr(self, "memory_semantic_enabled", True):
            return
        if not getattr(self, "memory_run_id", None):
            return
        if not getattr(self, "prompts", None):
            return
        if int(getattr(self, "_memory_reflection_count", 0) or 0) >= int(
            getattr(self, "memory_reflection_max_items_per_run", 60) or 60
        ):
            return

        cadence = int(getattr(self, "memory_reflection_cadence_rounds", 3) or 3)
        last_ref = getattr(self, "_memory_last_reflection_round", None)
        periodic_due = last_ref is None or (int(tid) - int(last_ref) >= cadence)

        if reason == "periodic" and not periodic_due:
            return

        min_events = int(getattr(self, "memory_reflection_min_events", 12) or 12)

        events_res = self._memory_api_post(
            "/memory/events_recent",
            {
                "run_id": self.memory_run_id,
                "limit": int(max(60, min_events * 4)),
            },
        )
        events = events_res.get("events") if isinstance(events_res, dict) else []
        if not isinstance(events, list) or len(events) < min_events:
            return

        base_query = self._memory_build_query_text(
            query_hint or "relationship patterns, recurring conflicts, agreements, humor, and trust shifts",
            f"round={tid}",
        )
        search_res = self._memory_search(
            query_text=base_query,
            other_user_id=other_user_id,
            thread_root_id=thread_root_id,
            types=["event", "reflection"],
            round_id=tid,
            k=int(max(12, getattr(self, "memory_search_k", 8))),
            include_evidence_tail=True,
        )
        if not isinstance(search_res, dict):
            return

        items = search_res.get("items") or []
        if not isinstance(items, list) or not items:
            return
        search_user_map = search_res.get("user_map") if isinstance(search_res.get("user_map"), dict) else {}

        def _lookup_search_username(uid):
            try:
                key = str(int(uid))
            except Exception:
                return ""
            try:
                val = search_user_map.get(key) or search_user_map.get(int(key))
            except Exception:
                val = None
            try:
                return str(val or "").strip().lstrip("@")
            except Exception:
                return ""

        importance_sum = 0.0
        for it in items[:12]:
            try:
                importance_sum += float(it.get("importance") or 0.0)
            except Exception:
                continue

        trigger_threshold = float(
            getattr(self, "memory_reflection_trigger_importance_sum", 3.5) or 3.5
        )
        if reason == "event" and not periodic_due and importance_sum < trigger_threshold:
            return

        evidence_lines = []
        for it in items[:12]:
            if not isinstance(it, dict):
                continue
            txt = (it.get("text_humanized") or it.get("text") or "").strip()
            if not txt:
                continue
            if len(txt) > 180:
                txt = txt[:177].rstrip() + "..."
            evidence_lines.append(
                f"- [{it.get('item_type')}] r{it.get('round_id')}: {txt}"
            )

        ctx = self._memory_fetch_context(other_user_id=other_user_id, thread_root_id=thread_root_id) or {}
        community_digest_text = ""
        relationship_hint = ""
        relationship_target_username = ""
        relationship_target_id = ""
        if isinstance(ctx, dict):
            digest = ctx.get("community_digest")
            if isinstance(digest, dict):
                community_digest_text = str(digest.get("digest_text") or "").strip()
            sc = ctx.get("social_card")
            try:
                relationship_target_username = str(ctx.get("other_username") or "").strip().lstrip("@")
            except Exception:
                relationship_target_username = ""
            if isinstance(sc, dict):
                if not relationship_target_username:
                    try:
                        relationship_target_username = str(sc.get("other_username") or "").strip().lstrip("@")
                    except Exception:
                        relationship_target_username = ""
                try:
                    relationship_hint = (
                        f"affinity={float(sc.get('affinity') or 0.0):.2f}, "
                        f"conflict={float(sc.get('conflict') or 0.0):.2f}, "
                        f"humor={float(sc.get('humor') or 0.0):.2f}, "
                        f"trust={float(sc.get('trust') or 0.0):.2f}"
                    )
                except Exception:
                    relationship_hint = str(sc.get("summary_text") or "").strip()
        if not relationship_target_username and other_user_id is not None:
            relationship_target_username = _lookup_search_username(other_user_id)
            relationship_target_id = str(int(other_user_id))

        if not relationship_target_username:
            uname_counts = {}
            for it in items[:16]:
                if not isinstance(it, dict):
                    continue
                uname = it.get("target_username") or it.get("other_username")
                try:
                    uname = str(uname or "").strip().lstrip("@")
                except Exception:
                    uname = ""
                if not uname:
                    continue
                uname_counts[uname] = int(uname_counts.get(uname, 0) or 0) + 1
            if uname_counts:
                relationship_target_username = sorted(uname_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

        if not relationship_target_id and other_user_id is not None:
            try:
                relationship_target_id = str(int(other_user_id))
            except Exception:
                relationship_target_id = ""

        relationship_actor = str(self.name or "").strip().lstrip("@") or "agent"
        if relationship_target_username:
            relationship_target = "@" + relationship_target_username
        elif relationship_target_id:
            relationship_target = f"user_id={relationship_target_id}"
        else:
            relationship_target = "unknown user"

        known_usernames = []
        if isinstance(search_user_map, dict):
            for val in search_user_map.values():
                try:
                    uname = str(val or "").strip().lstrip("@")
                except Exception:
                    uname = ""
                if uname and uname not in known_usernames:
                    known_usernames.append(uname)
        if relationship_target_username and relationship_target_username not in known_usernames:
            known_usernames.append(relationship_target_username)

        prompt = self.prompts.get("handler_memory_generate_reflections")
        if not prompt:
            prompt = (
                "You are generating consolidated memory reflections for a simulated Reddit agent.\n"
                "Build reflections that connect the agent to community dynamics, core stance, topics, and key relationships.\n"
                "Output ONLY JSON with field 'reflections', where reflections is a list of 2-4 objects.\n"
                "Each object fields:\n"
                "- text: <= 280 chars, high-level inference grounded in evidence\n"
                "- importance: float in [0,1]\n"
                "- supporting_event_ids: list of numeric ids (can be empty)\n"
                "- topic_tags: list of up to 4 short strings\n\n"
                "- link_kind: one of [community, core, topic, relationship]\n\n"
                "- optional facets object with:\n"
                "  - topical_experience: short string\n"
                "  - memorable_entities: list (usernames/post ids/short markers)\n"
                "  - negative_experience: short string\n"
                "  - behavior_labels: list from [harassment, spam, slur, dogpile, baiting, toxicity_escalation]\n\n"
                "COMMUNITY DIGEST:\n{community_digest}\n\n"
                "RELATIONSHIP HINT:\n{relationship_hint}\n\n"
                "RELATIONSHIP ACTOR: {relationship_actor}\n"
                "RELATIONSHIP TARGET: {relationship_target}\n"
                "RELATIONSHIP TARGET ID: {relationship_target_id}\n"
                "KNOWN USERNAMES: {known_usernames}\n\n"
                "EVIDENCE:\n{evidence}\n\n"
                "TRIGGER REASON: {reason}\n\n"
                "Rules:\n"
                "- Do not invent user names or events.\n"
                "- If a username is known, prefer '@username' in reflection text.\n"
                "- Avoid raw 'user_id' wording in text unless no username is available.\n"
                "- If inappropriate behavior appears in evidence, use behavior labels and avoid repeating explicit abusive wording.\n"
                "- Prefer one reflection per link_kind when evidence allows.\n"
                "Output JSON only."
            )

        cfg = self.__get_fresh_llm_config()
        cfg["temperature"] = 0.2
        cfg["max_tokens"] = 350

        u1 = AssistantAgent(
            name=f"{self.name}",
            llm_config=cfg,
            system_message=self.__effify(
                self.prompts.get("agent_roleplay_comments_share", self.prompts.get("agent_roleplay_simple", "")),
                interests=[],
            ),
            max_consecutive_auto_reply=1,
        )
        u2 = AssistantAgent(
            name="Handler",
            llm_config=cfg,
            system_message=self.__effify(
                self.prompts.get("handler_instructions_simple", "You are the Handler that specifies the actions to be taken.")
            ),
            max_consecutive_auto_reply=0,
        )

        raw = ""
        try:
            u2.initiate_chat(
                u1,
                message=self.__effify(
                    prompt,
                    evidence="\n".join(evidence_lines),
                    community_digest=community_digest_text,
                    relationship_hint=relationship_hint,
                    relationship_actor=relationship_actor,
                    relationship_target=relationship_target,
                    relationship_target_id=relationship_target_id,
                    known_usernames=", ".join([f"@{x}" for x in known_usernames[:12]]),
                    reason=reason,
                ),
                silent=True,
                max_turns=1,
            )
            raw = u1.chat_messages[u2][-1]["content"]
        except Exception as exc:
            self._memory_warn(f"reflection synthesis LLM call failed at round {tid}: {exc}")
            raw = ""
        finally:
            try:
                u1.reset()
                u2.reset()
            except Exception:
                pass

        parsed = self._memory_extract_json(raw)
        reflections = None
        if isinstance(parsed, list):
            reflections = parsed
        elif isinstance(parsed, dict):
            reflections = parsed.get("reflections")

        if not isinstance(reflections, list):
            if isinstance(raw, str) and raw.strip():
                self._memory_warn(
                    f"reflection parse fallback at round {tid}; raw={self._memory_truncate(raw, 220)}"
                )
            reflections = self._memory_build_reflections_fallback(
                items=items,
                other_user_id=other_user_id,
                other_username=relationship_target_username,
                community_digest_text=community_digest_text,
            )
        if not isinstance(reflections, list) or not reflections:
            return

        created = 0
        for ref in reflections[:4]:
            if isinstance(ref, str):
                ref = {"text": ref}
            if not isinstance(ref, dict):
                continue
            txt = ref.get("text")
            if not isinstance(txt, str) or not txt.strip():
                continue
            txt = self._memory_sanitize_summary_text(txt, 280)
            if not txt:
                continue
            imp = ref.get("importance")
            try:
                imp = float(imp)
            except Exception:
                imp = 0.6
            imp = max(0.0, min(1.0, imp))
            supporting_event_ids = ref.get("supporting_event_ids")
            if isinstance(supporting_event_ids, str):
                try:
                    supporting_event_ids = json.loads(supporting_event_ids)
                except Exception:
                    supporting_event_ids = []
            if not isinstance(supporting_event_ids, list):
                supporting_event_ids = []
            topic_tags = ref.get("topic_tags")
            if isinstance(topic_tags, str):
                try:
                    topic_tags = json.loads(topic_tags)
                except Exception:
                    topic_tags = []
            if not isinstance(topic_tags, list):
                topic_tags = []

            clean_support_ids = []
            for sid in supporting_event_ids:
                try:
                    clean_support_ids.append(int(sid))
                except Exception:
                    continue

            clean_topics = []
            seen_topics = set()
            for tg in topic_tags:
                t = re.sub(r"\s+", " ", str(tg).strip().lower())
                if not t or t in seen_topics:
                    continue
                seen_topics.add(t)
                clean_topics.append(t[:32])
                if len(clean_topics) >= 4:
                    break

            metadata = {"supporting_event_ids": clean_support_ids[:16], "reason": reason}
            link_kind = ref.get("link_kind")
            if isinstance(link_kind, str) and link_kind.strip():
                metadata["link_kind"] = link_kind.strip().lower()[:24]
            if relationship_target_username:
                metadata["other_username"] = relationship_target_username[:64]

            facets = ref.get("facets")
            if not isinstance(facets, dict):
                facets = {}

            topical_experience = facets.get("topical_experience")
            if not isinstance(topical_experience, str):
                topical_experience = ref.get("topical_experience")
            if isinstance(topical_experience, str) and topical_experience.strip():
                metadata["topical_experience"] = self._memory_truncate(
                    self._memory_sanitize_summary_text(topical_experience, 120),
                    120,
                )

            negative_experience = facets.get("negative_experience")
            if not isinstance(negative_experience, str):
                negative_experience = ref.get("negative_experience")
            if isinstance(negative_experience, str) and negative_experience.strip():
                metadata["negative_pattern"] = self._memory_truncate(
                    self._memory_sanitize_summary_text(negative_experience, 120),
                    120,
                )

            memorable_entities = facets.get("memorable_entities")
            if memorable_entities is None:
                memorable_entities = ref.get("memorable_entities")
            clean_entities = []
            if isinstance(memorable_entities, list):
                for ent in memorable_entities:
                    e = re.sub(r"\s+", " ", str(ent or "").strip())
                    if not e:
                        continue
                    clean_entities.append(self._memory_truncate(e, 64))
                    if len(clean_entities) >= 6:
                        break
            if clean_entities:
                metadata["memorable_entities"] = clean_entities

            behavior_labels = facets.get("behavior_labels")
            if behavior_labels is None:
                behavior_labels = ref.get("behavior_labels")
            if isinstance(behavior_labels, str):
                behavior_labels = [behavior_labels]
            clean_behavior_labels = []
            if isinstance(behavior_labels, list):
                allowed_behavior_labels = {
                    "harassment",
                    "spam",
                    "slur",
                    "dogpile",
                    "baiting",
                    "toxicity_escalation",
                }
                for lb in behavior_labels:
                    key = re.sub(r"\s+", "_", str(lb or "").strip().lower())
                    if key in allowed_behavior_labels and key not in clean_behavior_labels:
                        clean_behavior_labels.append(key)
                    if len(clean_behavior_labels) >= 4:
                        break
            if not clean_behavior_labels:
                clean_behavior_labels = self._memory_extract_behavior_labels(txt)[:3]
            if clean_behavior_labels:
                metadata["behavior_labels"] = clean_behavior_labels

            payload = {
                "run_id": self.memory_run_id,
                "agent_user_id": int(self.user_id),
                "item_type": "reflection",
                "text": txt.strip()[:280],
                "importance": imp,
                "round_id": int(tid),
                "thread_root_id": int(thread_root_id) if thread_root_id is not None else None,
                "other_user_id": int(other_user_id) if other_user_id is not None else None,
                "metadata": metadata,
                "topic_tags": clean_topics[:4],
            }
            payload = {k: v for k, v in payload.items() if v is not None}
            res = self._memory_api_post("/memory/item/upsert", payload)
            if isinstance(res, dict) and res.get("status") == 200:
                created += 1
            else:
                self._memory_warn(
                    f"reflection upsert failed at round {tid}: {res if isinstance(res, dict) else 'no response'}"
                )

        if created > 0:
            self._memory_reflection_count = int(getattr(self, "_memory_reflection_count", 0) or 0) + created
            self._memory_last_reflection_round = int(tid)

    def _memory_format_context_for_prompt(self, ctx: dict, other_username: str = None):
        if not ctx:
            return ""

        max_chars = int(
            getattr(self, "memory_prompt_max_chars", DEFAULT_MEMORY_PROMPT_MAX_CHARS)
            or DEFAULT_MEMORY_PROMPT_MAX_CHARS
        )

        def _loads_maybe(val):
            if val is None:
                return None
            if isinstance(val, (list, dict)):
                return val
            if isinstance(val, str):
                s = val.strip()
                if not s:
                    return None
                try:
                    return json.loads(s)
                except Exception:
                    return s
            return val

        lines = ["[MEMORY CONTEXT]"]

        digest = ctx.get("community_digest") if isinstance(ctx, dict) else None
        if isinstance(digest, dict):
            dt = (digest.get("digest_text") or "").strip()
            if dt:
                lines.append(f"Community vibe: {dt}")
            top_topics = _loads_maybe(digest.get("top_topics"))
            if isinstance(top_topics, list) and top_topics:
                lines.append("Top topics: " + ", ".join([str(x) for x in top_topics[:8] if str(x).strip()]))
            norms = _loads_maybe(digest.get("norms"))
            if isinstance(norms, list) and norms:
                lines.append("Norms: " + ", ".join([str(x) for x in norms[:5] if str(x).strip()]))
            memes = _loads_maybe(digest.get("memes"))
            if isinstance(memes, list) and memes:
                lines.append("Memes: " + ", ".join([str(x) for x in memes[:5] if str(x).strip()]))

        sc = ctx.get("social_card") if isinstance(ctx, dict) else None
        if isinstance(sc, dict):
            uname = other_username or ctx.get("other_username") or sc.get("other_username") or "this user"
            try:
                uname = str(uname).strip().lstrip("@") or "this user"
            except Exception:
                uname = "this user"
            try:
                aff = float(sc.get("affinity") or 0.0)
                con = float(sc.get("conflict") or 0.0)
                hum = float(sc.get("humor") or 0.0)
                tru = float(sc.get("trust") or 0.0)
                lines.append(
                    f"History with @{uname}: affinity={aff:.2f}, conflict={con:.2f}, humor={hum:.2f}, trust={tru:.2f}"
                )
            except Exception:
                pass
            lrl = sc.get("last_relation_label")
            if isinstance(lrl, str) and lrl.strip():
                lines.append("Last interaction label: " + lrl.strip())
            st = sc.get("summary_text")
            if isinstance(st, str) and st.strip():
                clean_summary = self._memory_sanitize_prompt_memory_text(st.strip(), 220)
                if clean_summary:
                    lines.append("Summary: " + clean_summary)
            et = _loads_maybe(sc.get("evidence_tail"))
            if isinstance(et, list) and et:
                ev_lines = []
                for ev in et[-3:]:
                    if not isinstance(ev, dict):
                        continue
                    if self._memory_is_vote_artifact(
                        text_value=ev.get("salient_claim") or "",
                        relation_label=ev.get("relation_label") or "",
                    ):
                        continue
                    rid = ev.get("round_id")
                    claim = (ev.get("salient_claim") or "").strip()
                    if claim and len(claim) > 120:
                        claim = claim[:117].rstrip() + "..."
                    if claim:
                        ev_lines.append(f"- r{rid}: {claim}".strip())
                if ev_lines:
                    lines.append("Evidence:")
                    lines.extend(ev_lines)

        tc = ctx.get("thread_card") if isinstance(ctx, dict) else None
        if isinstance(tc, dict):
            gt = tc.get("gist_text")
            if isinstance(gt, str) and gt.strip():
                lines.append("Thread gist: " + gt.strip())
            mr = tc.get("my_role")
            if isinstance(mr, str) and mr.strip():
                lines.append("My role in thread: " + mr.strip())
            pt = _loads_maybe(tc.get("participants_top"))
            if isinstance(pt, list) and pt:
                lines.append(
                    "Top participants: "
                    + ", ".join([str(x).lstrip("@") for x in pt[:6] if isinstance(x, str) and x.strip()])
                )
            ep = _loads_maybe(tc.get("entry_points"))
            if isinstance(ep, list) and ep:
                lines.append("Good reply targets (ids): " + ", ".join([str(x) for x in ep[:8]]))

        rpe = ctx.get("recent_pair_events") if isinstance(ctx, dict) else None
        if isinstance(rpe, list) and rpe:
            lines.append("Recent interactions:")
            for ev in rpe[-7:]:
                if not isinstance(ev, dict):
                    continue
                if self._memory_is_vote_artifact(
                    text_value=ev.get("salient_claim") or "",
                    event_type=ev.get("event_type") or "",
                    relation_label=ev.get("relation_label") or "",
                ):
                    continue
                rid = ev.get("round_id")
                et = ev.get("event_type")
                rl = ev.get("relation_label")
                claim = (ev.get("salient_claim") or "").strip()
                if claim and len(claim) > 120:
                    claim = claim[:117].rstrip() + "..."
                actor = str(ev.get("actor_username") or "").strip().lstrip("@")
                target = str(ev.get("target_username") or "").strip().lstrip("@")
                if actor or target:
                    pair = f" @{actor or 'user'} -> @{target or 'user'}"
                else:
                    pair = ""
                lines.append(f"- r{rid}:{pair} {et} ({rl}) {claim}".strip())

        out = "\n".join([ln for ln in lines if ln and str(ln).strip()]).strip()
        if max_chars > 0 and len(out) > max_chars:
            out = out[: max_chars - 3].rstrip() + "..."
        return out

    def _memory_llm_interaction_note(
        self,
        *,
        tid: int,
        event_type: str,
        other_username: str,
        other_text: str,
        my_text: str,
        memory_context_text: str,
    ):
        if not getattr(self, "prompts", None):
            return None

        prompt = self.prompts.get("handler_memory_interaction_note")
        if not prompt:
            prompt = (
                "You are writing a memory note for a simulated Reddit user.\n"
                "Output ONLY a valid JSON object (no markdown) with fields:\n"
                "- relation_label: one of [agree, disagree, funny, helpful, hostile, neutral]\n"
                "- tone_label: one of [supportive, snarky, angry, curious, neutral]\n"
                "- affinity_delta: float in [-1,1]\n"
                "- conflict_delta: float in [-1,1]\n"
                "- humor_delta: float in [-1,1]\n"
                "- trust_delta: float in [-1,1]\n"
                "- salient_claim: <=200 chars\n"
                "- topics: list of up to 5 short strings\n\n"
                "MEMORY CONTEXT:\n{memory_context_text}\n\n"
                "INTERACTION:\n"
                "- event_type: {event_type}\n"
                "- replying to @{other_username}: {other_text}\n"
                "- my text: {my_text}\n\n"
                "Output JSON only."
            )

        cfg = self.__get_fresh_llm_config()
        cfg["temperature"] = 0.2
        cfg["max_tokens"] = 300

        u1 = AssistantAgent(
            name=f"{self.name}",
            llm_config=cfg,
            system_message=self.__effify(
                self.prompts.get("agent_roleplay_comments_share", self.prompts.get("agent_roleplay_simple", "")),
                interests=[],
            ),
            max_consecutive_auto_reply=1,
        )

        u2 = AssistantAgent(
            name="Handler",
            llm_config=cfg,
            system_message=self.__effify(
                self.prompts.get("handler_instructions_simple", "You are the Handler that specifies the actions to be taken.")
            ),
            max_consecutive_auto_reply=0,
        )

        u2.initiate_chat(
            u1,
            message=self.__effify(
                prompt,
                memory_context_text=memory_context_text or "",
                event_type=event_type,
                other_username=other_username or "unknown",
                other_text=(other_text or "").strip(),
                my_text=(my_text or "").strip(),
            ),
            silent=True,
            max_turns=1,
        )

        raw = ""
        try:
            raw = u1.chat_messages[u2][-1]["content"]
        except Exception:
            raw = ""

        u1.reset()
        u2.reset()

        note = self._memory_extract_json(raw)
        if not isinstance(note, dict):
            return None

        return note

    def _memory_record_event(
        self,
        *,
        tid: int,
        event_type: str,
        target_user_id=None,
        thread_root_id=None,
        target_post_id=None,
        relation_label=None,
        tone_label=None,
        topics=None,
        salient_claim=None,
        weight: float = 1.0,
    ):
        if not getattr(self, "memory_enabled", False):
            return None
        if not getattr(self, "memory_run_id", None):
            return None

        cold_start_window = int(getattr(self, "memory_cold_start_window", 5) or 5)
        if cold_start_window < 1:
            cold_start_window = 1
        if cold_start_window > 1000:
            cold_start_window = 1000

        payload = {
            "run_id": self.memory_run_id,
            "round_id": int(tid),
            "actor_user_id": int(self.user_id),
            "event_type": str(event_type).strip().lower(),
            "weight": float(weight) if weight is not None else 1.0,
            "cold_start_window": cold_start_window,
        }
        if target_user_id is not None:
            try:
                payload["target_user_id"] = int(target_user_id)
            except Exception:
                pass
        if thread_root_id is not None:
            try:
                payload["thread_root_id"] = int(thread_root_id)
            except Exception:
                pass
        if target_post_id is not None:
            try:
                payload["target_post_id"] = int(target_post_id)
            except Exception:
                pass
        if isinstance(relation_label, str) and relation_label.strip():
            payload["relation_label"] = relation_label.strip().lower()[:16]
        if isinstance(tone_label, str) and tone_label.strip():
            payload["tone_label"] = tone_label.strip().lower()[:16]
        if topics is not None:
            payload["topics"] = topics
        if isinstance(salient_claim, str) and salient_claim.strip():
            payload["salient_claim"] = salient_claim.strip()[:200]

        result = self._memory_api_post("/memory/event", payload)

        # Track cold-start index from server response
        if isinstance(result, dict) and result.get("status") == 200:
            server_count = result.get("interaction_event_count")
            server_decay_level = result.get("cold_start_decay_level")
            try:
                server_count = int(server_count)
            except Exception:
                server_count = None
            try:
                server_decay_level = int(server_decay_level)
            except Exception:
                server_decay_level = None
            if server_count is not None and server_count >= 0:
                self._memory_global_interaction_index = server_count
            else:
                self._memory_global_interaction_index = int(
                    getattr(self, "_memory_global_interaction_index", 0) or 0
                ) + 1
            if server_decay_level is not None and server_decay_level >= 0:
                self._memory_cold_start_decay_level = server_decay_level
            else:
                self._memory_cold_start_decay_level = max(
                    0,
                    int(getattr(self, "_memory_global_interaction_index", 0) or 0)
                    - int(cold_start_window),
                )

        return result

    def _memory_upsert_social_card(
        self,
        *,
        tid: int,
        other_user_id: int,
        thread_root_id=None,
        deltas: dict,
        relation_label=None,
        tone_label=None,
        salient_claim=None,
        include_evidence=True,
        count_as_event=True,
    ):
        if not getattr(self, "memory_enabled", False):
            return
        if not getattr(self, "memory_run_id", None):
            return
        if other_user_id is None:
            return

        other_user_id = int(other_user_id)

        card = self._memory_cache_social.get(other_user_id)
        if not isinstance(card, dict):
            ctx = self._memory_fetch_context(other_user_id=other_user_id, thread_root_id=thread_root_id)
            card = ctx.get("social_card") if isinstance(ctx, dict) else None
        if not isinstance(card, dict):
            card = {}

        last_updated_round = card.get("last_updated_round")
        try:
            delta_rounds = int(tid) - int(last_updated_round) if last_updated_round is not None else 0
        except Exception:
            delta_rounds = 0

        def _getf(k):
            try:
                return float(card.get(k) or 0.0)
            except Exception:
                return 0.0

        affinity = self._memory_decay_value(_getf("affinity"), delta_rounds, self.memory_social_decay_lambda)
        conflict = self._memory_decay_value(_getf("conflict"), delta_rounds, self.memory_social_decay_lambda)
        humor = self._memory_decay_value(_getf("humor"), delta_rounds, self.memory_social_decay_lambda)
        trust = self._memory_decay_value(_getf("trust"), delta_rounds, self.memory_social_decay_lambda)

        def _clip(x):
            try:
                x = float(x)
            except Exception:
                x = 0.0
            return max(-5.0, min(5.0, x))

        affinity = _clip(affinity + float(deltas.get("affinity_delta", 0.0) or 0.0))
        conflict = _clip(conflict + float(deltas.get("conflict_delta", 0.0) or 0.0))
        humor = _clip(humor + float(deltas.get("humor_delta", 0.0) or 0.0))
        trust = _clip(trust + float(deltas.get("trust_delta", 0.0) or 0.0))

        evidence_tail = []
        et_raw = card.get("evidence_tail")
        if isinstance(et_raw, str) and et_raw.strip():
            try:
                evidence_tail = json.loads(et_raw)
            except Exception:
                evidence_tail = []
        if not isinstance(evidence_tail, list):
            evidence_tail = []

        if include_evidence:
            evidence_tail.append(
                {
                    "round_id": int(tid),
                    "thread_root_id": int(thread_root_id) if thread_root_id is not None else None,
                    "relation_label": relation_label,
                    "tone_label": tone_label,
                    "salient_claim": (salient_claim or "")[:200],
                }
            )
            evidence_tail = evidence_tail[-int(getattr(self, "memory_evidence_tail_max", 8) or 8) :]

        summary_text = card.get("summary_text")
        if isinstance(summary_text, str) and summary_text.strip():
            if not getattr(self, "memory_semantic_enabled", True):
                summary_text = self._memory_corrupt_text(summary_text.strip(), self.memory_social_corruption_rate)
            else:
                summary_text = summary_text.strip()
            summary_text = self._memory_sanitize_summary_text(summary_text, 800)
        else:
            summary_text = None

        reflection_hint = ""
        if getattr(self, "memory_semantic_enabled", True):
            try:
                refl = self._memory_search(
                    query_text=self._memory_build_query_text(
                        "high-level relationship reflections",
                        f"user={other_user_id}",
                    ),
                    other_user_id=other_user_id,
                    thread_root_id=thread_root_id,
                    types=["reflection"],
                    round_id=tid,
                    k=3,
                )
                reflection_hint = self._memory_format_search_brief(refl, 450) if isinstance(refl, dict) else ""
            except Exception:
                reflection_hint = ""

        event_count = 0
        try:
            event_count = int(card.get("event_count") or 0)
        except Exception:
            event_count = 0
        if count_as_event:
            event_count += 1

        last_relation_label = card.get("last_relation_label")
        if isinstance(relation_label, str) and relation_label.strip():
            last_relation_label = str(relation_label).strip().lower()[:16]

        if (
            count_as_event
            and self.memory_social_resummarize_every_events > 0
            and (event_count % self.memory_social_resummarize_every_events == 0)
            and getattr(self, "prompts", None)
        ):
            prompt = self.prompts.get("handler_memory_resummarize_social_card")
            if not prompt:
                prompt = (
                    "You are updating a short social memory about another Reddit user.\n"
                    "Output ONLY JSON: {\"summary_text\": \"...\"}.\n\n"
                    "STATS: affinity={affinity:.2f}, conflict={conflict:.2f}, humor={humor:.2f}, trust={trust:.2f}\n"
                    "OLD SUMMARY: {old_summary}\n"
                    "EXISTING REFLECTIONS:\n{reflection_hint}\n"
                    "RECENT EVIDENCE:\n{evidence_lines}\n\n"
                    "Rules:\n"
                    "- summary_text <= 400 chars\n"
                    "- Include recurring topics and one memorable interaction marker.\n"
                    "- Mention whether trust is rising/falling and whether friction is recurring.\n"
                    "- If inappropriate behavior appears, use behavior labels only (harassment/spam/slur/dogpile/baiting/toxicity_escalation).\n"
                    "- Do not quote explicit abusive language.\n"
                    "- Write in plain English.\n"
                    "- Output JSON only."
                )

            evidence_lines = []
            behavior_counts = {}
            for ev in evidence_tail[-6:]:
                if not isinstance(ev, dict):
                    continue
                rid = ev.get("round_id")
                claim = (ev.get("salient_claim") or "").strip()
                if claim and len(claim) > 120:
                    claim = claim[:117].rstrip() + "..."
                rel = ev.get("relation_label")
                tone = ev.get("tone_label")
                for lb in self._memory_extract_behavior_labels(
                    claim, relation_label=str(rel or ""), tone_label=str(tone or "")
                ):
                    behavior_counts[lb] = int(behavior_counts.get(lb, 0) or 0) + 1
                evidence_lines.append(f"- r{rid}: {claim}".strip())
            behavior_labels = [
                k
                for k, _ in sorted(behavior_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
            ]

            cfg = self.__get_fresh_llm_config()
            cfg["temperature"] = 0.2
            cfg["max_tokens"] = 250

            u1 = AssistantAgent(
                name=f"{self.name}",
                llm_config=cfg,
                system_message=self.__effify(
                    self.prompts.get("agent_roleplay_comments_share", self.prompts.get("agent_roleplay_simple", "")),
                    interests=[],
                ),
                max_consecutive_auto_reply=1,
            )
            u2 = AssistantAgent(
                name="Handler",
                llm_config=cfg,
                system_message=self.__effify(
                    self.prompts.get("handler_instructions_simple", "You are the Handler that specifies the actions to be taken.")
                ),
                max_consecutive_auto_reply=0,
            )

            u2.initiate_chat(
                u1,
                message=self.__effify(
                    prompt,
                    affinity=affinity,
                    conflict=conflict,
                    humor=humor,
                    trust=trust,
                    old_summary=summary_text or "",
                    reflection_hint=reflection_hint or "",
                    evidence_lines="\n".join(evidence_lines),
                    behavior_labels=", ".join(behavior_labels),
                ),
                silent=True,
                max_turns=1,
            )

            raw = ""
            try:
                raw = u1.chat_messages[u2][-1]["content"]
            except Exception:
                raw = ""
            u1.reset()
            u2.reset()

            parsed = self._memory_extract_json(raw)
            if isinstance(parsed, dict):
                st = parsed.get("summary_text")
                if isinstance(st, str) and st.strip():
                    summary_text = self._memory_sanitize_summary_text(st.strip(), 800)

        payload = {
            "run_id": self.memory_run_id,
            "agent_user_id": int(self.user_id),
            "other_user_id": int(other_user_id),
            "affinity": affinity,
            "conflict": conflict,
            "humor": humor,
            "trust": trust,
            "last_relation_label": last_relation_label,
            "last_round_id": int(tid),
            "last_thread_root_id": int(thread_root_id) if thread_root_id is not None else None,
            "last_updated_round": int(tid),
            "event_count": int(event_count),
            "summary_text": summary_text,
            "evidence_tail": evidence_tail,
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        self._memory_api_post("/memory/social/upsert", payload)

        self._memory_cache_social[other_user_id] = {
            "affinity": affinity,
            "conflict": conflict,
            "humor": humor,
            "trust": trust,
            "last_relation_label": payload.get("last_relation_label"),
            "last_round_id": int(tid),
            "last_thread_root_id": payload.get("last_thread_root_id"),
            "last_updated_round": int(tid),
            "event_count": int(event_count),
            "summary_text": summary_text,
            "evidence_tail": json.dumps(evidence_tail),
        }

    def _memory_maybe_update_thread_card(self, *, tid: int, thread_root_id: int, conv_text: str):
        if not getattr(self, "memory_enabled", False):
            return
        if not getattr(self, "memory_run_id", None):
            return
        if thread_root_id is None:
            return

        thread_root_id = int(thread_root_id)
        self._memory_thread_event_counts[thread_root_id] = self._memory_thread_event_counts.get(thread_root_id, 0) + 1
        count = self._memory_thread_event_counts[thread_root_id]

        ctx = self._memory_fetch_context(other_user_id=None, thread_root_id=thread_root_id)
        card = ctx.get("thread_card") if isinstance(ctx, dict) else None
        if not isinstance(card, dict):
            card = {}

        gist_text = card.get("gist_text")
        if isinstance(gist_text, str) and gist_text.strip():
            if not getattr(self, "memory_semantic_enabled", True):
                gist_text = self._memory_corrupt_text(gist_text.strip(), self.memory_thread_corruption_rate)
            else:
                gist_text = gist_text.strip()
        else:
            gist_text = None

        should_resummarize = (
            gist_text is None
            or (
                self.memory_thread_resummarize_every_events > 0
                and (count % self.memory_thread_resummarize_every_events == 0)
            )
        )

        my_role = card.get("my_role")
        participants_top = card.get("participants_top")
        entry_points = card.get("entry_points")
        reflection_hint = ""
        if getattr(self, "memory_semantic_enabled", True):
            try:
                refl = self._memory_search(
                    query_text=self._memory_build_query_text(
                        "thread-level reflection and key entry points",
                        f"thread_root_id={thread_root_id}",
                    ),
                    thread_root_id=thread_root_id,
                    types=["reflection", "summary"],
                    round_id=tid,
                    k=4,
                )
                reflection_hint = self._memory_format_search_brief(refl, 500) if isinstance(refl, dict) else ""
            except Exception:
                reflection_hint = ""

        if should_resummarize and getattr(self, "prompts", None):
            prompt = self.prompts.get("handler_memory_update_thread_card")
            if not prompt:
                prompt = (
                    "You are updating a short memory of a Reddit thread.\n"
                    "Output ONLY JSON with fields:\n"
                    "- gist_text: <= 500 chars summary of what this thread is about\n"
                    "- my_role: one of [supporter, skeptic, joker, explainer, lurker, neutral]\n"
                    "- participants_top: list of up to 6 usernames\n"
                    "- entry_points: list of up to 5 post/comment ids that are good reply targets\n\n"
                    "KNOWN REFLECTIONS:\n{reflection_hint}\n\n"
                    "THREAD SNIPPET:\n{thread_snippet}\n\n"
                    "Output JSON only."
                )

            snippet = (conv_text or "").strip()
            if len(snippet) > 3000:
                snippet = snippet[-3000:]

            cfg = self.__get_fresh_llm_config()
            cfg["temperature"] = 0.2
            cfg["max_tokens"] = 300

            u1 = AssistantAgent(
                name=f"{self.name}",
                llm_config=cfg,
                system_message=self.__effify(
                    self.prompts.get("agent_roleplay_comments_share", self.prompts.get("agent_roleplay_simple", "")),
                    interests=[],
                ),
                max_consecutive_auto_reply=1,
            )
            u2 = AssistantAgent(
                name="Handler",
                llm_config=cfg,
                system_message=self.__effify(
                    self.prompts.get("handler_instructions_simple", "You are the Handler that specifies the actions to be taken.")
                ),
                max_consecutive_auto_reply=0,
            )

            u2.initiate_chat(
                u1,
                message=self.__effify(prompt, thread_snippet=snippet, reflection_hint=reflection_hint or ""),
                silent=True,
                max_turns=1,
            )

            raw = ""
            try:
                raw = u1.chat_messages[u2][-1]["content"]
            except Exception:
                raw = ""
            u1.reset()
            u2.reset()

            parsed = self._memory_extract_json(raw)
            if isinstance(parsed, dict):
                gt = parsed.get("gist_text")
                if isinstance(gt, str) and gt.strip():
                    gist_text = gt.strip()[:2000]
                mr = parsed.get("my_role")
                if isinstance(mr, str) and mr.strip():
                    my_role = mr.strip().lower()[:16]
                pt = parsed.get("participants_top")
                if isinstance(pt, list):
                    participants_top = pt
                ep = parsed.get("entry_points")
                if isinstance(ep, list):
                    entry_points = ep

        payload = {
            "run_id": self.memory_run_id,
            "agent_user_id": int(self.user_id),
            "thread_root_id": int(thread_root_id),
            "gist_text": gist_text,
            "my_role": my_role,
            "participants_top": participants_top,
            "entry_points": entry_points,
            "last_seen_round_id": int(tid),
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        self._memory_api_post("/memory/thread/upsert", payload)

        self._memory_cache_thread[thread_root_id] = {
            "gist_text": gist_text,
            "my_role": my_role,
            "participants_top": json.dumps(participants_top) if isinstance(participants_top, (list, dict)) else participants_top,
            "entry_points": json.dumps(entry_points) if isinstance(entry_points, (list, dict)) else entry_points,
            "last_seen_round_id": int(tid),
        }

    def _memory_maybe_update_community_digest(self, *, tid: int):
        if not getattr(self, "memory_enabled", False):
            return
        if not getattr(self, "memory_run_id", None):
            return

        global _MEMORY_LAST_DIGEST_UPDATE_ROUND
        cadence = int(getattr(self, "memory_digest_update_cadence_rounds", 3) or 3)
        if cadence <= 0:
            return

        if _MEMORY_LAST_DIGEST_UPDATE_ROUND is not None:
            try:
                if int(tid) - int(_MEMORY_LAST_DIGEST_UPDATE_ROUND) < cadence:
                    return
            except Exception:
                pass

        _MEMORY_LAST_DIGEST_UPDATE_ROUND = int(tid)

        if not getattr(self, "prompts", None):
            return

        root_posts = self._memory_get_recent_root_posts(
            tid=int(tid),
            limit=int(getattr(self, "memory_digest_events_limit", 24) or 24),
            rounds_back=18,
        )
        maturity = self._memory_digest_maturity_meta(root_posts)

        prev_digest = None
        if isinstance(getattr(self, "_memory_cache_digest", None), dict):
            prev_digest = self._memory_cache_digest
        else:
            dg = self._memory_api_post("/memory/community/get", {"run_id": self.memory_run_id})
            if isinstance(dg, dict) and dg.get("status") == 200:
                prev_digest = dg

        prompt = self.prompts.get("handler_memory_update_community_digest")
        if not prompt:
            prompt = (
                "You are updating a short 'community digest' for a simulated Reddit forum.\n"
                "Output ONLY JSON with fields:\n"
                "- digest_text: <= 700 chars describing how people usually start threads here\n"
                "- top_topics: list of up to 8 topics\n"
                "- norms: list of up to 6 lowercase style descriptors\n"
                "- memes: list of up to 6 lowercase recurring post formats\n"
                "- polarizing_issues: list of up to 6 broad argument patterns\n\n"
                "PREVIOUS DIGEST:\n{prev_digest}\n\n"
                "RECENT ROOT POSTS:\n{recent_roots_text}\n\n"
                "Rules:\n"
                "- Summarize style and structure, not specific named shows, movies, or articles.\n"
                "- Keep norms and memes abstract, lowercase, and reusable as a loose style guide.\n"
                "- If you mention topics at all, keep them broad and generic.\n"
                "Output JSON only."
            )

        root_lines = []
        for row in root_posts[:24]:
            if not isinstance(row, dict):
                continue
            kind = self._post_root_origin_kind(row)
            parsed = self._memory_parse_root_post_text(row.get("tweet") or "")
            title = self._memory_truncate(parsed.get("title") or parsed.get("combined") or "", 120)
            body = self._memory_truncate(parsed.get("body") or "", 120)
            bits = [f"- r{row.get('round')}: {kind}"]
            if title:
                bits.append(f"title={title}")
            if body:
                bits.append(f"body={body}")
            root_lines.append(" | ".join(bits))
        recent_roots_text = "\n".join(root_lines)

        prev_digest_text = ""
        if isinstance(prev_digest, dict):
            prev_digest_text = (prev_digest.get("digest_text") or "").strip()

        cfg = self.__get_fresh_llm_config()
        cfg["temperature"] = 0.2
        cfg["max_tokens"] = 500

        u1 = AssistantAgent(
            name=f"{self.name}",
            llm_config=cfg,
            system_message=self.__effify(
                self.prompts.get("agent_roleplay_base", self.prompts.get("agent_roleplay_simple", "")),
                interests=[],
            ),
            max_consecutive_auto_reply=1,
        )
        u2 = AssistantAgent(
            name="Handler",
            llm_config=cfg,
            system_message=self.__effify(
                self.prompts.get("handler_instructions_simple", "You are the Handler that specifies the actions to be taken.")
            ),
            max_consecutive_auto_reply=0,
        )

        raw = ""
        try:
            u2.initiate_chat(
                u1,
                message=self.__effify(
                    prompt,
                    prev_digest=prev_digest_text,
                    recent_roots_text=recent_roots_text,
                    events_text=recent_roots_text,
                    root_count=int(maturity.get("root_count") or 0),
                    distinct_author_count=int(maturity.get("distinct_author_count") or 0),
                ),
                silent=True,
                max_turns=1,
            )
            raw = u1.chat_messages[u2][-1]["content"]
        except Exception as exc:
            self._memory_warn(f"community digest LLM call failed at round {tid}: {exc}")
            raw = ""
        finally:
            try:
                u1.reset()
                u2.reset()
            except Exception:
                pass

        parsed = self._memory_extract_json(raw)
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed and isinstance(parsed[0], dict) else None
        if not isinstance(parsed, dict):
            if isinstance(raw, str) and raw.strip():
                self._memory_warn(
                    f"community digest parse fallback at round {tid}; raw={self._memory_truncate(raw, 240)}"
                )
            parsed = self._memory_build_community_digest_fallback(root_posts, prev_digest_text=prev_digest_text)

        update_payload = {"run_id": self.memory_run_id, "round_id": int(tid)}
        for k in ["digest_text", "top_topics", "norms", "memes", "polarizing_issues"]:
            if k in parsed and parsed.get(k) is not None:
                update_payload[k] = parsed.get(k)

        if "digest_text" not in update_payload:
            fallback = self._memory_build_community_digest_fallback(root_posts, prev_digest_text=prev_digest_text)
            update_payload.update(fallback)
        else:
            update_payload["digest_text"] = self._memory_sanitize_summary_text(
                str(update_payload.get("digest_text") or ""),
                700,
            )

        update_res = self._memory_api_post("/memory/community/update", update_payload)
        if not (isinstance(update_res, dict) and update_res.get("status") == 200):
            self._memory_warn(
                f"community digest upsert failed at round {tid}: {update_res if isinstance(update_res, dict) else 'no response'}"
            )
            return

        self._memory_cache_digest = {
            "round_id": int(tid),
            "digest_text": update_payload.get("digest_text"),
            "top_topics": json.dumps(update_payload.get("top_topics"))
            if isinstance(update_payload.get("top_topics"), (list, dict))
            else update_payload.get("top_topics"),
            "norms": json.dumps(update_payload.get("norms"))
            if isinstance(update_payload.get("norms"), (list, dict))
            else update_payload.get("norms"),
            "memes": json.dumps(update_payload.get("memes"))
            if isinstance(update_payload.get("memes"), (list, dict))
            else update_payload.get("memes"),
            "polarizing_issues": json.dumps(update_payload.get("polarizing_issues"))
            if isinstance(update_payload.get("polarizing_issues"), (list, dict))
            else update_payload.get("polarizing_issues"),
        }

        self._memory_maybe_generate_reflections(
            tid=int(tid),
            reason="periodic",
            query_hint=update_payload.get("digest_text") or "community reflection",
        )

    def _memory_after_comment(
        self,
        *,
        tid: int,
        target_post_id: int,
        thread_root_id: int,
        other_user_id: int,
        other_username: str,
        other_text: str,
        my_text: str,
        conv_text: str,
    ):
        if not getattr(self, "memory_enabled", False):
            return

        mem_text, _ = self._memory_build_tiered_context(
            query_text=self._memory_build_query_text(
                "write interaction memory note",
                other_text or "",
                my_text or "",
            ),
            other_user_id=other_user_id,
            thread_root_id=thread_root_id,
            other_username=other_username,
            round_id=tid,
            uncertainty_score=0.5,
        )

        note = self._memory_llm_interaction_note(
            tid=int(tid),
            event_type="comment",
            other_username=other_username or "unknown",
            other_text=other_text or "",
            my_text=my_text or "",
            memory_context_text=mem_text,
        )

        relation_label = None
        tone_label = None
        salient_claim = None
        topics = None
        deltas = {"affinity_delta": 0.0, "conflict_delta": 0.0, "humor_delta": 0.0, "trust_delta": 0.0}

        if isinstance(note, dict):
            relation_label = note.get("relation_label")
            tone_label = note.get("tone_label")
            salient_claim = note.get("salient_claim")
            topics = note.get("topics")
            for k in ["affinity_delta", "conflict_delta", "humor_delta", "trust_delta"]:
                if k in note:
                    try:
                        deltas[k] = float(note.get(k))
                    except Exception:
                        pass

        if not salient_claim:
            snippet = (my_text or "").strip()
            if snippet and len(snippet) > 120:
                snippet = snippet[:117].rstrip() + "..."
            if snippet:
                salient_claim = f"comment replying to @{other_username or 'user'}: {snippet}"

        self._memory_record_event(
            tid=int(tid),
            event_type="comment",
            target_user_id=other_user_id,
            thread_root_id=thread_root_id,
            target_post_id=target_post_id,
            relation_label=relation_label,
            tone_label=tone_label,
            topics=topics,
            salient_claim=salient_claim,
            weight=1.0,
        )

        self._memory_upsert_social_card(
            tid=int(tid),
            other_user_id=int(other_user_id),
            thread_root_id=thread_root_id,
            deltas=deltas,
            relation_label=relation_label,
            tone_label=tone_label,
            salient_claim=salient_claim or (my_text or "")[:200],
        )

        self._memory_maybe_update_thread_card(tid=int(tid), thread_root_id=int(thread_root_id), conv_text=conv_text or "")
        self._memory_maybe_update_community_digest(tid=int(tid))
        self._memory_maybe_generate_reflections(
            tid=int(tid),
            other_user_id=other_user_id,
            thread_root_id=thread_root_id,
            reason="event",
            query_hint=self._memory_build_query_text(other_text, my_text),
        )

    def _memory_after_vote(self, *, tid: int, post_id: int, vote_type: str):
        if not getattr(self, "memory_enabled", False):
            return

        other_user_id, other_username = self._memory_get_author_id_and_username(int(post_id))
        thread_root_id = self._memory_get_thread_root_id(int(post_id)) or None

        event_type = "upvote" if vote_type == "like" else "downvote"
        deltas = {"affinity_delta": 0.0, "conflict_delta": 0.0, "humor_delta": 0.0, "trust_delta": 0.0}
        if event_type == "upvote":
            deltas["affinity_delta"] = 0.5
            deltas["trust_delta"] = 0.2
            deltas["conflict_delta"] = -0.1
        else:
            deltas["conflict_delta"] = 0.6
            deltas["affinity_delta"] = -0.3
            deltas["trust_delta"] = -0.2

        if not getattr(self, "memory_vote_signal_only", False):
            snippet = ""
            try:
                raw_post_text = self.__get_post(int(post_id))
                if isinstance(raw_post_text, str):
                    snippet = raw_post_text.strip()
            except Exception:
                snippet = ""
            if snippet and len(snippet) > 120:
                snippet = snippet[:117].rstrip() + "..."

            if snippet:
                salient_claim = f"{event_type} on @{other_username or 'user'}: {snippet}"
            else:
                salient_claim = f"{event_type} on a comment by @{other_username or 'user'}"

            self._memory_record_event(
                tid=int(tid),
                event_type=event_type,
                target_user_id=other_user_id,
                thread_root_id=thread_root_id,
                target_post_id=int(post_id),
                relation_label=None,
                tone_label=None,
                topics=None,
                salient_claim=salient_claim,
                weight=1.0,
            )

        if other_user_id is not None:
            self._memory_upsert_social_card(
                tid=int(tid),
                other_user_id=int(other_user_id),
                thread_root_id=thread_root_id,
                deltas=deltas,
                relation_label=None,
                tone_label=None,
                salient_claim=None,
                include_evidence=not getattr(self, "memory_vote_signal_only", False),
                count_as_event=not getattr(self, "memory_vote_signal_only", False),
            )

        if not getattr(self, "memory_vote_signal_only", False):
            self._memory_maybe_update_community_digest(tid=int(tid))
            self._memory_maybe_generate_reflections(
                tid=int(tid),
                other_user_id=other_user_id,
                thread_root_id=thread_root_id,
                reason="event",
                query_hint=salient_claim,
            )

    def _memory_after_post(self, *, tid: int, post_text: str, topics=None):
        if not getattr(self, "memory_enabled", False):
            return

        salient = (post_text or "").strip()
        if len(salient) > 200:
            salient = salient[:197].rstrip() + "..."

        self._memory_record_event(
            tid=int(tid),
            event_type="post",
            target_user_id=None,
            thread_root_id=None,
            target_post_id=None,
            relation_label=None,
            tone_label=None,
            topics=topics,
            salient_claim=salient,
            weight=1.0,
        )

        self._memory_maybe_update_community_digest(tid=int(tid))
        self._memory_maybe_generate_reflections(
            tid=int(tid),
            other_user_id=None,
            thread_root_id=None,
            reason="event",
            query_hint=salient,
        )

    def reset_round_post_count(self):
        """Reset the post counter and restore base temperature after a round."""
        self.posts_this_round = 0
        self.replies_this_round = 0
        self.writing_actions_this_round = 0
        self._recent_comments_by_round_parent = {}
        self._recent_generated_comments = []
        self._proactive_affect_this_round = 0
        self.llm_config["temperature"] = self._base_temperature

    def __get_fresh_llm_config(self):
        """
        Get a fresh LLM config with a new random seed.

        AutoGen caches LLM responses based on the seed value. Using the same
        seed with identical prompts returns cached responses, causing duplicate
        content. This method creates a new config with a unique seed for each call.

        Returns:
            dict: A copy of llm_config with a new random seed
        """
        import copy
        fresh_config = copy.deepcopy(self.llm_config)
        fresh_config["seed"] = np.random.randint(0, 100000)
        return fresh_config

    def _get_llm_config_for_write_action(self):
        """
        Return a fresh LLM config with per-round temperature ramping for writing actions.
        """
        fresh_config = self.__get_fresh_llm_config()
        temperature = min(
            self._base_temperature
            + (self.writing_actions_this_round * self._temperature_step),
            self._temperature_cap,
        )
        fresh_config["temperature"] = temperature
        return fresh_config

    def _record_writing_action(self):
        """Increment the per-round writing action counter."""
        self.writing_actions_this_round += 1

    def _consume_proactive_affect_budget(self) -> bool:
        """Consume one proactive-affect slot if under the round cap."""
        cap = int(getattr(self, "proactive_affect_cap_per_round", 2) or 2)
        used = int(getattr(self, "_proactive_affect_this_round", 0) or 0)
        if cap <= 0 or used >= cap:
            return False
        self._proactive_affect_this_round = used + 1
        return True

    def _normalize_post_text(self, text: str) -> str:
        if not text:
            return ""
        return re.sub(r"\s+", " ", text.strip().lower())

    def _normalize_comment_text(self, text: str) -> str:
        normalized = self._normalize_post_text(text)
        normalized = re.sub(r"[^\w\s]", "", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    def _comment_dedupe_key(self, text: str) -> str:
        normalized = self._normalize_comment_text(text)
        if not normalized:
            return ""
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()

    def _has_recent_identical_comment(self, tid: int, parent_post_id: int, text: str) -> bool:
        key = (int(tid), int(parent_post_id))
        normalized = self._normalize_comment_text(text)
        if not normalized:
            return False
        seen = self._recent_comments_by_round_parent.get(key, set())
        return normalized in seen

    def _record_recent_comment(self, tid: int, parent_post_id: int, text: str):
        key = (int(tid), int(parent_post_id))
        normalized = self._normalize_comment_text(text)
        if not normalized:
            return
        if key not in self._recent_comments_by_round_parent:
            self._recent_comments_by_round_parent[key] = set()
        self._recent_comments_by_round_parent[key].add(normalized)

    def _sentence_count(self, text: str) -> int:
        if not isinstance(text, str) or not text.strip():
            return 0
        chunks = re.split(r"[.!?]+", text)
        return len([c for c in chunks if re.search(r"\w", c or "")])

    def _trim_text_to_limits(
        self, text: str, *, max_chars: int, max_sentences: int
    ) -> str:
        if not isinstance(text, str):
            return ""
        out = re.sub(r"\s+", " ", text).strip()
        if not out:
            return ""

        if max_sentences > 0:
            parts = [
                p.strip()
                for p in re.split(r"(?<=[.!?])\s+", out)
                if isinstance(p, str) and p.strip()
            ]
            if parts and len(parts) > max_sentences:
                out = " ".join(parts[:max_sentences]).strip()

        if max_chars > 0 and len(out) > max_chars:
            clipped = out[:max_chars].rstrip()
            if " " in clipped:
                clipped = clipped.rsplit(" ", 1)[0]
            out = clipped.strip() or out[:max_chars].strip()
        return out

    def _is_text_over_limits(
        self, text: str, *, max_chars: int, max_sentences: int
    ) -> bool:
        if not isinstance(text, str):
            return False
        if max_chars > 0 and len(text) > max_chars:
            return True
        if max_sentences > 0 and self._sentence_count(text) > max_sentences:
            return True
        return False

    def _rewrite_text_to_limits(
        self,
        *,
        draft_text: str,
        mode: str,
        max_chars: int,
        max_sentences: int,
        context_text: str = "",
        interests=None,
    ) -> str:
        if not isinstance(self.prompts, dict):
            return draft_text or ""
        prompt = self.prompts.get("handler_reply_length_rewrite")
        if not isinstance(prompt, str) or not prompt.strip():
            return draft_text or ""

        cfg = self._get_llm_config_for_write_action()
        cfg["temperature"] = min(0.7, float(cfg.get("temperature", 0.6) or 0.6))
        cfg["max_tokens"] = int(max(120, min(280, max_chars)))
        interests = interests if isinstance(interests, list) else []

        u1 = AssistantAgent(
            name=f"{self.name}",
            llm_config=cfg,
            system_message=self.__effify(
                self.prompts.get(
                    "agent_roleplay_comments_share",
                    self.prompts.get("agent_roleplay_simple", ""),
                ),
                interests=interests,
            ),
            max_consecutive_auto_reply=1,
        )
        u2 = AssistantAgent(
            name="LengthRewrite",
            llm_config=cfg,
            system_message=self.__effify(self.prompts.get("handler_instructions", "")),
            max_consecutive_auto_reply=0,
        )

        rewritten = ""
        try:
            u2.initiate_chat(
                u1,
                message=self.__effify(
                    prompt,
                    mode=str(mode or "comment"),
                    draft_text=(draft_text or "").strip(),
                    max_chars=int(max_chars),
                    max_sentences=int(max_sentences),
                    context_text=self._memory_truncate(context_text or "", 1600),
                ),
                silent=True,
                max_turns=1,
            )
            rewritten = self._extract_generated_chat_content(
                u2,
                u1,
                prompt_hint=prompt,
                skip_emotion_like=True,
            )
        except Exception:
            rewritten = ""
        finally:
            try:
                u1.reset()
                u2.reset()
            except Exception:
                pass

        rewritten = self.__clean_text(rewritten)
        if not rewritten:
            return draft_text or ""
        return rewritten

    def _normalize_comment_for_repetition(self, text: str) -> str:
        norm = self._normalize_comment_text(text)
        return re.sub(r"\s+", " ", norm).strip()

    def _record_generated_comment(self, text: str):
        if not isinstance(text, str) or not text.strip():
            return
        norm = self._normalize_comment_for_repetition(text)
        if not norm:
            return
        self._recent_generated_comments.append(norm)
        max_keep = int(
            max(3, getattr(self, "anti_repetition_window_comments", 6) or 6)
        )
        if len(self._recent_generated_comments) > max_keep:
            self._recent_generated_comments = self._recent_generated_comments[-max_keep:]

    def _looks_repetitive_comment(self, text: str) -> bool:
        if not getattr(self, "anti_repetition_enabled", True):
            return False
        if not isinstance(text, str) or not text.strip():
            return False
        norm = self._normalize_comment_for_repetition(text)
        if not norm:
            return False
        history = list(getattr(self, "_recent_generated_comments", []) or [])
        if not history:
            return False

        if norm in history:
            return True

        tokens = norm.split()
        prefix = " ".join(tokens[:6])
        for prev in history:
            prev_tokens = prev.split()
            prev_prefix = " ".join(prev_tokens[:6])
            if prefix and prefix == prev_prefix:
                return True
            overlap = set(tokens[:18]).intersection(set(prev_tokens[:18]))
            if len(overlap) >= 8:
                return True
        return False

    def _is_thread_duplicate(self, text: str, thread_posts: list) -> bool:
        """Check if the proposed comment is a near-duplicate of any recent comment in the same thread."""
        if not isinstance(text, str) or not text.strip():
            return False
        if not thread_posts:
            return False
        norm = self._normalize_comment_for_repetition(text)
        if not norm:
            return False
        tokens = norm.split()
        if len(tokens) < 2:
            return False
        # Check against the last N comments in the thread (most recent, not all)
        check_limit = min(len(thread_posts), 30)
        for p in thread_posts[-check_limit:]:
            try:
                other_text = str(p) if isinstance(p, str) else str(p.get("text", "") if isinstance(p, dict) else p)
            except Exception:
                continue
            other_norm = self._normalize_comment_for_repetition(other_text)
            if not other_norm:
                continue
            if norm == other_norm:
                return True
            other_tokens = other_norm.split()
            # Check first-6-word prefix match
            if " ".join(tokens[:6]) == " ".join(other_tokens[:6]) and len(tokens[:6]) >= 3:
                return True
            # Check high token overlap (>= 60% of shorter text)
            if len(tokens) >= 4 and len(other_tokens) >= 4:
                overlap = set(tokens[:18]).intersection(set(other_tokens[:18]))
                threshold = min(8, int(0.6 * min(len(tokens[:18]), len(other_tokens[:18]))))
                if len(overlap) >= max(threshold, 4):
                    return True
        return False

    def _enforce_text_limits(
        self,
        *,
        text: str,
        mode: str,
        context_text: str = "",
        interests=None,
    ):
        out = self.__clean_text(text or "")
        if not getattr(self, "reply_length_enforcement_enabled", True):
            return out, {
                "mode": mode,
                "rewrite_used": False,
                "trim_fallback_used": False,
            }

        if str(mode) == "post":
            max_chars = int(getattr(self, "post_max_chars", 420) or 420)
            max_sentences = int(getattr(self, "post_max_sentences", 3) or 3)
        else:
            max_chars = int(getattr(self, "comment_max_chars", 220) or 220)
            max_sentences = int(getattr(self, "comment_max_sentences", 2) or 2)

        rewrite_used = False
        trim_fallback_used = False
        repetitive_flag = False

        over_limits = self._is_text_over_limits(
            out, max_chars=max_chars, max_sentences=max_sentences
        )
        if str(mode) == "comment":
            repetitive_flag = self._looks_repetitive_comment(out)

        rewrite_attempts = int(getattr(self, "reply_rewrite_max_attempts", 1) or 0)
        if (over_limits or repetitive_flag) and rewrite_attempts > 0:
            rewrite_used = True
            out = self._rewrite_text_to_limits(
                draft_text=out,
                mode=mode,
                max_chars=max_chars,
                max_sentences=max_sentences,
                context_text=context_text,
                interests=interests,
            )

        still_over = self._is_text_over_limits(
            out, max_chars=max_chars, max_sentences=max_sentences
        )
        still_repetitive = str(mode) == "comment" and self._looks_repetitive_comment(out)
        if (still_over or still_repetitive) and getattr(
            self, "reply_trim_fallback_enabled", True
        ):
            trim_fallback_used = True
            out = self._trim_text_to_limits(
                out, max_chars=max_chars, max_sentences=max_sentences
            )
            if str(mode) == "comment" and self._looks_repetitive_comment(out):
                tokens = out.split()
                if len(tokens) > 8:
                    out = " ".join(tokens[2:]).strip()
                    out = self._trim_text_to_limits(
                        out, max_chars=max_chars, max_sentences=max_sentences
                    )

        meta = {
            "mode": str(mode or "comment"),
            "pre_chars": len(text or "") if isinstance(text, str) else 0,
            "post_chars": len(out or ""),
            "pre_sentences": self._sentence_count(text or ""),
            "post_sentences": self._sentence_count(out or ""),
            "max_chars": max_chars,
            "max_sentences": max_sentences,
            "rewrite_used": bool(rewrite_used),
            "trim_fallback_used": bool(trim_fallback_used),
            "repetitive_input": bool(repetitive_flag),
        }
        return out, meta

    def _parse_structured_post_text(self, text: str):
        if not isinstance(text, str):
            return "", ""
        raw = text.strip()
        if not raw:
            return "", ""

        match = re.match(
            r"^[\*_]{0,2}(TITLE|TITTLE|TITEL)\s*:\s*",
            raw,
            flags=re.IGNORECASE,
        )
        if match:
            remaining = raw[match.end():]
            lines = remaining.split("\n", 1)
            title = re.sub(r"[\*_]{1,2}$", "", lines[0]).strip()
            body = lines[1].lstrip() if len(lines) > 1 else ""
        else:
            title = ""
            body = raw
            blocks = re.split(r"\n\s*\n", raw, maxsplit=1)
            if len(blocks) > 1:
                title = blocks[0].strip()
                body = blocks[1].lstrip()

        body = re.sub(
            r"^[\*_]{0,2}(TITLE|TITTLE|TITEL)\s*:\s*",
            "",
            body,
            flags=re.IGNORECASE,
        ).lstrip()
        return title.strip(), body.strip()

    def _validate_structured_post_text(self, text: str):
        title, body = self._parse_structured_post_text(text)
        valid = True
        reasons = []

        if not title:
            valid = False
            reasons.append("missing_title")
        if not body:
            valid = False
            reasons.append("missing_body")
        if title and body and title.strip().lower() == body.strip().lower():
            valid = False
            reasons.append("title_body_identical")
        if body and self._is_text_over_limits(
            body,
            max_chars=int(getattr(self, "post_max_chars", 420) or 420),
            max_sentences=int(getattr(self, "post_max_sentences", 3) or 3),
        ):
            valid = False
            reasons.append("body_over_limits")

        return {
            "valid": bool(valid),
            "title": title,
            "body": body,
            "reasons": reasons,
        }

    def _repair_structured_post_text(self, draft_text: str, interests=None):
        if not isinstance(self.prompts, dict):
            return draft_text or ""
        prompt = self.prompts.get("handler_post_structure_repair")
        if not isinstance(prompt, str) or not prompt.strip():
            return draft_text or ""

        interests = interests if isinstance(interests, list) else []
        cfg = self._get_llm_config_for_write_action()
        cfg["temperature"] = min(0.5, float(cfg.get("temperature", 0.6) or 0.6))
        cfg["max_tokens"] = int(max(200, min(420, getattr(self, "post_max_chars", 420) or 420)))

        u1 = AssistantAgent(
            name=f"{self.name}",
            llm_config=cfg,
            system_message=self.__effify(
                self.prompts.get("agent_roleplay", self.prompts.get("agent_roleplay_simple", "")),
                interests=interests,
            ),
            max_consecutive_auto_reply=1,
        )
        u2 = AssistantAgent(
            name="PostRepair",
            llm_config=cfg,
            system_message=self.__effify(self.prompts.get("handler_instructions", "")),
            max_consecutive_auto_reply=0,
        )

        rewritten = ""
        try:
            u2.initiate_chat(
                u1,
                message=self.__effify(
                    prompt,
                    draft_text=(draft_text or "").strip(),
                    interests=interests,
                    max_chars=int(getattr(self, "post_max_chars", 420) or 420),
                    max_sentences=int(getattr(self, "post_max_sentences", 3) or 3),
                ),
                silent=True,
                max_turns=1,
            )
            rewritten = self._extract_generated_chat_content(
                u2,
                u1,
                prompt_hint=prompt,
                skip_emotion_like=True,
            )
        except Exception:
            rewritten = ""
        finally:
            try:
                u1.reset()
                u2.reset()
            except Exception:
                pass

        rewritten = self.__clean_text(rewritten)
        return rewritten or (draft_text or "")

    def _build_comment_client_action_id(self, tid: int, parent_post_id: int, text: str) -> str:
        dedupe_key = self._comment_dedupe_key(text)
        run_id = str(getattr(self, "memory_run_id", "") or "no_run")
        base = f"{run_id}|{int(self.user_id)}|{int(tid)}|{int(parent_post_id)}|{dedupe_key}"
        digest = hashlib.sha1(base.encode("utf-8")).hexdigest()
        return f"cmt_{digest[:40]}"

    def _is_recent_duplicate(self, text: str) -> bool:
        norm = self._normalize_post_text(text)
        return norm in self._recent_posts

    def __record_post(self, text: str):
        norm = self._normalize_post_text(text)
        if not norm:
            return
        self._recent_posts.append(norm)
        # Keep only the last N posts to bound memory
        if len(self._recent_posts) > 50:
            self._recent_posts = self._recent_posts[-50:]

    def __effify(self, non_f_str: str, **kwargs):
        """
        Effify the string.

        :param non_f_str: the string to effify
        :param kwargs: the keyword arguments
        :return: the effified string
        """
        # Ensure both interest and interests variables are available in context
        # to handle any template that might use either one
        if "interests" in kwargs and "interest" not in kwargs:
            kwargs["interest"] = kwargs["interests"]
        elif "interest" in kwargs and "interests" not in kwargs:
            kwargs["interests"] = kwargs["interest"]
        elif "interest" not in kwargs and "interests" not in kwargs:
            kwargs["interest"] = []
            kwargs["interests"] = []

        kwargs["self"] = self
        rendered = eval(f'f"""{non_f_str}"""', kwargs)
        if isinstance(rendered, str):
            rendered = f"{rendered.rstrip()}\n\n{NO_EM_EN_DASH_PROMPT_RULE}"
        return rendered

    def set_prompts(self, prompts):
        """
        Set the LLM prompts.

        :param prompts: the prompts
        """
        self.prompts = prompts

        try:
            # if the agent has custom prompts substitute the default ones
            aprompt = session.query(Agent_Custom_Prompt).filter_by(agent_name=self.name).first()
            if aprompt:
                vibe_suffix = (
                    " Subreddit vibe: {self.subreddit_vibe}. Stay casual + on-topic; "
                    "avoid unrelated personal life and real-world politics/history unless the thread/vibe explicitly calls for it."
                )
                suffix = f"{vibe_suffix} - Act as requested by the Handler."
                self.prompts["agent_roleplay"] = f"{aprompt.prompt}{suffix}"
                self.prompts["agent_roleplay_simple"] = f"{aprompt.prompt}{suffix}"
                self.prompts["agent_roleplay_base"] = f"{aprompt.prompt}{suffix}"
                self.prompts["agent_roleplay_comments_share"] = f"{aprompt.prompt}{suffix}"
        except:
            pass

    def set_rec_sys(self, content_recsys, follow_recsys):
        """
        Set the recommendation systems.

        :param content_recsys: the content recommendation system
        :param follow_recsys: the follow recommendation system
        """
        if self.content_rec_sys is None:
            self.content_rec_sys = content_recsys
            self.content_rec_sys.add_user_id(self.user_id)
            self.content_rec_sys_name = content_recsys.name

            api_url = f"{self.base_url}update_user"

            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            params = {
                "username": self.name,
                "email": self.email,
                "recsys_type": content_recsys.name,
            }
            st = json.dumps(params)
            post(f"{api_url}", headers=headers, data=st)

        if self.follow_rec_sys is None:
            self.follow_rec_sys = follow_recsys
            self.follow_rec_sys.add_user_id(self.user_id)
            self.follow_rec_sys_name = follow_recsys.name

            api_url = f"{self.base_url}update_user"

            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            params = {
                "username": self.name,
                "email": self.email,
                "frecsys_type": follow_recsys.name,
            }
            st = json.dumps(params)
            post(f"{api_url}", headers=headers, data=st)

        return {"status": 200}

    def __extract_components(self, text, c_type="hashtags"):
        """
        Extract the components from the text.

        :param text: the text to extract the components from
        :param c_type: the component type
        :return: the extracted components
        """
        # Define the regex pattern
        if c_type == "hashtags":
            pattern = re.compile(r"#\w+")
        elif c_type == "mentions":
            pattern = re.compile(r"@\w+")
        else:
            return []
        # Find all matches in the input text
        hashtags = pattern.findall(text)
        return hashtags

    def __get_user(self):
        """
        Get the user from the service.

        :return: the user
        """
        res = json.loads(self._check_credentials())
        if res["status"] == 404:
            raise Exception("User not found")
        api_url = f"{self.base_url}get_user"

        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        params = {"username": self.name, "email": self.email}
        st = json.dumps(params)

        response = post(f"{api_url}", headers=headers, data=st)

        return response.__dict__["_content"].decode("utf-8")

    def _check_credentials(self):
        """
        Check if the credentials are correct.

        :return: the response from the service
        """
        api_url = f"{self.base_url}user_exists"

        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        params = {"name": self.name, "email": self.email}

        st = json.dumps(params)
        response = post(f"{api_url}", headers=headers, data=st)

        return response.__dict__["_content"].decode("utf-8")

    def __register(self):
        """
        Register the agent to the service.

        :return: the response from the service
        """

        st = json.dumps(
            {
                "name": self.name,
                "email": self.email,
                "password": self.pwd,
                "leaning": self.leaning,
                "age": self.age,
                "user_type": self.type,
                "oe": self.oe,
                "co": self.co,
                "ex": self.ex,
                "ag": self.ag,
                "ne": self.ne,
                "language": self.language,
                "owner": self.owner,
                "education_level": self.education_level,
                "round_actions": self.round_actions,
                "gender": self.gender,
                "nationality": self.nationality,
                "toxicity": self.toxicity,
                "joined_on": self.joined_on,
                "is_page": self.is_page,
                "daily_activity_level": self.daily_activity_level,
                "profession": self.profession,
            }
        )

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        api_url = f"{self.base_url}/register"
        post(f"{api_url}", headers=headers, data=st)

        try:
            res = json.loads(self.__get_user())
            uid = int(res["id"])
        except Exception as e:
            print(f"Agent registration failed for {self.name}: {e}")
            return None

        api_url = f"{self.base_url}/set_user_interests"
        data = {"user_id": uid, "interests": self.interests, "round": self.joined_on}

        post(f"{api_url}", headers=headers, data=json.dumps(data))

        return uid

    def __get_interests(self, tid):
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        # current round
        if tid == -1:
            # get last round id
            api_url = f"{self.base_url}/current_time"
            response = get(f"{api_url}", headers=headers)
            data = json.loads(response.__dict__["_content"].decode("utf-8"))
            tid = int(data["id"])

        api_url = f"{self.base_url}/get_user_interests"

        data = {
            "user_id": self.user_id,
            "round_id": tid,
            "n_interests": self.interests if isinstance(self.interests, int) else len(self.interests),
            "time_window": self.attention_window,
        }
        response = get(f"{api_url}", headers=headers, data=json.dumps(data))
        data = json.loads(response.__dict__["_content"].decode("utf-8"))
        try:
            # select a random interest without replacement
            if len(data) >= 3:
                selected = np.random.choice(range(len(data)), np.random.randint(1, 3), replace=False)
            else:
                selected = np.random.choice(range(len(data)), len(data), replace=False)

            interests = [data[i]["topic"] for i in selected]
            interests_id = [data[i]["id"] for i in selected]
        except:
            return [], []

        return interests, interests_id

    @log_execution_time
    def post(self, tid):
        """
        Post a message to the service.

        :param tid: the round id
        """

        # obtain the most recent (and frequent) interests of the agent
        interests, interests_id = self.__get_interests(tid)

        # Make sure we have valid interests to avoid template errors
        if not interests:
            interests = []
            interests_id = []
            print(f"Warning: No interests found for agent {self.name}. Using empty list.")

        # get recent sentiment on the selected interests
        api_url = f"{self.base_url}/get_sentiment"
        data = {"user_id": self.user_id, "interests": interests}
        response = post(f"{api_url}", headers={"Content-Type": "application/x-www-form-urlencoded"}, data=json.dumps(data))
        sentiment = json.loads(response.__dict__["_content"].decode("utf-8"))

        self.topics_opinions = "Your opinion on the topics you are interested in is: "
        for s in sentiment:
            self.topics_opinions += f"{s['topic']}: {s['sentiment']} "
        if len(sentiment) == 0:
            self.topics_opinions = ""

        # Handle subsequent posts in the same round for topic variation
        is_subsequent_post = self.posts_this_round > 0

        style_context_text = ""
        style_context_meta = {
            "usage": "none",
            "root_count": 0,
            "distinct_author_count": 0,
            "mature": False,
        }

        # Add variation prompt for subsequent posts to ensure different content
        post_prompt = self.__effify(self.prompts["handler_post"], interests=interests)
        if is_subsequent_post:
            post_prompt = post_prompt + f"\n\nIMPORTANT: This is post #{self.posts_this_round + 1} - write about a COMPLETELY DIFFERENT topic than before. Pick a different interest."

        # Community style guidance can influence structure and tone, but not the topic.
        try:
            if getattr(self, "memory_enabled", False):
                style_context_text, style_context_meta = self._memory_build_post_style_context(
                    tid=int(tid)
                )
                if style_context_text:
                    post_prompt = style_context_text + "\n\n" + post_prompt
        except Exception:
            pass

        try:
            self._decision_log(
                {
                    "decision_type": "post_prompt_memory_usage",
                    "run_id": getattr(self, "memory_run_id", None),
                    "agent_user_id": int(getattr(self, "user_id", -1) or -1),
                    "agent_name": getattr(self, "name", None),
                    "tid": int(tid),
                    "usage": style_context_meta.get("usage") or "none",
                    "root_count": int(style_context_meta.get("root_count") or 0),
                    "distinct_author_count": int(style_context_meta.get("distinct_author_count") or 0),
                    "mature": bool(style_context_meta.get("mature")),
                }
            )
        except Exception:
            pass

        max_attempts = 3
        post_text = ""
        emotion_eval = ""
        post_structure_meta = {
            "valid_first_pass": True,
            "repair_used": False,
            "repair_valid": True,
            "reasons": [],
        }
        novelty_avoid_titles = []
        for attempt in range(1, max_attempts + 1):
            fresh_config = self._get_llm_config_for_write_action()
            if attempt > 1:
                fresh_config["temperature"] = min(
                    fresh_config["temperature"] + (0.2 * (attempt - 1)),
                    self._temperature_cap,
                )

            u1 = AssistantAgent(
                name=f"{self.name}",
                llm_config=fresh_config,
                system_message=self.__effify(
                    self.prompts["agent_roleplay"], interests=interests
                ),
                max_consecutive_auto_reply=1,
            )

            u2 = AssistantAgent(
                name=f"Handler",
                llm_config=fresh_config,
                system_message=self.prompts["handler_instructions"],
                max_consecutive_auto_reply=1,
            )

            prompt = post_prompt
            if attempt > 1:
                prompt = (
                    post_prompt
                    + "\n\nIMPORTANT: Do NOT repeat any previous posts. "
                      "Write something meaningfully different with new phrasing."
                )
            if attempt > 1 and novelty_avoid_titles:
                prompt += (
                    "\n\nNOVELTY NOTE: avoid recycling the recent topic family around "
                    + ", ".join(novelty_avoid_titles[:2])
                    + ". Pick a different angle or different interest."
                )

            u2.initiate_chat(
                u1,
                message=prompt,
                silent=True,
                max_turns=1,
            )

            emotion_raw = self._extract_emotion_chat_content(u2, u1)
            emotion_eval = self.__clean_emotion(emotion_raw)

            post_text = self._extract_generated_chat_content(
                u2, u1, prompt_hint=prompt, skip_emotion_like=True
            )
            post_text = self.__clean_text(post_text)
            # Avoid markdown emphasis causing bold rendering in the UI.
            post_text = post_text.replace("*", "")
            post_text, length_meta = self._enforce_text_limits(
                text=post_text,
                mode="post",
                context_text=post_prompt,
                interests=interests,
            )
            try:
                self._decision_log(
                    {
                        "decision_type": "length_enforcement",
                        "run_id": getattr(self, "memory_run_id", None),
                        "agent_user_id": int(getattr(self, "user_id", -1) or -1),
                        "agent_name": getattr(self, "name", None),
                        "tid": int(tid),
                        "mode": "post",
                        **length_meta,
                    }
                )
            except Exception:
                pass

            if len(post_text) < 3:
                return

            if getattr(self, "forum_post_structure_strict", False):
                validation = self._validate_structured_post_text(post_text)
                post_structure_meta = {
                    "valid_first_pass": bool(validation.get("valid")),
                    "repair_used": False,
                    "repair_valid": bool(validation.get("valid")),
                    "reasons": list(validation.get("reasons") or []),
                }
                if not validation.get("valid"):
                    repaired = self._repair_structured_post_text(
                        post_text,
                        interests=interests,
                    )
                    repaired = repaired.replace("*", "")
                    repaired, _ = self._enforce_text_limits(
                        text=repaired,
                        mode="post",
                        context_text=post_prompt,
                        interests=interests,
                    )
                    repaired_validation = self._validate_structured_post_text(repaired)
                    post_structure_meta["repair_used"] = True
                    post_structure_meta["repair_valid"] = bool(
                        repaired_validation.get("valid")
                    )
                    post_structure_meta["reasons"] = list(
                        repaired_validation.get("reasons") or validation.get("reasons") or []
                    )
                    if repaired_validation.get("valid"):
                        post_text = repaired
                    else:
                        post_text = ""

                try:
                    self._decision_log(
                        {
                            "decision_type": "post_structure_validation",
                            "run_id": getattr(self, "memory_run_id", None),
                            "agent_user_id": int(getattr(self, "user_id", -1) or -1),
                            "agent_name": getattr(self, "name", None),
                            "tid": int(tid),
                            "attempt": int(attempt),
                            "post_structure_valid_first_pass": bool(
                                post_structure_meta.get("valid_first_pass")
                            ),
                            "post_structure_repair_used": bool(
                                post_structure_meta.get("repair_used")
                            ),
                            "post_structure_repair_valid": bool(
                                post_structure_meta.get("repair_valid")
                            ),
                            "reasons": list(post_structure_meta.get("reasons") or []),
                        }
                    )
                except Exception:
                    pass

                if len(post_text) < 3:
                    continue

            fingerprint, topic_matches = self._post_find_recent_topic_matches(
                text_value=post_text,
                tid=int(tid),
            )
            is_stale_topic = bool(topic_matches)
            try:
                self._decision_log(
                    {
                        "decision_type": "post_topic_freshness",
                        "run_id": getattr(self, "memory_run_id", None),
                        "agent_user_id": int(getattr(self, "user_id", -1) or -1),
                        "agent_name": getattr(self, "name", None),
                        "tid": int(tid),
                        "attempt": int(attempt),
                        "stale_topic": bool(is_stale_topic),
                        "topic_title": fingerprint.get("title") or "",
                        "topic_entities": fingerprint.get("entities") or [],
                        "recent_matches": topic_matches,
                    }
                )
            except Exception:
                pass
            if is_stale_topic:
                novelty_avoid_titles = [
                    match.get("title")
                    for match in topic_matches
                    if isinstance(match, dict) and str(match.get("title") or "").strip()
                ][:2]
                post_text = ""
                continue

            if not self._is_recent_duplicate(post_text):
                break
            logging.info(
                f"[{self.name}] Duplicate post detected (attempt {attempt}/{max_attempts}), regenerating..."
            )

        # avoid posting empty messages
        if len(post_text) < 3:
            return

        hashtags = self.__extract_components(post_text, c_type="hashtags")
        mentions = self.__extract_components(post_text, c_type="mentions")

        st = json.dumps(
            {
                "user_id": self.user_id,
                "tweet": post_text.replace('"', ""),
                "emotions": emotion_eval,
                "hashtags": hashtags,
                "mentions": mentions,
                "tid": tid,
                "topics": interests_id
            }
        )

        u1.reset()
        u2.reset()

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        api_url = f"{self.base_url}/post"
        post(f"{api_url}", headers=headers, data=st)
        self.__record_post(post_text)

        # Run-scoped memory: record that we authored a post and refresh community digest periodically.
        try:
            self._memory_after_post(tid=int(tid), post_text=post_text, topics=interests)
        except Exception:
            pass

        # update topic of interest with the ones used to generate the post
        api_url = f"{self.base_url}/set_user_interests"
        data = {"user_id": self.user_id, "interests": interests, "round": tid}
        post(f"{api_url}", headers=headers, data=json.dumps(data))

        # Increment post counter for this round
        self.posts_this_round += 1
        self._record_writing_action()

    def share_link(self, tid, article, website):
        """
        Share a link (article) with commentary.

        :param tid: the round id
        :param article: the article
        :param website: the website
        :return: the response from the service
        """
        fresh_config = self._get_llm_config_for_write_action()

        # Use the same handler_news prompt for link sharing
        u1 = AssistantAgent(
            name=f"{self.name}",
            llm_config=fresh_config,
            system_message=self.__effify(self.prompts["agent_roleplay_comments_share"]),
            max_consecutive_auto_reply=1,
        )

        u2 = AssistantAgent(
            name=f"Handler",
            llm_config=fresh_config,
            system_message=self.__effify(self.prompts["handler_instructions"]),
            max_consecutive_auto_reply=1,
        )

        link_prompt = self.__effify(
            self.prompts["handler_news"], website=website, article=article
        )
        u2.initiate_chat(
            u1,
            message=link_prompt,
            silent=True,
            max_turns=1,
        )

        emotion_raw = self._extract_emotion_chat_content(u2, u1)
        emotion_eval = self.__clean_emotion(emotion_raw)

        post_text = self._extract_generated_chat_content(
            u2, u1, prompt_hint=link_prompt, skip_emotion_like=True
        )
        post_text = self.__clean_text(post_text)

        # Strip reproduced article content from LLM output
        post_text = self._strip_reproduced_article_content(post_text, article.summary)
        # Avoid markdown emphasis causing bold rendering in the UI.
        post_text = post_text.replace("*", "")

        post_text, length_meta = self._enforce_text_limits(
            text=post_text,
            mode="post",
            context_text=link_prompt,
            interests=None,
        )
        try:
            self._decision_log(
                {
                    "decision_type": "length_enforcement",
                    "run_id": getattr(self, "memory_run_id", None),
                    "agent_user_id": int(getattr(self, "user_id", -1) or -1),
                    "agent_name": getattr(self, "name", None),
                    "tid": int(tid),
                    "subtype": "share_link",
                    **length_meta,
                }
            )
        except Exception:
            pass

        # Extract hashtags and mentions
        hashtags = self.__extract_components(post_text, c_type="hashtags")
        mentions = self.__extract_components(post_text, c_type="mentions")
        article_fetched_on = getattr(article, "published", None)
        if article_fetched_on is None:
            article_fetched_on = getattr(website, "last_fetched", None)
        try:
            article_fetched_on = int(article_fetched_on)
        except (TypeError, ValueError):
            article_fetched_on = getattr(website, "last_fetched", None)

        # Create payload for the server
        st = json.dumps(
            {
                "user_id": self.user_id,
                "tweet": post_text.replace('"', ""),
                "emotions": emotion_eval,
                "hashtags": hashtags,
                "mentions": mentions,
                "tid": tid,
                "title": article.title,
                "summary": article.summary,
                "link": article.link,
                "publisher": website.name,
                "rss": website.rss,
                "leaning": website.leaning,
                "country": website.country,
                "language": website.language,
                "category": website.category,
                "fetched_on": article_fetched_on,
                "image_url": getattr(article, 'image_url', None),
            }
        )

        u1.reset()
        u2.reset()

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        # Use the same /news endpoint for sharing links
        api_url = f"{self.base_url}/news"
        res = post(f"{api_url}", headers=headers, data=st)
        self._record_writing_action()
        return res

    def _caption_leaks_description(self, caption: str, description: str) -> bool:
        """
        Detect if the caption contains VLM description fragments.

        This prevents the LLM from simply paraphrasing the VLM description
        instead of generating a creative, emotional reaction.

        :param caption: Generated caption text
        :param description: VLM image description
        :return: True if leak detected
        """
        desc_lower = description.lower()
        caption_lower = caption.lower()

        # Check for 3-word phrase matches (significant overlap)
        words = desc_lower.split()
        for i in range(len(words) - 2):
            phrase = " ".join(words[i : i + 3])
            # Only check substantial phrases (>10 chars)
            if len(phrase) > 10 and phrase in caption_lower:
                return True

        # Check for obvious descriptive indicators
        leak_indicators = [
            "the image shows",
            "the image depicts",
            "in this image",
            "the picture shows",
            "this is an image of",
            "we can see",
            "there is a",
            "appears to be",
            "seems to show",
            "the photo shows",
            "the photo depicts",
            "this photo shows",
            "you can see",
            "depicting a",
            "shows a",
            "features a",
        ]
        return any(ind in caption_lower for ind in leak_indicators)

    def _strip_reproduced_article_content(self, post_text: str, article_summary: str) -> str:
        """
        Strip reproduced article content from LLM-generated post text.

        Uses Jaccard similarity to detect when the LLM has copied the article
        summary into its response, and removes the duplicated content.

        :param post_text: The LLM-generated post text
        :param article_summary: The original article summary
        :return: Post text with reproduced content removed
        """
        if not post_text or not article_summary:
            return post_text

        # Calculate Jaccard similarity
        def get_words(text):
            return set(re.findall(r'\b\w+\b', text.lower()))

        post_words = get_words(post_text)
        summary_words = get_words(article_summary)

        if not post_words or not summary_words:
            return post_text

        # Always filter sentences to remove reproduced content
        # Split into sentences and filter out those too similar to article
        sentences = re.split(r'(?<=[.!?])\s+', post_text)
        filtered_sentences = []

        for sentence in sentences:
            sentence_words = get_words(sentence)
            if sentence_words:
                # Check overlap ratio with article
                overlap_ratio = len(sentence_words & summary_words) / len(sentence_words)
                sentence_similarity = len(sentence_words & summary_words) / len(sentence_words | summary_words) if (sentence_words | summary_words) else 0

                # Keep sentences that are sufficiently different (lowered thresholds)
                # sentence_similarity < 0.20 (was 0.36)
                # overlap_ratio < 0.45 (was 0.7)
                if sentence_similarity < 0.20 and overlap_ratio < 0.45:
                    filtered_sentences.append(sentence)

        if filtered_sentences:
            return ' '.join(filtered_sentences)
        else:
            # If all sentences were filtered, return a minimal response
            return ""

    def share_image(self, tid: int, image_post: ImagePosts):
        """
        Share a standalone image with commentary.

        :param tid: the round id
        :param image_post: the ImagePosts record
        :return: the response from the service
        """
        fresh_config = self._get_llm_config_for_write_action()
        released = False

        try:
            from y_client.clients.client_web import session as global_session

            image_session = global_session or session
        except ImportError:
            image_session = session

        def _release_image():
            nonlocal released
            if released or image_post is None or image_session is None:
                return
            try:
                image_post.used = False
                image_session.commit()
            except Exception:
                try:
                    image_session.rollback()
                except Exception:
                    pass
            released = True

        try:
            description = image_post.description or "An image"

            u1 = AssistantAgent(
                name=f"{self.name}",
                llm_config=fresh_config,
                system_message=self.__effify(self.prompts["agent_roleplay_comments_share"]),
                max_consecutive_auto_reply=1,
            )

            u2 = AssistantAgent(
                name=f"Handler",
                llm_config=fresh_config,
                system_message=self.__effify(self.prompts["handler_instructions"]),
                max_consecutive_auto_reply=1,
            )

            image_prompt = self.__effify(
                self.prompts["handler_image_post"], description=description
            )
            u2.initiate_chat(
                u1,
                message=image_prompt,
                silent=True,
                max_turns=1,
            )

            emotion_raw = self._extract_emotion_chat_content(u2, u1)
            emotion_eval = self.__clean_emotion(emotion_raw)

            post_text = self._extract_generated_chat_content(
                u2, u1, prompt_hint=image_prompt, skip_emotion_like=True
            )
            post_text = self.__clean_text(post_text)
            if len(post_text) < 3:
                _release_image()
                return None

            if self._is_recent_duplicate(post_text):
                logging.info(
                    f"[{self.name}] Skipping duplicate image post: '{post_text[:50]}...'"
                )
                _release_image()
                return None

            if self._caption_leaks_description(post_text, description):
                logging.info(f"[{self.name}] Caption leak detected, regenerating...")
                u1.reset()
                u2.reset()

                retry_prompt = (
                    "WRITE A NEW CAPTION. Your previous attempt described the image "
                    "instead of reacting to it emotionally. Write a SHORT reaction "
                    "(not a description). DO NOT say what's in the image. "
                    "Just react emotionally in 1 sentence, max 10 words. "
                    "Examples: 'mood', 'me rn', 'this hits different', 'absolute legend'.\n\n"
                    f"Previous bad attempt: {post_text}"
                )

                u2.initiate_chat(
                    u1,
                    message=retry_prompt,
                    silent=True,
                    max_turns=1,
                )

                post_text = self._extract_generated_chat_content(
                    u2, u1, prompt_hint=retry_prompt, skip_emotion_like=True
                )
                post_text = self.__clean_text(post_text)
                if len(post_text) < 3 or self._caption_leaks_description(
                    post_text, description
                ):
                    _release_image()
                    return None
                if self._is_recent_duplicate(post_text):
                    logging.info(
                        f"[{self.name}] Skipping duplicate image post after retry: '{post_text[:50]}...'"
                    )
                    _release_image()
                    return None

            hashtags = self.__extract_components(post_text, c_type="hashtags")
            mentions = self.__extract_components(post_text, c_type="mentions")

            st = json.dumps(
                {
                    "user_id": self.user_id,
                    "tweet": post_text.replace('"', ""),
                    "image_url": image_post.url,
                    "image_description": description,
                    "emotions": emotion_eval,
                    "hashtags": hashtags,
                    "mentions": mentions,
                    "tid": tid,
                }
            )

            u1.reset()
            u2.reset()

            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            api_url = f"{self.base_url}/image_post"
            response = post(f"{api_url}", headers=headers, data=st)

            success = response.status_code == 200
            if success:
                try:
                    payload = json.loads(response.__dict__["_content"].decode("utf-8"))
                    success = payload.get("status") == 200
                except Exception:
                    success = True
            if not success:
                _release_image()
                return response

            self._record_writing_action()
            self.__record_post(post_text)
            return response
        except Exception:
            _release_image()
            logging.exception("Standalone image share failed for %s", self.name)
            return None

    def _annotate_image_if_needed(self, image, current_session):
        """
        Annotate image with description if missing, using vision LLM.

        :param image: ImagePosts record to annotate
        :param current_session: Database session
        :return: The image record if it has a description, None otherwise
        """
        if image.description is not None:
            return image

        try:
            an = Annotator(config=self.llm_v_config)
            description = an.annotate(image.url)
            if description:
                image.description = description
                current_session.commit()
                logging.info(f"[{self.name}] Annotated image {image.id}: {description[:50]}...")
                return image
        except Exception as e:
            logging.warning(f"[{self.name}] Failed to annotate image {image.id}: {e}")
            current_session.rollback()

        # No description - don't share this image
        return None

    def select_standalone_image(self, tid: int = None):
        """
        Select an unused standalone image from ImagePosts, matching agent interests.
        Excludes images this agent has already posted (per-user duplicate prevention).
        Will annotate images on-demand if they don't have descriptions.

        :param tid: optional round id used for decision logging
        :return: ImagePosts record or None
        """
        try:
            tid_for_log = int(tid) if tid is not None else None
        except Exception:
            tid_for_log = None

        def _image_to_log(image_obj, source_stage: str):
            if image_obj is None:
                return None
            return {
                "image_post_id": int(getattr(image_obj, "id", -1))
                if getattr(image_obj, "id", None) is not None
                else None,
                "subreddit": self._decision_compact_text(getattr(image_obj, "subreddit", ""), 64),
                "title": self._decision_compact_text(getattr(image_obj, "title", ""), 120),
                "description_snippet": self._decision_compact_text(
                    getattr(image_obj, "description", ""), 220
                ),
                "image_url": self._decision_compact_text(getattr(image_obj, "url", ""), 220),
                "source": source_stage,
            }

        def _log_and_return(selected_image, selection_mode: str, fallback_reason: str = ""):
            selected_for_log = _image_to_log(selected_image, selection_mode) if selected_image else None
            try:
                self._decision_log(
                    {
                        "decision_type": "image_share_candidate_decision",
                        "run_id": getattr(self, "memory_run_id", None),
                        "agent_user_id": int(getattr(self, "user_id", -1) or -1),
                        "agent_name": getattr(self, "name", None),
                        "tid": tid_for_log,
                        "selection_mode": selection_mode,
                        "fallback_reason": fallback_reason,
                        "interests": interests_for_log,
                        "matching_subreddits": matching_subreddits_for_log,
                        "persona": self._decision_persona_snapshot(),
                        "candidates": candidates_for_log[:6],
                        "selected": selected_for_log,
                    }
                )
            except Exception:
                pass
            return selected_image

        def _reserve_image(selected_image, selection_mode: str, fallback_reason: str = ""):
            if selected_image is None:
                return _log_and_return(None, selection_mode, fallback_reason)
            selected_image.used = True
            try:
                current_session.commit()
            except Exception as exc:
                logging.warning(
                    "[%s] Could not reserve image %s: %s",
                    self.name,
                    getattr(selected_image, "id", None),
                    exc,
                )
                current_session.rollback()
                return _log_and_return(None, "error", "image_reservation_failed")
            return _log_and_return(selected_image, selection_mode, fallback_reason)

        # Get the session from global scope (same pattern as select_link)
        try:
            from y_client.clients.client_web import session as global_session
            if global_session is not None:
                current_session = global_session
            else:
                current_session = session
        except ImportError:
            current_session = session

        interests_for_log = []
        matching_subreddits_for_log = []
        candidates_for_log = []

        if current_session is None:
            logging.warning("Database session is None in select_standalone_image")
            return _log_and_return(None, "error", "db_session_none")

        # Get image_post_ids this agent has already posted (per-user duplicate check)
        try:
            result = current_session.execute(
                text("SELECT image_post_id FROM post WHERE user_id = :uid AND image_post_id IS NOT NULL"),
                {"uid": self.user_id}
            )
            my_image_post_ids = {row[0] for row in result.fetchall()}
        except Exception as e:
            logging.warning(f"Could not check agent's posted images: {e}")
            current_session.rollback()
            my_image_post_ids = set()

        # Get agent's interests
        interests = self.interests if isinstance(self.interests, list) else []
        interests_for_log = [self._decision_compact_text(v, 48) for v in interests[:10]]

        # Try to load interest_map from config if available
        # This maps interests to subreddits
        interest_map = getattr(self, 'image_interest_map', {})

        # Find matching subreddits based on interests
        matching_subreddits = set()
        for interest in interests:
            interest_lower = interest.lower() if isinstance(interest, str) else str(interest).lower()
            for key, subreddits in interest_map.items():
                if interest_lower in key.lower() or key.lower() in interest_lower:
                    matching_subreddits.update(subreddits)
        matching_subreddits_for_log = sorted(list(matching_subreddits))[:10]

        # Try to get unused image from matching subreddits (not already posted by this agent)
        if matching_subreddits:
            # First try: prefer images that already have descriptions
            query = current_session.query(ImagePosts).filter(
                ImagePosts.used == False,
                ImagePosts.description.isnot(None),
                ImagePosts.subreddit.in_(list(matching_subreddits)),
            )
            if my_image_post_ids:
                query = query.filter(~ImagePosts.id.in_(my_image_post_ids))
            candidates = query.order_by(func.random()).limit(6).all()
            candidates_for_log.extend(
                [_image_to_log(img, "interest_match_with_description") for img in candidates]
            )
            if candidates:
                image = random.choice(candidates)
                return _reserve_image(image, "interest_match_with_description")

            # Second try: images without descriptions (will annotate on-demand)
            query = current_session.query(ImagePosts).filter(
                ImagePosts.used == False,
                ImagePosts.subreddit.in_(list(matching_subreddits)),
            )
            if my_image_post_ids:
                query = query.filter(~ImagePosts.id.in_(my_image_post_ids))
            candidates = query.order_by(func.random()).limit(6).all()
            candidates_for_log.extend(
                [_image_to_log(img, "interest_match_needs_annotation") for img in candidates]
            )
            for image in candidates:
                annotated = self._annotate_image_if_needed(image, current_session)
                if annotated:
                    return _reserve_image(annotated, "interest_match_needs_annotation")
            # Annotation failed for shortlist, continue to fallback

        # Fallback: any unused image with description (not already posted by this agent)
        query = current_session.query(ImagePosts).filter(
            ImagePosts.used == False,
            ImagePosts.description.isnot(None)
        )
        if my_image_post_ids:
            query = query.filter(~ImagePosts.id.in_(my_image_post_ids))
        candidates = query.order_by(func.random()).limit(6).all()
        candidates_for_log.extend([_image_to_log(img, "fallback_with_description") for img in candidates])
        if candidates:
            image = random.choice(candidates)
            return _reserve_image(image, "fallback_with_description")

        # Final fallback: any unused image without description (will annotate on-demand)
        query = current_session.query(ImagePosts).filter(
            ImagePosts.used == False
        )
        if my_image_post_ids:
            query = query.filter(~ImagePosts.id.in_(my_image_post_ids))
        candidates = query.order_by(func.random()).limit(6).all()
        candidates_for_log.extend([_image_to_log(img, "fallback_needs_annotation") for img in candidates])
        for image in candidates:
            annotated = self._annotate_image_if_needed(image, current_session)
            if annotated:
                return _reserve_image(annotated, "fallback_needs_annotation")

        # No unused images with descriptions available
        return _log_and_return(None, "none_available", "no_candidate_image")

    def __get_thread(self, post_id: int, max_tweets=None):
        """
        Get the thread of a post.

        :param post_id: The post id to get the thread.
        :param max_tweets: The maximum number of tweets to read for context.
        """
        api_url = f"{self.base_url}/post_thread"

        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        params = {"post_id": post_id}
        st = json.dumps(params)
        response = post(f"{api_url}", headers=headers, data=st)

        res = json.loads(response.__dict__["_content"].decode("utf-8"))

        if max_tweets is not None and len(res) > max_tweets:
            return res[-max_tweets:]

        return res

    def __get_thread_tree(self, post_id: int, limit: int = None):
        """
        Fetch a structured thread representation for sequential traversal (tree order).

        Server response schema (200):
          {
            "status": 200,
            "thread_root_id": <int>,
            "posts": [
              {"post_id":int,"comment_to":int|null,"user_id":int,"username":str,
               "text":str,"round":int|null,"reaction_count":int}
            ]
          }
        """
        api_url = f"{self.base_url}/get_thread_tree"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        if limit is None:
            limit = getattr(self, "thread_browse_max_nodes", 400) or 400
        st = json.dumps({"post_id": int(post_id), "limit": int(limit)})
        response = post(f"{api_url}", headers=headers, data=st)
        return json.loads(response.__dict__["_content"].decode("utf-8"))

    def _thread_tree_dfs_order(self, posts: list, thread_root_id: int):
        """
        Build a deterministic DFS traversal order from a flat post list.

        Returns a list of tuples: (node_dict, depth)
        """
        by_id = {}
        order_ids = []
        for p in posts or []:
            try:
                pid = int(p.get("post_id"))
            except Exception:
                continue
            if pid in by_id:
                continue
            by_id[pid] = p
            order_ids.append(pid)

        root_id = int(thread_root_id) if thread_root_id is not None else None
        if root_id is None:
            root_id = order_ids[0] if order_ids else None
        if root_id is None:
            return []

        children = {}
        for pid in order_ids:
            if pid == root_id:
                continue
            parent = by_id.get(pid, {}).get("comment_to", None)
            if parent in (-1, None):
                parent = root_id
            try:
                parent = int(parent)
            except Exception:
                parent = root_id
            if parent not in by_id and parent != root_id:
                parent = root_id
            children.setdefault(parent, []).append(pid)

        # Ensure deterministic ordering: children follow the DB order (Post.id asc).
        visited = set()
        out = []

        def dfs(cur_id: int, depth: int):
            if cur_id in visited:
                return
            visited.add(cur_id)
            node = by_id.get(cur_id)
            if node is not None:
                out.append((node, depth))
            for child_id in children.get(cur_id, []):
                dfs(child_id, depth + 1)

        dfs(root_id, 0)

        # Include any stragglers (e.g., if we hit a partial limit).
        for pid in order_ids:
            if pid not in visited:
                node = by_id.get(pid)
                if node is not None:
                    out.append((node, 0))

        return out

    def _build_thread_analysis_maps(self, posts: list, thread_root_id: int):
        node_by_id = {}
        children_by_parent = {}

        try:
            root_id = int(thread_root_id) if thread_root_id is not None else None
        except Exception:
            root_id = None

        for p in posts or []:
            try:
                pid = int((p or {}).get("post_id"))
            except Exception:
                continue
            if pid in node_by_id:
                continue
            node_by_id[pid] = p

        for pid, node in node_by_id.items():
            if root_id is not None and pid == root_id:
                continue
            parent = node.get("comment_to", root_id)
            if parent in (-1, None):
                parent = root_id
            try:
                parent = int(parent) if parent is not None else root_id
            except Exception:
                parent = root_id
            if parent not in node_by_id and parent != root_id:
                parent = root_id
            children_by_parent.setdefault(parent, []).append(pid)

        ordered_pairs = self._thread_tree_dfs_order(posts, thread_root_id)
        ordered_nodes = []
        ordered_index = {}
        depth_by_post_id = {}
        for idx, (node, depth) in enumerate(ordered_pairs):
            try:
                pid = int((node or {}).get("post_id"))
            except Exception:
                continue
            ordered_nodes.append(node)
            ordered_index[pid] = idx
            depth_by_post_id[pid] = int(depth)

        return {
            "node_by_id": node_by_id,
            "children_by_parent": children_by_parent,
            "ordered_pairs": ordered_pairs,
            "ordered_nodes": ordered_nodes,
            "ordered_index": ordered_index,
            "depth_by_post_id": depth_by_post_id,
        }

    def _forum_basic_text_quality(self, text: str, *, depth: int = 1):
        raw = re.sub(r"\s+", " ", str(text or "").strip())
        norm = self._normalize_comment_for_repetition(raw)
        tokens = norm.split() if norm else []
        stop = {
            "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "get",
            "got", "had", "has", "have", "he", "her", "hers", "him", "his", "i", "if", "in",
            "into", "is", "it", "its", "just", "like", "me", "my", "of", "on", "or", "our",
            "ours", "really", "she", "so", "that", "the", "their", "them", "they", "this",
            "to", "too", "us", "was", "we", "were", "what", "when", "where", "who", "why",
            "will", "with", "would", "you", "your", "yours",
        }
        content_tokens = [tok for tok in tokens if len(tok) >= 3 and tok not in stop]
        lower_raw = raw.lower()

        challenge_patterns = [
            "how do you figure",
            "what makes you",
            "what makes that",
            "based on what",
            "why would",
            "why do you think",
            "how is",
            "what evidence",
            "what exactly",
            "what part of",
        ]
        is_question = "?" in raw
        is_direct_challenge = bool(
            any(pattern in lower_raw for pattern in challenge_patterns)
            or (
                is_question
                and any(token in lower_raw for token in ["how ", "why ", "what ", "which ", "where "])
            )
        )

        crowd_match = re.match(
            r"^(fans|people|everyone|they|viewers|audiences|folks|nobody)\s+"
            r"(will|wont|would|wouldnt|might|gonna|are|ll|could|should)\b",
            norm,
        )
        generic_crowd_reaction = bool(crowd_match)
        if not generic_crowd_reaction:
            generic_crowd_reaction = bool(
                re.search(
                    r"\b(fans|people|everyone|they|viewers|audiences|folks)\b.*\b"
                    r"(hyped|disappointed|upset|mad|like|love|hate|care)\b",
                    norm,
                )
                and len(content_tokens) <= 4
            )

        low_substance = False
        if int(depth or 0) > 0 and not is_direct_challenge:
            low_substance = bool(
                len(raw) < 40
                or len(tokens) < 7
                or len(content_tokens) < 3
                or generic_crowd_reaction
            )

        crowd_subject = crowd_match.group(1) if crowd_match else ""
        return {
            "raw_text": raw,
            "normalized_text": norm,
            "char_count": len(raw),
            "token_count": len(tokens),
            "content_token_count": len(content_tokens),
            "is_question": bool(is_question),
            "is_direct_challenge": bool(is_direct_challenge),
            "generic_crowd_reaction": bool(generic_crowd_reaction),
            "crowd_subject": crowd_subject,
            "low_substance": bool(low_substance),
        }

    def _forum_texts_are_redundant(self, left_text: str, right_text: str) -> bool:
        left_norm = self._normalize_comment_for_repetition(left_text)
        right_norm = self._normalize_comment_for_repetition(right_text)
        if not left_norm or not right_norm:
            return False
        if left_norm == right_norm:
            return True
        left_tokens = left_norm.split()
        right_tokens = right_norm.split()
        if len(left_tokens) >= 4 and len(right_tokens) >= 4:
            if " ".join(left_tokens[:4]) == " ".join(right_tokens[:4]):
                return True
        overlap = set(left_tokens[:18]).intersection(set(right_tokens[:18]))
        threshold = min(6, int(0.6 * min(len(left_tokens[:18]), len(right_tokens[:18]))))
        return len(overlap) >= max(threshold, 4)

    def _analyze_forum_reply_target(
        self,
        node: dict,
        depth: int,
        *,
        thread_root_id: int = None,
        node_by_id=None,
        children_by_parent=None,
        ordered_nodes=None,
        ordered_index=None,
    ):
        node = node or {}
        node_by_id = node_by_id or {}
        children_by_parent = children_by_parent or {}
        ordered_nodes = ordered_nodes or []
        ordered_index = ordered_index or {}

        try:
            post_id = int(node.get("post_id"))
        except Exception:
            post_id = None
        parent_post_id = node.get("comment_to", -1)
        try:
            parent_post_id = int(parent_post_id) if parent_post_id not in (None, "") else -1
        except Exception:
            parent_post_id = -1

        basic = self._forum_basic_text_quality(node.get("text", ""), depth=depth)
        nearby_seen = set()
        nearby_nodes = []

        def _append_nearby(candidate):
            if not isinstance(candidate, dict):
                return
            try:
                cand_id = int(candidate.get("post_id"))
            except Exception:
                cand_id = None
            if cand_id is None or cand_id == post_id or cand_id in nearby_seen:
                return
            nearby_seen.add(cand_id)
            nearby_nodes.append(candidate)

        if parent_post_id not in (-1, None):
            _append_nearby(node_by_id.get(parent_post_id))
            for child_id in children_by_parent.get(parent_post_id, []):
                _append_nearby(node_by_id.get(child_id))

        idx = ordered_index.get(post_id)
        if idx is not None:
            start = max(0, int(idx) - 4)
            for near_node in ordered_nodes[start:int(idx)]:
                _append_nearby(near_node)

        nearby_low_substance_count = 0
        redundant_match_count = 0
        for near_node in nearby_nodes:
            near_depth = 1
            try:
                near_pid = int(near_node.get("post_id"))
            except Exception:
                near_pid = None
            if near_pid is not None and thread_root_id is not None and near_pid == int(thread_root_id):
                near_depth = 0
            near_basic = self._forum_basic_text_quality(near_node.get("text", ""), depth=near_depth)
            if not near_basic.get("low_substance"):
                continue
            nearby_low_substance_count += 1
            same_crowd_frame = bool(
                basic.get("generic_crowd_reaction")
                and near_basic.get("generic_crowd_reaction")
                and basic.get("crowd_subject")
                and basic.get("crowd_subject") == near_basic.get("crowd_subject")
            )
            if same_crowd_frame or self._forum_texts_are_redundant(
                basic.get("normalized_text", ""),
                near_basic.get("normalized_text", ""),
            ):
                redundant_match_count += 1

        redundant_branch = bool(basic.get("low_substance") and redundant_match_count >= 2)
        basic.update(
            {
                "post_id": post_id,
                "parent_post_id": parent_post_id,
                "reply_depth": int(depth or 0),
                "redundant_branch": redundant_branch,
                "nearby_low_substance_count": int(nearby_low_substance_count),
                "redundant_match_count": int(redundant_match_count),
            }
        )
        return basic

    def _filter_forum_browse_candidates(
        self,
        candidates,
        *,
        thread_root_id: int,
        node_by_id,
        children_by_parent,
        ordered_nodes,
        ordered_index,
    ):
        filtered = []
        stats = {
            "filtered_low_substance_candidates": 0,
            "filtered_redundant_candidates": 0,
        }
        try:
            root_id = int(thread_root_id) if thread_root_id is not None else None
        except Exception:
            root_id = None

        for candidate in candidates or []:
            node = candidate.get("node") or {}
            try:
                post_id = int(node.get("post_id"))
            except Exception:
                post_id = None
            depth = candidate.get("depth")
            if depth is None and post_id is not None:
                depth = ordered_index.get(post_id, 0)
            quality_meta = self._analyze_forum_reply_target(
                node,
                int(depth or 0),
                thread_root_id=root_id,
                node_by_id=node_by_id,
                children_by_parent=children_by_parent,
                ordered_nodes=ordered_nodes,
                ordered_index=ordered_index,
            )
            candidate["quality_meta"] = quality_meta

            is_root = bool(root_id is not None and post_id == root_id)
            if not is_root and quality_meta.get("redundant_branch"):
                stats["filtered_redundant_candidates"] += 1
                continue
            if not is_root and quality_meta.get("low_substance"):
                stats["filtered_low_substance_candidates"] += 1
                continue
            filtered.append(candidate)

        return filtered, stats

    def _resolve_forum_reply_target_meta(
        self,
        *,
        post_id: int,
        thread_root_id: int = None,
        reply_depth: int = None,
        reply_to_text: str = "",
    ):
        try:
            root_id = int(thread_root_id) if thread_root_id is not None else int(post_id)
        except Exception:
            root_id = int(post_id)
        fallback_depth = int(reply_depth) if reply_depth is not None else (0 if int(post_id) == root_id else 1)
        fallback_node = {
            "post_id": int(post_id),
            "comment_to": -1 if int(post_id) == root_id else root_id,
            "text": reply_to_text or "",
        }
        fallback_meta = self._analyze_forum_reply_target(fallback_node, fallback_depth, thread_root_id=root_id)

        try:
            tree = self.__get_thread_tree(root_id, limit=getattr(self, "thread_browse_max_nodes", 400))
            if not isinstance(tree, dict) or tree.get("status") != 200 or not tree.get("posts"):
                return fallback_meta
            maps = self._build_thread_analysis_maps(tree.get("posts") or [], tree.get("thread_root_id"))
            target_node = maps["node_by_id"].get(int(post_id))
            if not isinstance(target_node, dict):
                return fallback_meta
            depth = maps["depth_by_post_id"].get(int(post_id), fallback_depth)
            return self._analyze_forum_reply_target(
                target_node,
                int(depth or 0),
                thread_root_id=tree.get("thread_root_id"),
                node_by_id=maps["node_by_id"],
                children_by_parent=maps["children_by_parent"],
                ordered_nodes=maps["ordered_nodes"],
                ordered_index=maps["ordered_index"],
            )
        except Exception:
            return fallback_meta

    def _build_comment_style_options(self, *, reply_depth: int = 1, effective_affect_signal: dict = None):
        effective = effective_affect_signal if isinstance(effective_affect_signal, dict) else {}
        is_high_affect = bool(effective.get("is_high_affect"))
        quick_styles_allowed = int(reply_depth or 0) <= 0
        styles = []
        if quick_styles_allowed:
            styles.extend(
                [
                    ("QUICK_AFFIRM", "Very short agreement/reaction (1-5 words)."),
                    ("QUICK_DISAGREE", "Very short confrontation or snarky dismissal (1-5 words)."),
                ]
            )
        styles.extend(
            [
                ("QUESTION", "Challenging question that pressures evidence or expertise (3-15 words)."),
                ("MEDIUM_ENGAGE", "2-3 sentence direct reply with disagreement/sarcasm when relevant."),
                ("PERSONAL_ANECDOTE", "2-3 sentence personal anecdote tied to the argument; can still disagree."),
            ]
        )
        if getattr(self, "style_elaborate_enabled", False):
            styles.append(
                ("ELABORATE", "3-5 sentence text-criticism breakdown with sharper confrontation.")
            )
        if is_high_affect:
            tox_val = float(getattr(self, "toxicity_val", 0) or 0)
            if tox_val > 0.3:
                styles.append(("TROLL", "Hard trolling with insults and provocation."))
            elif tox_val > 0:
                styles.append(("TROLL_SOFT", "Mild sarcastic trolling that provokes without insults."))
        return {
            "styles": styles,
            "style_names": [name for name, _ in styles],
            "quick_styles_allowed": bool(quick_styles_allowed),
        }

    def _forum_forced_skip_reason(self, target_quality: dict) -> str:
        if not isinstance(target_quality, dict):
            return ""
        if target_quality.get("low_substance") and target_quality.get("redundant_branch"):
            return "redundant_low_substance_branch"
        return ""

    def _format_forum_target_quality_hint(self, target_quality: dict) -> str:
        if not isinstance(target_quality, dict):
            return ""
        parts = []
        if target_quality.get("low_substance"):
            parts.append("Parent is short or generic.")
        if target_quality.get("generic_crowd_reaction"):
            parts.append("It reads like a vague crowd-reaction prediction.")
        if target_quality.get("redundant_branch"):
            parts.append("This branch already repeats the same weak point.")
        if target_quality.get("is_direct_challenge"):
            parts.append("The parent contains a specific challenge worth answering directly.")
        return " ".join(parts).strip()

    def _log_comment_style_selection(
        self,
        *,
        tid,
        post_id: int,
        thread_root_id: int,
        reply_depth: int,
        target_quality: dict,
        style_selection: dict,
    ):
        try:
            self._decision_log(
                {
                    "decision_type": "comment_style_selection",
                    "run_id": getattr(self, "memory_run_id", None),
                    "agent_user_id": int(getattr(self, "user_id", -1) or -1),
                    "agent_name": getattr(self, "name", None),
                    "tid": int(tid),
                    "post_id": int(post_id),
                    "thread_root_id": int(thread_root_id) if thread_root_id is not None else None,
                    "reply_depth": int(reply_depth or 0),
                    "selected_style": style_selection.get("selected_style"),
                    "available_styles": style_selection.get("available_styles") or [],
                    "quick_styles_allowed": bool(style_selection.get("quick_styles_allowed")),
                    "low_substance_parent": bool((target_quality or {}).get("low_substance")),
                    "redundant_branch": bool((target_quality or {}).get("redundant_branch")),
                    "generic_reaction_parent": bool((target_quality or {}).get("generic_crowd_reaction")),
                    "forced_skip_reason": style_selection.get("forced_skip_reason") or "",
                }
            )
        except Exception:
            pass

    def _thread_browse_score(self, node: dict, depth: int, interests_lower: list):
        """Lightweight heuristic to shortlist interesting targets; final choice is LLM-driven."""
        try:
            if node.get("user_id") == getattr(self, "user_id", None):
                return -1e9
        except Exception:
            pass

        # Absolute depth limit: replying would create depth+1, so exclude if already at max.
        max_abs_depth = int(getattr(self, "max_absolute_reply_depth", 0) or 0)
        if max_abs_depth > 0 and depth >= max_abs_depth:
            return -1e9

        text = (node.get("text") or "").strip()
        if not text:
            return -1e9
        t = text.lower()

        score = 0.0

        # Questions and invitations to respond.
        if "?" in t:
            score += 1.0
        if any(w in t for w in ["anyone else", "thoughts", "what do you think", "help", "why", "how"]):
            score += 0.5

        # Conflict / engagement hooks.
        if any(w in t for w in ["actually", "no", "wrong", "but", "however", "agree", "disagree", "imo", "hot take"]):
            score += 0.5

        # Interest match (substring; crude but cheap).
        try:
            hits = 0
            for it in interests_lower:
                if it and it in t:
                    hits += 1
                    if hits >= 2:
                        break
            score += 2.0 * min(hits, 2)
        except Exception:
            pass

        # Existing engagement.
        try:
            rc = int(node.get("reaction_count") or 0)
            if rc > 0:
                score += float(min(2.0, np.log1p(rc)))
        except Exception:
            pass

        # Slight preference for replying to a comment (vs always replying to root).
        if depth >= 1:
            score += 0.2

        return score

    def _format_thread_browse_line(self, node: dict, depth: int, max_chars: int):
        username = (node.get("username") or "unknown").strip()
        text = (node.get("text") or "").strip()
        if max_chars and len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."
        indent = "  " * max(0, int(depth))
        try:
            pid = int(node.get("post_id"))
        except Exception:
            pid = node.get("post_id")
        return f"{indent}- (id {pid}) @{username}: {text}"

    def _select_comment_target_via_thread_browse(self, seed_post_id: int, tid: int):
        """
        Human-like sequential thread browsing.

        Returns:
            dict with keys: post_id, username, text
            or None if the agent decides not to comment.
        """
        mode = getattr(self, "thread_browse_mode", "hybrid_llm") or "hybrid_llm"
        mode_lower = str(mode).lower()
        if mode_lower in ["off", "disabled", "none"]:
            return {"post_id": int(seed_post_id), "username": None, "text": None}
        # "hybrid_llm": heuristic shortlist + LLM final choice
        # "llm": LLM chooses comment target directly from the sequentially-read window (no heuristic filtering)
        use_heuristic = mode_lower in ["hybrid_llm", "hybrid", "hybrid-llm"]

        # Fetch thread tree from server (fallback to seed if unavailable).
        tree = None
        try:
            tree = self.__get_thread_tree(int(seed_post_id), limit=getattr(self, "thread_browse_max_nodes", 400))
        except Exception:
            tree = None

        if not isinstance(tree, dict) or tree.get("status") != 200 or not tree.get("posts"):
            return {"post_id": int(seed_post_id), "username": None, "text": None}

        thread_root_id = tree.get("thread_root_id")
        posts = tree.get("posts") or []

        # Thread comment cap: if thread already has too many comments, skip commenting entirely.
        max_comments_per_thread = int(getattr(self, "max_comments_per_thread", 0) or 0)
        if max_comments_per_thread > 0 and len(posts) >= max_comments_per_thread:
            logging.info(
                "[%s] Thread %s has %d comments (cap %d) — skipping comment",
                self.name, thread_root_id, len(posts), max_comments_per_thread,
            )
            return None

        thread_maps = self._build_thread_analysis_maps(posts, thread_root_id)
        ordered = thread_maps.get("ordered_pairs") or []
        if not ordered:
            return {"post_id": int(seed_post_id), "username": None, "text": None}

        browse_memory_context = ""
        browse_memory_meta = {"usage": "none"}
        try:
            if getattr(self, "memory_enabled", False) and thread_root_id is not None:
                browse_memory_context, browse_memory_meta = self._memory_build_thread_browse_context(
                    thread_root_id=int(thread_root_id),
                    tid=int(tid),
                )
        except Exception:
            browse_memory_context = ""
            browse_memory_meta = {"usage": "none"}

        # Interests for decision prompt and heuristics.
        try:
            current_interests, _ = self.__get_interests(tid)
        except Exception:
            current_interests = []
        interests_lower = [str(x).strip().lower() for x in (current_interests or []) if str(x).strip()]
        relationship_priority_enabled = bool(
            getattr(self, "memory_relationship_priority_enabled", True)
        )
        low_toxicity_trust_gate_applied = bool(
            relationship_priority_enabled and self._memory_is_low_toxicity_profile()
        )
        trust_gate_threshold = float(getattr(self, "memory_trust_gate_threshold", 0.15) or 0.15)
        alignment_gate_threshold = float(
            getattr(self, "memory_alignment_gate_threshold", 0.0) or 0.0
        )
        max_priority_targets = int(
            getattr(self, "memory_relationship_priority_max_targets", 3) or 3
        )
        relationship_signal_cache = {}

        # Shortlist candidates as we read. Map: post_id -> (score, node, depth, seen_idx)
        shortlist = {}

        def _update_shortlist(node: dict, depth: int, seen_idx: int):
            try:
                pid = int(node.get("post_id"))
            except Exception:
                return
            score = self._thread_browse_score(node, depth, interests_lower)
            if score <= -1e6:
                return
            prev = shortlist.get(pid)
            if prev is None or score > prev[0]:
                shortlist[pid] = (score, node, depth, seen_idx)

        chunk_size = int(getattr(self, "thread_browse_chunk_size", 20) or 20)
        top_k = int(getattr(self, "thread_browse_top_k", 6) or 6)
        max_steps = int(getattr(self, "thread_browse_max_llm_steps", 3) or 3)
        snippet_chars = int(getattr(self, "thread_browse_snippet_chars", 220) or 220)
        context_window = int(getattr(self, "thread_browse_context_window", 30) or 30)

        # Track what we've "read". Keep only the last context_window lines/nodes.
        read_lines = []
        read_nodes = []  # recent window of (node, depth) for LLM-only mode

        cursor = 0
        step = 0

        while cursor < len(ordered) and step < max_steps:
            chunk = ordered[cursor : cursor + chunk_size]
            cursor += chunk_size
            step += 1

            # Update read log + shortlist from this chunk.
            for i, (node, depth) in enumerate(chunk):
                line = self._format_thread_browse_line(node, depth, snippet_chars)
                read_lines.append(line)
                read_nodes.append({"node": node, "depth": depth})
                if len(read_lines) > context_window:
                    read_lines = read_lines[-context_window:]
                    read_nodes = read_nodes[-context_window:]
                if use_heuristic:
                    _update_shortlist(node, depth, seen_idx=(cursor - len(chunk) + i))

            # Build candidate list.
            candidates = []
            if use_heuristic:
                ranked = sorted(
                    shortlist.items(), key=lambda kv: (kv[1][0], kv[1][3]), reverse=True
                )
                for pid, (score, node, depth, seen_idx) in ranked[:top_k]:
                    candidates.append(
                        {"post_id": pid, "node": node, "depth": depth, "score": score}
                    )
            else:
                # No heuristic bias: expose recently-read comments as potential reply targets.
                seen_pids = set()
                max_abs_depth_llm = int(getattr(self, "max_absolute_reply_depth", 0) or 0)
                for rn in reversed(read_nodes):
                    n = rn.get("node") or {}
                    try:
                        pid = int(n.get("post_id"))
                    except Exception:
                        continue
                    if pid in seen_pids:
                        continue
                    # Skip nodes at or beyond absolute depth limit
                    node_depth = int(rn.get("depth", 0))
                    if max_abs_depth_llm > 0 and node_depth >= max_abs_depth_llm:
                        continue
                    seen_pids.add(pid)
                    candidates.append(
                        {"post_id": pid, "node": n, "depth": node_depth, "score": 0.0}
                    )

            candidates, filter_stats = self._filter_forum_browse_candidates(
                candidates,
                thread_root_id=thread_root_id,
                node_by_id=thread_maps.get("node_by_id"),
                children_by_parent=thread_maps.get("children_by_parent"),
                ordered_nodes=thread_maps.get("ordered_nodes"),
                ordered_index=thread_maps.get("ordered_index"),
            )

            priority_post_ids = set()
            priority_targets_for_log = []
            options_priority_buckets_for_log = []

            if relationship_priority_enabled and candidates:
                ranked_targets = []
                for c in candidates:
                    n = c.get("node") or {}
                    try:
                        uid = int(n.get("user_id"))
                    except Exception:
                        uid = None
                    if uid is None:
                        c["relationship_signal"] = {}
                        c["priority_score"] = float(c.get("score") or 0.0)
                        c["priority_eligible"] = False
                        continue

                    signal = relationship_signal_cache.get(uid)
                    if not isinstance(signal, dict):
                        signal = self._memory_get_relationship_signal_for_user(uid, int(tid))
                        relationship_signal_cache[uid] = signal

                    trust_score = float(signal.get("trust_score", 0.0) or 0.0)
                    affinity_score = float(signal.get("affinity_score", 0.0) or 0.0)
                    conflict_score = float(signal.get("conflict_score", 0.0) or 0.0)
                    recency_score = float(signal.get("recency_score", 0.0) or 0.0)
                    interaction_norm = float(signal.get("interaction_norm", 0.0) or 0.0)
                    relation_energy = max(abs(affinity_score), abs(conflict_score), abs(trust_score))

                    if low_toxicity_trust_gate_applied:
                        trust_ok = trust_score >= trust_gate_threshold
                        align_ok = affinity_score >= alignment_gate_threshold
                        priority_eligible = bool(trust_ok and align_ok and interaction_norm > 0.0)
                        priority_score = (
                            1.7 * trust_score
                            + 1.4 * affinity_score
                            + 0.8 * recency_score
                            + 0.5 * interaction_norm
                            - 0.9 * max(0.0, conflict_score)
                        )
                    else:
                        trust_ok = True
                        align_ok = True
                        priority_eligible = bool(interaction_norm > 0.0 or relation_energy > 0.10)
                        priority_score = (
                            1.0 * interaction_norm
                            + 0.8 * recency_score
                            + 0.7 * relation_energy
                        )

                    c["relationship_signal"] = signal
                    c["priority_score"] = float(priority_score)
                    c["priority_eligible"] = bool(priority_eligible)
                    c["priority_reason"] = (
                        "low_toxicity_trust_gate"
                        if low_toxicity_trust_gate_applied
                        else "toxicity_relaxed_gate"
                    )
                    c["priority_gate_passed"] = bool(trust_ok and align_ok)

                    ranked_targets.append(c)

                ranked_targets = sorted(
                    ranked_targets,
                    key=lambda item: (
                        0 if item.get("priority_eligible") else 1,
                        -float(item.get("priority_score") or 0.0),
                        -float(item.get("score") or 0.0),
                    ),
                )
                for c in ranked_targets:
                    if len(priority_post_ids) >= max_priority_targets:
                        break
                    if not c.get("priority_eligible"):
                        continue
                    try:
                        priority_post_ids.add(int(c.get("post_id")))
                    except Exception:
                        continue
                    signal = c.get("relationship_signal") if isinstance(c.get("relationship_signal"), dict) else {}
                    priority_targets_for_log.append(
                        {
                            "post_id": c.get("post_id"),
                            "username": (c.get("node") or {}).get("username"),
                            "user_id": (c.get("node") or {}).get("user_id"),
                            "priority_score": round(float(c.get("priority_score") or 0.0), 4),
                            "trust_score": round(float(signal.get("trust_score") or 0.0), 4),
                            "affinity_score": round(float(signal.get("affinity_score") or 0.0), 4),
                            "conflict_score": round(float(signal.get("conflict_score") or 0.0), 4),
                            "interaction_norm": round(float(signal.get("interaction_norm") or 0.0), 4),
                            "behavior_labels": signal.get("behavior_labels") if isinstance(signal.get("behavior_labels"), list) else [],
                        }
                    )

                if priority_post_ids:
                    def _priority_bucket_for_candidate(item):
                        try:
                            return 0 if int(item.get("post_id")) in priority_post_ids else 1
                        except Exception:
                            return 1

                    candidates = sorted(
                        candidates,
                        key=lambda item: (
                            _priority_bucket_for_candidate(item),
                            -float(item.get("priority_score") or 0.0),
                            -float(item.get("score") or 0.0),
                        ),
                    )
            restrict_comment_options_to_priority = bool(
                low_toxicity_trust_gate_applied and bool(priority_post_ids)
            )
            filtered_non_priority_candidates = 0

            # If we can't use the LLM (no prompts), fallback to best heuristic candidate.
            if not getattr(self, "prompts", None):
                if candidates:
                    best = candidates[0]["node"]
                    return {
                        "post_id": int(best.get("post_id")),
                        "username": best.get("username"),
                        "text": best.get("text"),
                    }
                return {"post_id": int(seed_post_id), "username": None, "text": None}

            prompt = self.prompts.get("handler_thread_browse_decision")
            if not prompt:
                prompt = (
                    "You are browsing a Reddit thread like a real human.\n"
                    "You are reading comments in order. At each step decide what to do next based on:\n"
                    "- the content you just read\n"
                    "- your persona\n"
                    "- your interests\n"
                    "- the subreddit vibe\n\n"
                    "Choose ONE action by outputting ONLY its number.\n\n"
                    "SUBREDDIT VIBE: {subreddit_vibe}\n"
                    "YOUR INTERESTS: {interests_str}\n\n"
                    "RECENTLY READ:\n{scan_snippets}\n\n"
                    "ACTIONS:\n{options}\n\n"
                    "Rules:\n"
                    "- If you COMMENT: pick a specific comment to reply to (not your own).\n"
                    "- If you have nothing meaningful to add: STOP.\n"
                    "- Output ONLY a single number.\n"
                )

            scan_snippets = "\n".join(read_lines)
            if browse_memory_context:
                scan_snippets = browse_memory_context + "\n\n" + scan_snippets

            try:
                self._decision_log(
                    {
                        "decision_type": "browse_memory_usage",
                        "run_id": getattr(self, "memory_run_id", None),
                        "agent_user_id": int(getattr(self, "user_id", -1) or -1),
                        "agent_name": getattr(self, "name", None),
                        "tid": int(tid),
                        "thread_root_id": int(thread_root_id) if thread_root_id is not None else None,
                        "usage": browse_memory_meta.get("usage") or "none",
                    }
                )
            except Exception:
                pass

            options = []
            # On the final decision step, force a stop-or-comment choice (no infinite reading).
            if cursor < len(ordered) and step < max_steps:
                options.append({"type": "continue"})
            options.append({"type": "stop"})

            for c in candidates:
                n = c["node"]
                try:
                    pid = int(n.get("post_id"))
                except Exception:
                    continue
                # Prevent replying to self just in case.
                try:
                    if n.get("user_id") == getattr(self, "user_id", None):
                        continue
                except Exception:
                    pass
                if restrict_comment_options_to_priority and pid not in priority_post_ids:
                    filtered_non_priority_candidates += 1
                    continue
                options.append({"type": "comment", "post_id": pid, "username": n.get("username"), "text": n.get("text")})

            candidate_by_post = {}
            for c in candidates:
                try:
                    candidate_by_post[int(c.get("post_id"))] = c
                except Exception:
                    continue

            shuffle_temperature = float(
                getattr(self, "memory_option_shuffle_temperature", 0.35) or 0.35
            )
            indexed_options = []
            for idx, opt in enumerate(options):
                opt_type = str(opt.get("type") or "")
                post_id = opt.get("post_id")
                if opt_type == "comment":
                    try:
                        pid = int(post_id)
                    except Exception:
                        pid = None
                    cand = candidate_by_post.get(pid) if pid is not None else None
                    if pid is not None and pid in priority_post_ids:
                        bucket = 0
                        base_rank = -float((cand or {}).get("priority_score") or 0.0)
                    else:
                        bucket = 1
                        base_rank = -float((cand or {}).get("score") or 0.0)
                else:
                    bucket = 2
                    base_rank = 0.0

                jitter = random.random() * shuffle_temperature if shuffle_temperature > 0 else 0.0
                indexed_options.append((bucket, base_rank + jitter, idx, opt))
                priority_score_for_log = 0.0
                if opt_type == "comment":
                    try:
                        priority_score_for_log = float(
                            candidate_by_post.get(int(post_id), {}).get("priority_score") or 0.0
                        )
                    except Exception:
                        priority_score_for_log = 0.0
                options_priority_buckets_for_log.append(
                    {
                        "type": opt_type,
                        "post_id": post_id,
                        "bucket": int(bucket),
                        "priority_score": round(priority_score_for_log, 4),
                    }
                )

            options = [row[3] for row in sorted(indexed_options, key=lambda row: (row[0], row[1], row[2]))]

            def _opt_line(opt):
                if opt["type"] == "continue":
                    return "CONTINUE - keep reading more of the thread."
                if opt["type"] == "stop":
                    return "STOP - stop reading without commenting."
                # comment
                uname = (opt.get("username") or "unknown").strip()
                txt = (opt.get("text") or "").strip()
                if snippet_chars and len(txt) > snippet_chars:
                    txt = txt[: snippet_chars - 3].rstrip() + "..."
                return f"COMMENT - reply to (id {opt.get('post_id')}) @{uname}: {txt}"

            options_text = "\n".join([f"{i}. {_opt_line(opt)}" for i, opt in enumerate(options, start=1)])
            interests_str = ", ".join(current_interests) if current_interests else "none"

            decision_config = self.__get_fresh_llm_config()
            decision_config["temperature"] = 0.2
            decision_config["max_tokens"] = 40

            u1 = AssistantAgent(
                name=f"{self.name}",
                llm_config=decision_config,
                system_message=self.__effify(
                    self.prompts.get(
                        "agent_roleplay_comments_share",
                        self.prompts.get("agent_roleplay_simple", ""),
                    ),
                    interests=current_interests,
                ),
                # Must allow a single LLM response; otherwise u1 never replies and we only see the prompt.
                max_consecutive_auto_reply=1,
            )

            u2 = AssistantAgent(
                name="Handler",
                llm_config=decision_config,
                system_message=self.__effify(
                    self.prompts.get(
                        "handler_instructions_simple",
                        "You are the Handler that specifies the actions to be taken.",
                    )
                ),
                max_consecutive_auto_reply=0,
            )

            # Dominant view detection for thread browse context
            dominant_view_block = ""
            try:
                if getattr(self, "proactive_affect_enabled", True):
                    dom = self._detect_dominant_view_pressure(scan_snippets)
                    if dom.get("dominant_view_detected"):
                        profile = self._build_persona_affect_profile()
                        if profile.get("disagree_propensity", 0) > 0.4:
                            dominant_view_block = (
                                "This thread shows strong consensus. "
                                "Your personality may drive you to challenge or probe."
                            )
            except Exception:
                dominant_view_block = ""

            u2.initiate_chat(
                u1,
                message=self.__effify(
                    prompt,
                    subreddit_vibe=getattr(self, "subreddit_vibe", "") or "",
                    interests_str=interests_str,
                    scan_snippets=scan_snippets,
                    options=options_text,
                    dominant_view_block=dominant_view_block,
                ),
                silent=True,
                max_turns=1,
            )

            raw = ""
            try:
                raw = u1.chat_messages[u2][-1]["content"].strip()
            except Exception:
                raw = ""

            u1.reset()
            u2.reset()

            # Prefer strict "number only" output, but be resilient to minor formatting (e.g. "2.").
            m = re.match(r"^\s*(\d+)\s*$", raw) or re.search(r"\b(\d+)\b", raw)
            choice = None
            if m:
                try:
                    choice = int(m.group(1))
                except Exception:
                    choice = None

            # Structured decision log (best-effort) before acting.
            try:
                options_for_log = []
                for opt in options:
                    if opt.get("type") == "comment":
                        options_for_log.append(
                            {
                                "type": "comment",
                                "post_id": opt.get("post_id"),
                                "username": opt.get("username"),
                            }
                        )
                    else:
                        options_for_log.append({"type": opt.get("type")})

                cand_for_log = []
                for c in candidates[: min(len(candidates), top_k)]:
                    n = c.get("node") or {}
                    signal = c.get("relationship_signal") if isinstance(c.get("relationship_signal"), dict) else {}
                    quality_meta = c.get("quality_meta") if isinstance(c.get("quality_meta"), dict) else {}
                    cand_for_log.append(
                        {
                            "post_id": n.get("post_id"),
                            "username": n.get("username"),
                            "user_id": n.get("user_id"),
                            "score": c.get("score"),
                            "depth": c.get("depth"),
                            "priority_score": c.get("priority_score"),
                            "priority_eligible": bool(c.get("priority_eligible", False)),
                            "trust_score": signal.get("trust_score"),
                            "affinity_score": signal.get("affinity_score"),
                            "conflict_score": signal.get("conflict_score"),
                            "interaction_norm": signal.get("interaction_norm"),
                            "low_substance": bool(quality_meta.get("low_substance")),
                            "generic_crowd_reaction": bool(quality_meta.get("generic_crowd_reaction")),
                            "redundant_branch": bool(quality_meta.get("redundant_branch")),
                            "reply_depth": quality_meta.get("reply_depth"),
                        }
                    )

                valid = bool(choice is not None and 1 <= choice <= len(options))
                selected_for_log = None
                if valid:
                    opt = options[choice - 1]
                    if opt.get("type") == "comment":
                        selected_for_log = {
                            "type": "comment",
                            "post_id": opt.get("post_id"),
                            "username": opt.get("username"),
                        }
                    else:
                        selected_for_log = {"type": opt.get("type")}

                self._decision_log(
                    {
                        "decision_type": "thread_browse_decision",
                        "run_id": getattr(self, "memory_run_id", None),
                        "agent_user_id": int(getattr(self, "user_id", -1) or -1),
                        "agent_name": getattr(self, "name", None),
                        "tid": int(tid),
                        "browse_mode": getattr(self, "thread_browse_mode", None),
                        "heuristic_shortlist": bool(use_heuristic),
                        "thread_root_id": int(thread_root_id) if thread_root_id is not None else None,
                        "seed_post_id": int(seed_post_id),
                        "step": int(step),
                        "cursor": int(cursor),
                        "ordered_len": int(len(ordered)),
                        "scan_snippets": scan_snippets,
                        "candidates": cand_for_log,
                        "options": options_for_log,
                        "relationship_priority_enabled": bool(relationship_priority_enabled),
                        "low_toxicity_trust_gate_applied": bool(low_toxicity_trust_gate_applied),
                        "restrict_comment_options_to_priority": bool(
                            restrict_comment_options_to_priority
                        ),
                        "filtered_non_priority_candidates": int(
                            filtered_non_priority_candidates
                        ),
                        "filtered_low_substance_candidates": int(
                            filter_stats.get("filtered_low_substance_candidates", 0)
                        ),
                        "filtered_redundant_candidates": int(
                            filter_stats.get("filtered_redundant_candidates", 0)
                        ),
                        "priority_targets": priority_targets_for_log,
                        "options_priority_buckets": options_priority_buckets_for_log,
                        "llm_raw": raw,
                        "parsed_choice": choice,
                        "valid_choice": valid,
                        "selected": selected_for_log,
                    }
                )
            except Exception:
                pass

            if choice is None or not (1 <= choice <= len(options)):
                # Default: keep reading if possible, else stop.
                if cursor < len(ordered) and step < max_steps:
                    continue
                return None

            selected = options[choice - 1]

            if selected["type"] == "continue":
                continue
            if selected["type"] == "stop":
                return None
            if selected["type"] == "comment":
                selected_quality = {}
                try:
                    selected_quality = candidate_by_post.get(
                        int(selected.get("post_id"))
                    ).get("quality_meta") or {}
                except Exception:
                    selected_quality = {}
                return {
                    "post_id": int(selected.get("post_id")),
                    "username": selected.get("username"),
                    "text": selected.get("text"),
                    "depth": selected_quality.get("reply_depth"),
                    "quality_meta": selected_quality,
                }

        return None

    def get_user_from_post(self, post_id: int):
        """
        Get the user from a post.

        :param post_id: The post id to get the user.
        :return: the user
        """
        api_url = f"{self.base_url}/get_user_from_post"

        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        params = {"post_id": post_id}
        st = json.dumps(params)
        response = post(f"{api_url}", headers=headers, data=st)

        res = json.loads(response.__dict__["_content"].decode("utf-8"))
        return res

    def get_username_from_post(self, post_id: int):
        """
        Get the author's user id and username for a post/comment.

        External Reddit servers historically returned only a username for /get_user_from_post.
        This helper prefers /get_username_from_post (status + user_id + username) but falls
        back to /get_user_from_post + /get_user_id.

        :param post_id: The post id to get the author for.
        :return: (user_id, username)
        """
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        # Preferred: new endpoint returns both id + username
        try:
            api_url = f"{self.base_url}/get_username_from_post"
            st = json.dumps({"post_id": int(post_id)})
            response = post(f"{api_url}", headers=headers, data=st)
            payload = json.loads(response.__dict__["_content"].decode("utf-8"))
            if isinstance(payload, dict) and payload.get("status") == 200:
                uid = payload.get("user_id")
                uname = payload.get("username")
                try:
                    uid = int(uid) if uid is not None else None
                except (TypeError, ValueError):
                    uid = None
                if not isinstance(uname, str) or not uname:
                    uname = None
                return uid, uname
        except Exception:
            pass

        # Fallback: old endpoint returns username only
        uname = None
        try:
            uname = self.get_user_from_post(post_id)
        except Exception:
            uname = None

        uid = None
        if isinstance(uname, str) and uname:
            try:
                api_url = f"{self.base_url}/get_user_id"
                st = json.dumps({"username": uname})
                response = post(f"{api_url}", headers=headers, data=st)
                payload = json.loads(response.__dict__["_content"].decode("utf-8"))
                if isinstance(payload, dict):
                    uid = payload.get("id")
                    try:
                        uid = int(uid) if uid is not None else None
                    except (TypeError, ValueError):
                        uid = None
            except Exception:
                uid = None

        return uid, uname if isinstance(uname, str) and uname else None

    def __get_article(self, post_id: int):
        """
        Get the article.

        :param post_id: The article id to get the article.
        :return: the article
        """
        api_url = f"{self.base_url}/get_article"

        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        params = {"post_id": int(post_id)}
        st = json.dumps(params)
        response = post(f"{api_url}", headers=headers, data=st)
        if response.status_code == 404:
            return None
        res = json.loads(response.__dict__["_content"].decode("utf-8"))
        return res

    def __get_post(self, post_id: int):
        """
        Get the thread of a post.

        :param post_id: The post id to get the thread.
        :return: the post
        """
        api_url = f"{self.base_url}/get_post"

        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        params = {"post_id": post_id}
        st = json.dumps(params)
        response = post(f"{api_url}", headers=headers, data=st)

        res = json.loads(response.__dict__["_content"].decode("utf-8"))
        return res

    def _parse_choice_number(self, response: str, *, min_choice: int, max_choice: int):
        """
        Parse first standalone integer token from LLM output and validate bounds.
        """
        if not isinstance(response, str):
            return None
        if min_choice > max_choice:
            return None

        match = re.search(r"\b(\d+)\b", response)
        if not match:
            return None

        try:
            choice = int(match.group(1))
        except Exception:
            return None

        if choice < int(min_choice) or choice > int(max_choice):
            return None
        return choice

    def _select_comment_style(
        self,
        conv: str,
        interests: list,
        *,
        reply_depth: int = 1,
        target_quality: dict = None,
        return_metadata: bool = False,
        memory_cues_block: str = "",
        memory_scope: str = "none",
        memory_callback_hint: str = "",
        memory_argument_hint: str = "",
        memory_tone_hint: str = "",
        high_affect_flags: str = "",
        recalled_memories_block: str = "",
        memory_usage_requirement: str = "",
        effective_affect_signal: dict = None,
        proactive_affect_block: str = "",
    ):
        """
        Select comment style using LLM based on persona and context.

        :param conv: the conversation text
        :param interests: agent's interests
        :return: selected style name by default, or metadata dict when return_metadata=True
        """
        style_catalog = self._build_comment_style_options(
            reply_depth=reply_depth,
            effective_affect_signal=effective_affect_signal,
        )
        all_styles = style_catalog.get("styles") or []
        available_style_names = style_catalog.get("style_names") or []
        quick_styles_allowed = bool(style_catalog.get("quick_styles_allowed"))
        target_quality_hint = self._format_forum_target_quality_hint(target_quality)

        def _finalize(selected_style: str, forced_skip_reason: str = ""):
            payload = {
                "selected_style": selected_style,
                "available_styles": available_style_names,
                "quick_styles_allowed": quick_styles_allowed,
                "forced_skip_reason": forced_skip_reason,
            }
            if return_metadata:
                return payload
            return selected_style

        # Check if style selection prompt exists
        if "style_select_comment" not in self.prompts:
            fallback = random.choice(all_styles)[0] if all_styles else "SKIP"
            return _finalize(fallback)

        # Get parent comment snippet for context (expanded to 500 chars)
        parent_snippet = conv[-500:] if len(conv) > 500 else conv

        # Format interests as comma-separated string
        interests_str = ", ".join(interests) if interests else "general topics"

        # Shuffle to avoid primacy bias
        shuffled_styles = all_styles.copy()
        random.shuffle(shuffled_styles)

        # Build numbered styles list for prompt (1-6 instead of names)
        styles_list = "\n".join([f"{i+1}. {desc}" for i, (name, desc) in enumerate(shuffled_styles)])

        style_config = self.llm_config.copy()
        style_config["temperature"] = 0.7

        u1 = AssistantAgent(
            name=f"{self.name}",
            llm_config=style_config,
            system_message=self.__effify(
                self.prompts["agent_roleplay_simple"]
            ),
            max_consecutive_auto_reply=1,
        )

        u2 = AssistantAgent(
            name="StyleSelector",
            llm_config=style_config,
            system_message="You select response styles. Output only a number.",
            max_consecutive_auto_reply=0,
        )

        u2.initiate_chat(
            u1,
            message=self.__effify(
                self.prompts["style_select_comment"],
                parent_snippet=parent_snippet,
                interests_str=interests_str,
                styles_list=styles_list,
                num_styles=str(len(shuffled_styles)),
                memory_cues_block=memory_cues_block or "",
                memory_scope=memory_scope or "none",
                memory_callback_hint=memory_callback_hint or "",
                memory_argument_hint=memory_argument_hint or "",
                memory_tone_hint=memory_tone_hint or "",
                high_affect_flags=high_affect_flags or "",
                recalled_memories_block=recalled_memories_block or "",
                memory_usage_requirement=memory_usage_requirement or "",
                proactive_affect_block=proactive_affect_block or "",
                target_quality_hint=target_quality_hint or "",
            ),
            silent=True,
            max_turns=1,
        )

        response = u1.chat_messages[u2][-1]["content"].strip()

        choice = self._parse_choice_number(
            response,
            min_choice=0,
            max_choice=len(shuffled_styles),
        )
        if choice is not None:
            if choice == 0:
                return _finalize("SKIP")
            return _finalize(shuffled_styles[choice - 1][0])

        # Random fallback if no valid number found
        fallback = random.choice(all_styles)[0] if all_styles else "SKIP"
        return _finalize(fallback)

    def _select_share_style(self, article_title: str, article_summary: str, interests: list) -> str:
        """
        Select share style using LLM based on persona and article.

        :param article_title: the article title
        :param article_summary: the article summary
        :param interests: agent's interests
        :return: selected style name (e.g., 'CRITICAL_ESSAY', 'MINIMAL')
        """
        # Define all available share styles
        all_styles = [
            ("CRITICAL_ESSAY", "2-4 sentence critical breakdown (claims, framing, contradictions)."),
            ("ENTHUSIASTIC", "Short supportive share with strong positive stance."),
            ("QUESTIONING", "Provocative question that invites debate."),
            ("MINIMAL", "One-line take."),
            ("SKEPTICAL", "Snarky skeptical share or callout."),
            ("CONTEXT_ADD", "Add corrective context or fact-check angle."),
        ]

        # Check if style selection prompt exists
        if "style_select_share" not in self.prompts:
            return random.choice(all_styles)[0]  # random fallback

        # Article summary snippet (200 chars max)
        article_summary_snippet = article_summary[:200] if len(article_summary) > 200 else article_summary

        # Format interests as comma-separated string
        interests_str = ", ".join(interests) if interests else "general topics"

        # Shuffle to avoid primacy bias
        shuffled_styles = all_styles.copy()
        random.shuffle(shuffled_styles)

        # Build numbered styles list for prompt (1-6 instead of names)
        share_styles_list = "\n".join([f"{i+1}. {desc}" for i, (name, desc) in enumerate(shuffled_styles)])

        style_config = self.llm_config.copy()
        style_config["temperature"] = 0.7

        u1 = AssistantAgent(
            name=f"{self.name}",
            llm_config=style_config,
            system_message=self.__effify(
                self.prompts["agent_roleplay_simple"]
            ),
            max_consecutive_auto_reply=1,
        )

        u2 = AssistantAgent(
            name="StyleSelector",
            llm_config=style_config,
            system_message="You select sharing styles. Output only a number.",
            max_consecutive_auto_reply=0,
        )

        u2.initiate_chat(
            u1,
            message=self.__effify(
                self.prompts["style_select_share"],
                article_title=article_title,
                article_summary_snippet=article_summary_snippet,
                interests_str=interests_str,
                share_styles_list=share_styles_list
            ),
            silent=True,
            max_turns=1,
        )

        response = u1.chat_messages[u2][-1]["content"].strip()

        choice = self._parse_choice_number(
            response,
            min_choice=1,
            max_choice=len(shuffled_styles),
        )
        if choice is not None:
            return shuffled_styles[choice - 1][0]

        # Random fallback if no valid number found
        return random.choice(all_styles)[0]

    @log_execution_time
    def comment(
        self,
        post_id: int,
        tid,
        max_length_threads=None,
        reply_to_username=None,
        reply_to_text=None,
        reply_target_depth=None,
        reply_target_meta=None,
    ):
        """
        Generate a comment to an existing post

        :param post_id: the post id
        :param tid: the round id
        :param max_length_threads: the maximum length of the thread to read for context
        :param reply_to_username: optional explicit author of the last message (for accurate targeting)
        :param reply_to_text: optional explicit last message text (for accurate targeting)
        :param reply_target_depth: optional structured depth of the reply target
        :param reply_target_meta: optional quality metadata from thread browsing
        """

        conversation = self.__get_thread(post_id, max_tweets=max_length_threads)
        conv = "".join(conversation)

        # Ensure the intended reply target is the LAST line in the prompt, even if the thread is long.
        if reply_to_username and reply_to_text:
            try:
                uname = str(reply_to_username).strip().lstrip("@")
            except Exception:
                uname = None
            try:
                txt = str(reply_to_text).strip()
            except Exception:
                txt = ""

            if uname and txt:
                conv = conv.rstrip() + "\n" + f"@{uname} - {txt}\n"

        # Keep the last part of the thread to reduce context length for Ollama while preserving
        # the immediate reply target and nearby context.
        max_chars = (
            getattr(self, "max_thread_context_chars", DEFAULT_MAX_THREAD_CONTEXT_CHARS)
            or DEFAULT_MAX_THREAD_CONTEXT_CHARS
        )
        if max_chars > 0 and len(conv) > max_chars:
            conv = conv[-max_chars:]

        conv_raw = conv

        # Get the author of the post being replied to
        if reply_to_username:
            parent_author = str(reply_to_username).strip().lstrip("@")
        else:
            parent_author = self.get_user_from_post(post_id)
        self.replying_to = parent_author  # Store for use in __effify

        # Memory context (run-scoped): social + thread + community digest.
        other_user_id, other_username = (None, None)
        try:
            other_user_id, other_username = self._memory_get_author_id_and_username(int(post_id))
        except Exception:
            other_user_id, other_username = (None, None)

        thread_root_id = None
        try:
            thread_root_id = self._memory_get_thread_root_id(int(post_id))
        except Exception:
            thread_root_id = None
        if thread_root_id is None:
            thread_root_id = int(post_id)
        target_quality = (
            dict(reply_target_meta)
            if isinstance(reply_target_meta, dict)
            else self._resolve_forum_reply_target_meta(
                post_id=int(post_id),
                thread_root_id=int(thread_root_id),
                reply_depth=reply_target_depth,
                reply_to_text=reply_to_text or "",
            )
        )
        reply_depth = int(
            target_quality.get(
                "reply_depth",
                reply_target_depth if reply_target_depth is not None else (0 if int(post_id) == int(thread_root_id) else 1),
            )
            or 0
        )
        subtle_memory_mode = self._memory_use_subtle_prompt_mode()

        memory_text = ""
        memory_meta = {}
        memory_cues = {}
        memory_cues_block = ""
        memory_plan_hint = ""
        high_affect_signal = {
            "is_high_affect": False,
            "confidence": 0.0,
            "source": "rules",
            "triggers": {},
        }
        high_affect_flags = ""
        recalled_pack = {
            "items": [],
            "counts_by_bucket": {},
            "has_usable_memories": False,
            "prompt_block": "",
        }
        recalled_memories_block = ""
        memory_usage_requirement = ""

        try:
            if getattr(self, "memory_enabled", False):
                query_text = self._memory_build_query_text(
                    "compose a reply in this thread",
                    conv_raw,
                    reply_to_text or "",
                    f"reply_to={other_username or parent_author or ''}",
                )
                if subtle_memory_mode:
                    mem_text, memory_meta = self._memory_build_reply_context(
                        query_text=query_text,
                        other_user_id=other_user_id,
                        thread_root_id=thread_root_id,
                        other_username=other_username or parent_author,
                        round_id=tid,
                    )
                else:
                    mem_text, memory_meta = self._memory_build_tiered_context(
                        query_text=query_text,
                        other_user_id=other_user_id,
                        thread_root_id=thread_root_id,
                        other_username=other_username or parent_author,
                        round_id=tid,
                        uncertainty_score=0.55,
                    )
                if mem_text:
                    memory_text = mem_text
                    if not subtle_memory_mode:
                        conv = mem_text + "\n\n" + conv
        except Exception:
            pass

        # obtain the most recent (and frequent) interests of the agent
        # interests, _ = self.__get_interests(tid)

        # get the post_id topics
        api_url = f"{self.base_url}/get_post_topics_name"
        response = get(f"{api_url}", headers={"Content-Type": "application/x-www-form-urlencoded"},
                        data=json.dumps({"post_id": post_id}))
        interests = json.loads(response.__dict__["_content"].decode("utf-8"))

        # get the opinion on the topics (if present)
        self.topics_opinions = ""
        if len(interests) > 0:
            # get recent sentiment on the selected interests
            api_url = f"{self.base_url}/get_sentiment"
            data = {"user_id": self.user_id, "interests": interests}
            response = post(f"{api_url}", headers={"Content-Type": "application/x-www-form-urlencoded"},
                            data=json.dumps(data))
            sentiment = json.loads(response.__dict__["_content"].decode("utf-8"))

            self.topics_opinions = "Your opinion on the topics of the post you are responding to are: "
            for s in sentiment:
                self.topics_opinions += f"{s['topic']}: {s['sentiment']} "
            if len(sentiment) == 0:
                self.topics_opinions = ""

        try:
            if getattr(self, "memory_nuance_enabled", True):
                memory_cues = self._memory_build_conversation_cues(
                    memory_text=memory_text,
                    memory_meta=memory_meta,
                    target_username=other_username or parent_author,
                    mode="comment",
                )
                memory_cues_block = self._memory_format_conversation_cues(memory_cues)
                scope = memory_cues.get("scope") if isinstance(memory_cues, dict) else "none"
                if (
                    memory_cues_block
                    and scope in {"strong", "partial", "degraded"}
                    and getattr(self, "memory_nuance_planner_enabled", True)
                    and not subtle_memory_mode
                ):
                    memory_plan_hint = self._memory_plan_reply_strategy(
                        mode="comment",
                        mention_author=other_username or parent_author or "",
                        mention_text=reply_to_text or "",
                        thread_context=conv_raw,
                        memory_cues_block=memory_cues_block,
                        interests=interests,
                    )
        except Exception:
            memory_cues = {}
            memory_cues_block = ""
            memory_plan_hint = ""

        try:
            self._decision_log(
                {
                    "decision_type": "reply_memory_mode",
                    "run_id": getattr(self, "memory_run_id", None),
                    "agent_user_id": int(getattr(self, "user_id", -1) or -1),
                    "agent_name": getattr(self, "name", None),
                    "tid": int(tid),
                    "post_id": int(post_id),
                    "mode": "comment",
                    "memory_prompt_mode": "subtle_forum" if subtle_memory_mode else "legacy",
                    "cross_thread_callback_used": bool(
                        memory_meta.get("cross_thread_callback_candidate")
                    ),
                    "continuity_text": self._decision_compact_text(
                        memory_meta.get("continuity_text"), 320
                    ),
                }
            )
        except Exception:
            pass

        incoming_text_for_signal = str(reply_to_text or "").strip()
        incoming_author_for_signal = str(reply_to_username or "").strip().lstrip("@")
        if not incoming_text_for_signal:
            incoming_text_for_signal, incoming_author_for_signal = self._extract_last_thread_message(conv_raw)
        if not incoming_author_for_signal:
            incoming_author_for_signal = str(other_username or parent_author or "").strip().lstrip("@")

        try:
            if getattr(self, "memory_high_affect_enabled", True) and not subtle_memory_mode:
                high_affect_signal = self._detect_high_affect_signal(
                    incoming_text=incoming_text_for_signal,
                    thread_context=conv_raw,
                    other_user_id=other_user_id,
                    thread_root_id=thread_root_id,
                    round_id=tid,
                    target_username=incoming_author_for_signal,
                )
                high_affect_flags = self._memory_format_high_affect_flags(high_affect_signal)

                if bool(high_affect_signal.get("is_high_affect")):
                    recalled_pack = self._memory_collect_high_affect_recall(
                        incoming_text=incoming_text_for_signal,
                        thread_context=conv_raw,
                        other_user_id=other_user_id,
                        thread_root_id=thread_root_id,
                        round_id=tid,
                        target_username=incoming_author_for_signal,
                    )
                    recalled_memories_block = str(recalled_pack.get("prompt_block") or "").strip()
                    if recalled_memories_block:
                        requirement_template = (
                            self.prompts.get("memory_callback_requirements_comment")
                            if isinstance(self.prompts, dict)
                            else ""
                        )
                        if isinstance(requirement_template, str) and requirement_template.strip():
                            try:
                                memory_usage_requirement = requirement_template.format(
                                    high_affect_flags=high_affect_flags,
                                    recalled_memories_block=recalled_memories_block,
                                    incoming_text=incoming_text_for_signal,
                                    incoming_author=incoming_author_for_signal,
                                ).strip()
                            except Exception:
                                memory_usage_requirement = requirement_template.strip()
                        else:
                            memory_usage_requirement = (
                                "Use one recalled memory naturally in your reply. "
                                "Do not invent events."
                            )
                        triggers = high_affect_signal.get("triggers")
                        if isinstance(triggers, dict) and triggers.get("incoming_anecdote"):
                            memory_usage_requirement += (
                                " Since the incoming message used a personal story, "
                                "mirror with one short relevant memory if available."
                            )
        except Exception:
            high_affect_signal = {
                "is_high_affect": False,
                "confidence": 0.0,
                "source": "rules",
                "triggers": {},
            }
            high_affect_flags = ""
            recalled_pack = {
                "items": [],
                "counts_by_bucket": {},
                "has_usable_memories": False,
                "prompt_block": "",
            }
            recalled_memories_block = ""
            memory_usage_requirement = ""

        try:
            self._decision_log(
                {
                    "decision_type": "high_affect_detection",
                    "run_id": getattr(self, "memory_run_id", None),
                    "agent_user_id": int(getattr(self, "user_id", -1) or -1),
                    "agent_name": getattr(self, "name", None),
                    "tid": int(tid),
                    "post_id": int(post_id),
                    "target_username": incoming_author_for_signal,
                    "incoming_text": self._decision_compact_text(incoming_text_for_signal, 280),
                    "high_affect_signal": high_affect_signal,
                }
            )
            if bool(high_affect_signal.get("is_high_affect")):
                self._decision_log(
                    {
                        "decision_type": "high_affect_recall",
                        "run_id": getattr(self, "memory_run_id", None),
                        "agent_user_id": int(getattr(self, "user_id", -1) or -1),
                        "agent_name": getattr(self, "name", None),
                        "tid": int(tid),
                        "post_id": int(post_id),
                        "counts_by_bucket": recalled_pack.get("counts_by_bucket"),
                        "has_usable_memories": bool(recalled_pack.get("has_usable_memories")),
                        "recalled_memories_block": self._decision_compact_text(
                            recalled_memories_block, 800
                        ),
                    }
                )
        except Exception:
            pass

        # Proactive affect detection (personality-driven, complements reactive)
        proactive_signal = {
            "is_proactive_high_affect": False,
            "mode": None,
            "confidence": 0.0,
            "reasons": [],
            "dominant_view_detected": False,
        }
        proactive_affect_block = ""
        proactive_triggered = False
        try:
            if getattr(self, "proactive_affect_enabled", True):
                proactive_signal = self._detect_proactive_high_affect_signal(
                    thread_context=conv_raw,
                    round_id=tid,
                )
                if proactive_signal.get("is_proactive_high_affect"):
                    proactive_affect_block = self._format_proactive_affect_block(proactive_signal)
                    proactive_triggered = True
        except Exception:
            pass

        effective_affect_signal = self._build_effective_affect_signal(high_affect_signal, proactive_signal)

        try:
            if proactive_signal.get("is_proactive_high_affect"):
                self._decision_log({
                    "decision_type": "proactive_affect_detection",
                    "run_id": getattr(self, "memory_run_id", None),
                    "agent_user_id": int(getattr(self, "user_id", -1) or -1),
                    "agent_name": getattr(self, "name", None),
                    "tid": int(tid),
                    "post_id": int(post_id),
                    "proactive_signal": proactive_signal,
                    "general_opinion_fallback_used": bool(
                        recalled_pack.get("general_opinion_fallback_used")
                        or memory_meta.get("general_opinion_fallback_used")
                    ),
                })
        except Exception:
            pass

        style_catalog = self._build_comment_style_options(
            reply_depth=reply_depth,
            effective_affect_signal=effective_affect_signal,
        )
        forced_skip_reason = self._forum_forced_skip_reason(target_quality)
        if forced_skip_reason:
            style_selection = {
                "selected_style": "SKIP",
                "available_styles": style_catalog.get("style_names") or [],
                "quick_styles_allowed": bool(style_catalog.get("quick_styles_allowed")),
                "forced_skip_reason": forced_skip_reason,
            }
        else:
            style_selection = self._select_comment_style(
                conv,
                interests,
                reply_depth=reply_depth,
                target_quality=target_quality,
                return_metadata=True,
                memory_cues_block=memory_cues_block,
                memory_scope=(memory_cues.get("scope") if isinstance(memory_cues, dict) else "none"),
                memory_callback_hint=(
                    memory_cues.get("callback_hint") if isinstance(memory_cues, dict) else ""
                ),
                memory_argument_hint=(
                    memory_cues.get("argument_hint") if isinstance(memory_cues, dict) else ""
                ),
                memory_tone_hint=(memory_cues.get("tone_hint") if isinstance(memory_cues, dict) else ""),
                high_affect_flags=high_affect_flags,
                recalled_memories_block=recalled_memories_block,
                memory_usage_requirement=memory_usage_requirement,
                effective_affect_signal=effective_affect_signal,
                proactive_affect_block=proactive_affect_block,
            )
        comment_style = style_selection.get("selected_style") or "SKIP"
        self._log_comment_style_selection(
            tid=tid,
            post_id=int(post_id),
            thread_root_id=int(thread_root_id),
            reply_depth=reply_depth,
            target_quality=target_quality,
            style_selection=style_selection,
        )

        # Style skip: LLM decided there's nothing substantive to add
        if comment_style == "SKIP":
            logging.info("[%s] Style selector chose SKIP — declining to comment on post %s", self.name, post_id)
            return False

        # Step 2: Get style-specific prompt (fallback to handler_comment if not found)
        style_prompt_key = f"handler_comment_{comment_style}"
        if style_prompt_key not in self.prompts:
            style_prompt_key = "handler_comment"

        fresh_config = self._get_llm_config_for_write_action()

        u1 = AssistantAgent(
            name=f"{self.name}",
            llm_config=fresh_config,
            system_message=self.__effify(
                self.prompts["agent_roleplay_comments_share"], interests=interests
            ),
            max_consecutive_auto_reply=1,
        )

        u2 = AssistantAgent(
            name=f"Handler",
            llm_config=fresh_config,
            system_message=self.__effify(self.prompts["handler_instructions"]),
            max_consecutive_auto_reply=1,
        )

        conv_for_prompt = conv
        if proactive_affect_block:
            conv_for_prompt = proactive_affect_block + "\n\n" + conv_for_prompt
        if high_affect_flags and not subtle_memory_mode:
            conv_for_prompt = high_affect_flags + "\n\n" + conv_for_prompt
        if recalled_memories_block and not subtle_memory_mode:
            conv_for_prompt = recalled_memories_block + "\n\n" + conv_for_prompt
        if memory_usage_requirement and not subtle_memory_mode:
            conv_for_prompt = memory_usage_requirement + "\n\n" + conv_for_prompt
        if memory_cues_block and not subtle_memory_mode:
            conv_for_prompt = memory_cues_block + "\n\n" + conv_for_prompt
        if memory_plan_hint and not subtle_memory_mode:
            conv_for_prompt = memory_plan_hint + "\n\n" + conv_for_prompt

        comment_prompt = self.__effify(
            self.prompts[style_prompt_key],
            conv=conv_for_prompt,
            memory_cues_block=memory_cues_block,
            memory_scope=(memory_cues.get("scope") if isinstance(memory_cues, dict) else "none"),
            memory_callback_hint=(
                memory_cues.get("callback_hint") if isinstance(memory_cues, dict) else ""
            ),
            memory_argument_hint=(
                memory_cues.get("argument_hint") if isinstance(memory_cues, dict) else ""
            ),
            memory_tone_hint=(memory_cues.get("tone_hint") if isinstance(memory_cues, dict) else ""),
            memory_plan_hint=memory_plan_hint,
            high_affect_flags=high_affect_flags,
            recalled_memories_block=recalled_memories_block,
            proactive_affect_block=proactive_affect_block,
            memory_usage_requirement=memory_usage_requirement,
        )
        u2.initiate_chat(
            u1,
            message=comment_prompt,
            silent=True,
            max_turns=1,
        )

        emotion_raw = self._extract_emotion_chat_content(u2, u1)
        emotion_eval = self.__clean_emotion(emotion_raw)

        post_text = self._extract_generated_chat_content(
            u2, u1, prompt_hint=comment_prompt, skip_emotion_like=True
        )

        # cleaning the post text of some unwanted characters
        post_text = self.__clean_text(post_text)

        memory_required = bool(
            (not subtle_memory_mode)
            and bool(high_affect_signal.get("is_high_affect"))
            and bool(recalled_pack.get("has_usable_memories"))
            and bool(recalled_pack.get("items"))
        )
        callback_ok, callback_reason = self._memory_reply_references_recalled_item(
            post_text,
            recalled_pack.get("items"),
        )
        callback_retry_used = False
        callback_retry_ok = callback_ok

        max_retries = int(getattr(self, "memory_high_affect_callback_retry_count", 1) or 0)
        if subtle_memory_mode:
            max_retries = 0
        if memory_required and not callback_ok and max_retries > 0:
            callback_retry_used = True
            rewritten = self._memory_rewrite_reply_with_callback(
                draft_text=post_text,
                thread_context=conv_for_prompt,
                recalled_memories_block=recalled_memories_block,
                memory_usage_requirement=memory_usage_requirement,
                high_affect_flags=high_affect_flags,
                interests=interests,
            )
            if isinstance(rewritten, str) and rewritten.strip():
                post_text = rewritten.strip()
            callback_retry_ok, callback_reason = self._memory_reply_references_recalled_item(
                post_text,
                recalled_pack.get("items"),
            )

        try:
            self._decision_log(
                {
                    "decision_type": "memory_callback_check",
                    "run_id": getattr(self, "memory_run_id", None),
                    "agent_user_id": int(getattr(self, "user_id", -1) or -1),
                    "agent_name": getattr(self, "name", None),
                    "tid": int(tid),
                    "post_id": int(post_id),
                    "memory_required": bool(memory_required),
                    "callback_ok_first_pass": bool(callback_ok),
                    "callback_retry_used": bool(callback_retry_used),
                    "callback_ok_final": bool(callback_retry_ok),
                    "callback_reason": callback_reason,
                }
            )
        except Exception:
            pass

        post_text, length_meta = self._enforce_text_limits(
            text=post_text,
            mode="comment",
            context_text=conv_for_prompt,
            interests=interests,
        )
        try:
            self._decision_log(
                {
                    "decision_type": "length_enforcement",
                    "run_id": getattr(self, "memory_run_id", None),
                    "agent_user_id": int(getattr(self, "user_id", -1) or -1),
                    "agent_name": getattr(self, "name", None),
                    "tid": int(tid),
                    "post_id": int(post_id),
                    **length_meta,
                }
            )
        except Exception:
            pass

        # avoid posting empty messages or SKIP signals from the LLM
        if len(post_text) < 3 or post_text.strip().upper() == "SKIP":
            return False

        hashtags = self.__extract_components(post_text, c_type="hashtags")
        mentions = self.__extract_components(post_text, c_type="mentions")

        payload_text = (
            post_text.replace('"', "")
            .replace(f"{self.name}", "")
            .replace(":", "")
            .replace("*", "")
            .strip()
        )
        if len(payload_text) < 3:
            return False

        # Policy: within the same round and same parent, allow only if text differs.
        if self._has_recent_identical_comment(tid=tid, parent_post_id=post_id, text=payload_text):
            logging.info(
                "[%s] Skipping identical same-round reply on parent %s",
                self.name,
                post_id,
            )
            return False

        # Thread-level dedup: check if a near-duplicate already exists in the thread.
        try:
            if self._is_thread_duplicate(payload_text, conversation):
                logging.info(
                    "[%s] Skipping thread-duplicate comment on post %s",
                    self.name,
                    post_id,
                )
                return False
        except Exception:
            pass
        self._record_generated_comment(payload_text)

        client_action_id = self._build_comment_client_action_id(
            tid=tid,
            parent_post_id=post_id,
            text=payload_text,
        )

        # Ensure the parent author is notified even if the reply text doesn't include an @ mention.
        try:
            if isinstance(parent_author, str) and parent_author and parent_author != self.name:
                mention_token = f"@{parent_author.lstrip('@')}"
                if mention_token not in mentions:
                    mentions.append(mention_token)
        except Exception:
            pass

        st = json.dumps(
            {
                "user_id": self.user_id,
                "post_id": post_id,
                "text": payload_text,
                "emotions": emotion_eval,
                "hashtags": hashtags,
                "mentions": mentions,
                "tid": tid,
                "client_action_id": client_action_id,
            }
        )

        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        api_url = f"{self.base_url}/comment"
        response = post(f"{api_url}", headers=headers, data=st)
        if int(getattr(response, "status_code", 0) or 0) != 200:
            return False

        deduped = False
        try:
            parsed = json.loads(response.__dict__["_content"].decode("utf-8"))
            deduped = bool(parsed.get("deduped", False))
        except Exception:
            deduped = False

        self._record_recent_comment(tid=tid, parent_post_id=post_id, text=payload_text)

        if not deduped:
            self._record_writing_action()
            if proactive_triggered:
                self._consume_proactive_affect_budget()

        # Run-scoped memory: record interaction + update social/thread cards.
        try:
            target_text = reply_to_text
            if target_text is None:
                try:
                    target_text = self.__get_post(post_id)
                except Exception:
                    target_text = ""
            if other_user_id is not None and thread_root_id is not None:
                self._memory_after_comment(
                    tid=int(tid),
                    target_post_id=int(post_id),
                    thread_root_id=int(thread_root_id),
                    other_user_id=int(other_user_id),
                    other_username=str(other_username or parent_author or "").strip().lstrip("@"),
                    other_text=str(target_text or "").strip(),
                    my_text=str(post_text or "").strip(),
                    conv_text=str(conv_raw or ""),
                )
        except Exception:
            pass

        res = self.__evaluate_follow(post_text, post_id, "follow", tid)

        # update topic of interest with the ones from the post
        # get the root post id
        api_url = f"{self.base_url}/get_thread_root"
        response = get(
            f"{api_url}", headers=headers, data=json.dumps({"post_id": post_id})
        )
        data = json.loads(response.__dict__["_content"].decode("utf-8"))
        self.__update_user_interests(data, tid)

        # if not followed, test unfollow
        if res is None:
            self.__evaluate_follow(post_text, post_id, "unfollow", tid)

        return True

    def __update_user_interests(self, post_id, tid):
        """
        Update the user interests based on the post topics.

        :param post_id: id of the post
        :param tid: round id
        """
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        api_url = f"{self.base_url}/get_post_topics"
        data = {"post_id": post_id}
        response = get(f"{api_url}", headers=headers, data=json.dumps(data))
        data = json.loads(response.__dict__["_content"].decode("utf-8"))
        if len(data) > 0:
            api_url = f"{self.base_url}/set_user_interests"
            data = {"user_id": self.user_id, "interests": data, "round": tid}
            post(f"{api_url}", headers=headers, data=json.dumps(data))

    @log_execution_time
    def share(self, post_id: int, tid):
        """
        Share a post containing a news article.

        :param post_id: the post id
        :param tid: the round id
        :return: the response from the service
        """
        logging.info(
            "Agent %s: SHARE action skipped (disabled for forum experiments).",
            self.name,
        )
        return

        article = self.__get_article(post_id)
        if "status" in article:
            return

        post_text = self.__get_post(post_id)

        # obtain the most recent (and frequent) interests of the agent
        # interests, _ = self.__get_interests(tid)

        # get the post_id topics
        api_url = f"{self.base_url}/get_post_topics_name"
        response = get(f"{api_url}", headers={"Content-Type": "application/x-www-form-urlencoded"},
                       data=json.dumps({"post_id": post_id}))
        interests = json.loads(response.__dict__["_content"].decode("utf-8"))

        # get the opinion on the topics (if present)
        self.topics_opinions = ""
        if len(interests) > 0:
            # get recent sentiment on the selected interests
            api_url = f"{self.base_url}/get_sentiment"
            data = {"user_id": self.user_id, "interests": interests}
            response = post(f"{api_url}", headers={"Content-Type": "application/x-www-form-urlencoded"},
                            data=json.dumps(data))
            sentiment = json.loads(response.__dict__["_content"].decode("utf-8"))

            self.topics_opinions = "Your opinion topics of the post you are responding to are: "
            for s in sentiment:
                self.topics_opinions += f"{s['topic']}: {s['sentiment']} "
            if len(sentiment) == 0:
                self.topics_opinions = ""
        else:
            interests, _ = self.__get_interests(tid)

        # Step 1: Select share style using LLM
        article_title = article.get('title', '') if isinstance(article, dict) else str(article)
        article_summary = article.get('summary', '') if isinstance(article, dict) else ''
        share_style = self._select_share_style(article_title, article_summary, interests)

        # Step 2: Get style-specific prompt (fallback to handler_share if not found)
        style_prompt_key = f"handler_share_{share_style}"
        if style_prompt_key not in self.prompts:
            style_prompt_key = "handler_share"

        fresh_config = self._get_llm_config_for_write_action()

        u1 = AssistantAgent(
            name=f"{self.name}",
            llm_config=fresh_config,
            system_message=self.__effify(
                self.prompts["agent_roleplay_comments_share"], interests=interests
            ),
            max_consecutive_auto_reply=1,
        )

        u2 = AssistantAgent(
            name=f"Handler",
            llm_config=fresh_config,
            system_message=self.__effify(self.prompts["handler_instructions"]),
            max_consecutive_auto_reply=1,
        )

        share_prompt = self.__effify(
            self.prompts[style_prompt_key],
            article=article,
            article_title=article_title,
            article_summary=article_summary,
            post_text=post_text
        )
        u2.initiate_chat(
            u1,
            message=share_prompt,
            silent=True,
            max_turns=1,
        )

        emotion_raw = self._extract_emotion_chat_content(u2, u1)
        emotion_eval = self.__clean_emotion(emotion_raw)

        post_text = self._extract_generated_chat_content(
            u2, u1, prompt_hint=share_prompt, skip_emotion_like=True
        )

        post_text = (
            post_text.split(":")[-1]
            .split("-")[-1]
            .replace("@ ", "")
            .replace("  ", " ")
            .replace(". ", ".")
            .replace(" ,", ",")
            .replace("[", "")
            .replace("]", "")
            .replace("@,", "")
        )
        post_text = post_text.replace(f"@{self.name}", "")

        # Strip reproduced article content from LLM output
        post_text = self._strip_reproduced_article_content(post_text, article_summary)
        # Avoid markdown emphasis causing bold rendering in the UI.
        post_text = post_text.replace("*", "")

        hashtags = self.__extract_components(post_text, c_type="hashtags")
        mentions = self.__extract_components(post_text, c_type="mentions")

        st = json.dumps(
            {
                "user_id": self.user_id,
                "post_id": post_id,
                "text": post_text.replace('"', ""),
                "emotions": emotion_eval,
                "hashtags": hashtags,
                "mentions": mentions,
                "tid": tid,
            }
        )

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        api_url = f"{self.base_url}/share"
        post(f"{api_url}", headers=headers, data=st)
        self._record_writing_action()

    @log_execution_time
    def reaction(self, post_id: int, tid: int, check_follow=True):
        """
        Generate a reaction to a post/comment.

        :param post_id: the post id
        :param tid: the round id
        :param check_follow: whether to evaluate a follow cascade action
        :return: the response from the service
        """

        post_text = self.__get_post(post_id)

        # Memory context can influence voting/reaction decisions.
        post_text_for_prompt = post_text
        try:
            if getattr(self, "memory_enabled", False):
                other_user_id, other_username = self._memory_get_author_id_and_username(int(post_id))
                thread_root_id = self._memory_get_thread_root_id(int(post_id))
                query_text = self._memory_build_query_text(
                    "decide reaction: upvote/downvote/neutral",
                    post_text,
                    f"author={other_username or ''}",
                )
                mem_text, _ = self._memory_build_tiered_context(
                    query_text=query_text,
                    other_user_id=other_user_id,
                    thread_root_id=thread_root_id,
                    other_username=other_username,
                    round_id=tid,
                    uncertainty_score=0.65,
                )
                if mem_text:
                    post_text_for_prompt = mem_text + "\n\nTARGET POST:\n" + str(post_text)
        except Exception:
            post_text_for_prompt = post_text

        u1 = AssistantAgent(
            name=f"{self.name}",
            llm_config=self.llm_config,
            system_message=self.__effify(self.prompts["agent_roleplay_simple"]),
            max_consecutive_auto_reply=1,
        )

        u2 = AssistantAgent(
            name=f"Handler",
            llm_config=self.llm_config,
            system_message=self.__effify(self.prompts["handler_instructions_simple"]),
            max_consecutive_auto_reply=0,
        )

        u2.initiate_chat(
            u1,
            message=self.__effify(
                self.prompts["handler_reactions"], post_text=post_text_for_prompt
            ),
            silent=True,
            max_turns=1,
        )

        text = (u1.chat_messages[u2][-1]["content"] or "").replace("!", "")

        u1.reset()
        u2.reset()

        # Be robust to punctuation like "YES." or "NO,".
        tokens = set(re.findall(r"[A-Z]+", text.upper()))

        if "YES" in tokens:
            st = json.dumps(
                {
                    "user_id": self.user_id,
                    "post_id": post_id,
                    "type": "like",
                    "tid": tid,
                }
            )
            flag = "follow"

        elif "NO" in tokens:
            st = json.dumps(
                {
                    "user_id": self.user_id,
                    "post_id": post_id,
                    "type": "dislike",
                    "tid": tid,
                }
            )
            flag = "unfollow"
            # always evaluate unfollow in case of dislike
            self.__evaluate_follow(post_text, post_id, flag, tid)
        elif "NEUTRAL" in tokens:
            try:
                self._decision_log(
                    {
                        "decision_type": "reaction_decision",
                        "run_id": getattr(self, "memory_run_id", None),
                        "agent_user_id": int(getattr(self, "user_id", -1) or -1),
                        "agent_name": getattr(self, "name", None),
                        "tid": int(tid),
                        "post_id": int(post_id),
                        "llm_raw": text,
                        "tokens": sorted(list(tokens))[:20],
                        "selected_action": "neutral",
                    }
                )
            except Exception:
                pass
            return
        else:
            try:
                self._decision_log(
                    {
                        "decision_type": "reaction_decision",
                        "run_id": getattr(self, "memory_run_id", None),
                        "agent_user_id": int(getattr(self, "user_id", -1) or -1),
                        "agent_name": getattr(self, "name", None),
                        "tid": int(tid),
                        "post_id": int(post_id),
                        "llm_raw": text,
                        "tokens": sorted(list(tokens))[:20],
                        "selected_action": "invalid",
                    }
                )
            except Exception:
                pass
            return

        try:
            self._decision_log(
                {
                    "decision_type": "reaction_decision",
                    "run_id": getattr(self, "memory_run_id", None),
                    "agent_user_id": int(getattr(self, "user_id", -1) or -1),
                    "agent_name": getattr(self, "name", None),
                    "tid": int(tid),
                    "post_id": int(post_id),
                    "llm_raw": text,
                    "tokens": sorted(list(tokens))[:20],
                    "selected_action": "like" if flag == "follow" else "dislike",
                }
            )
        except Exception:
            pass

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        api_url = f"{self.base_url}/reaction"
        post(f"{api_url}", headers=headers, data=st)

        # Run-scoped memory: record this vote.
        try:
            self._memory_after_vote(
                tid=int(tid),
                post_id=int(post_id),
                vote_type="like" if flag == "follow" else "dislike",
            )
        except Exception:
            pass

        # evaluate follow only upon explicit request
        if check_follow and flag == "follow":
            self.__evaluate_follow(post_text, post_id, flag, tid)

        # update user interests after reaction
        self.__update_user_interests(post_id, tid)

    def vote(self, post_id: int, tid: int, vote_type: str) -> bool:
        """
        Cast an explicit upvote/downvote (like/dislike) without an additional LLM step.

        Intended for mention notification handling where the LLM already decided the action.
        """
        if vote_type not in ("like", "dislike"):
            return False

        st = json.dumps(
            {
                "user_id": self.user_id,
                "post_id": int(post_id),
                "type": vote_type,
                "tid": int(tid),
            }
        )

        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        api_url = f"{self.base_url}/reaction"

        try:
            post(f"{api_url}", headers=headers, data=st)
        except Exception:
            return False

        # Update interests based on the thread root so reactions reinforce topics.
        try:
            api_url = f"{self.base_url}/get_thread_root"
            response = get(
                f"{api_url}", headers=headers, data=json.dumps({"post_id": int(post_id)})
            )
            root_id = json.loads(response.__dict__["_content"].decode("utf-8"))
            self.__update_user_interests(root_id, tid)
        except Exception:
            pass

        # Run-scoped memory: record vote + update social card + digest.
        try:
            self._memory_after_vote(tid=int(tid), post_id=int(post_id), vote_type=str(vote_type))
        except Exception:
            pass

        return True

    def __evaluate_follow(self, post_text, post_id, action, tid):
        """
        Evaluate a follow action.

        :param post_text: the post text
        :param post_id: the post id
        :param action: the action, either follow or unfollow
        :param tid: the round id
        :return: the response from the service
        """

        u1 = AssistantAgent(
            name=f"{self.name}",
            llm_config=self.llm_config,
            system_message=self.__effify(self.prompts["agent_roleplay_simple"]),
            max_consecutive_auto_reply=1,
        )

        u2 = AssistantAgent(
            name=f"Handler",
            llm_config=self.llm_config,
            system_message=self.__effify(self.prompts["handler_instructions_simple"]),
            max_consecutive_auto_reply=0,
        )

        u2.initiate_chat(
            u1,
            message=self.__effify(
                self.prompts["handler_follow"], post_text=post_text, action=action
            ),
            silent=True,
            max_turns=1,
        )

        text = (u1.chat_messages[u2][-1]["content"] or "").replace("!", "")

        u1.reset()
        u2.reset()

        tokens = set(re.findall(r"[A-Z]+", text.upper()))
        if "YES" in tokens:
            if action == "follow":
                # follow with a probability of 0.2 (@ToDo: make this a parameter?)
                if np.random.rand() < 0.2:
                    self.follow(post_id=post_id, action=action, tid=tid)
                    return action
            else:
                self.follow(post_id=post_id, action=action, tid=tid)
                return action
        else:
            return None

    @log_execution_time
    def follow(
        self, tid: int, target: int = None, post_id: int = None, action="follow"
    ):
        """
        Follow a user

        :param tid: the round id
        :param action: the action, either follow or unfollow
        :param post_id: the post id
        :param post_id: the post id
        :param target: the target user id
        """

        if post_id is not None:
            # YServerReddit /get_user_from_post returns username (string), but /follow expects user_id.
            uid, _uname = self.get_username_from_post(post_id)
            target = uid

        if target is None:
            return

        st = json.dumps(
            {
                "user_id": self.user_id,
                "target": int(target),
                "action": action,
                "tid": tid,
            }
        )

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        api_url = f"{self.base_url}/follow"
        post(f"{api_url}", headers=headers, data=st)

    def followers(self):
        """
        Get the followers of the user.

        :return: the response from the service
        """

        st = json.dumps({"user_id": self.user_id})

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        api_url = f"{self.base_url}/followers"
        response = get(f"{api_url}", headers=headers, data=st)

        return response.__dict__["_content"].decode("utf-8")

    def timeline(self):
        """
        Get the timeline of the user.

        :return: the response from the service
        """

        st = json.dumps({"user_id": self.user_id})

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        api_url = f"{self.base_url}/timeline"
        response = get(f"{api_url}", headers=headers, data=st)

        return response.__dict__["_content"].decode("utf-8")

    @log_execution_time
    def cast(self, post_id: int, tid: int):
        """
        Cast a voting intention (political simulation)

        :param post_id: the post id
        :param tid: the round id
        :return: the response from the service
        """

        post_text = self.__get_post(post_id)

        u1 = AssistantAgent(
            name=f"{self.name}",
            llm_config=self.llm_config,
            system_message=self.__effify(self.prompts["agent_roleplay_simple"]),
            max_consecutive_auto_reply=1,
        )

        u2 = AssistantAgent(
            name=f"Handler",
            llm_config=self.llm_config,  # self.llm_config,
            system_message=self.__effify(self.prompts["handler_instructions_simple"]),
            max_consecutive_auto_reply=0,
        )

        u2.initiate_chat(
            u1,
            message=self.__effify(self.prompts["handler_cast"], post_text=post_text),
            silent=True,
            max_turns=1,
        )

        text = u1.chat_messages[u2][-1]["content"].replace("!", "").upper()

        u1.reset()
        u2.reset()

        data = {
            "user_id": self.user_id,
            "post_id": post_id,
            "content_type": "Post",
            "tid": tid,
            "content_id": post_id,
        }

        if "RIGHT" in text.split():
            data["vote"] = "R"
            st = json.dumps(data)

        elif "LEFT" in text.split():
            data["vote"] = "D"
            st = json.dumps(data)

        elif "NONE" in text.split():
            data["vote"] = "U"
            st = json.dumps(data)
        else:
            return

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        api_url = f"{self.base_url}/cast_preference"
        post(f"{api_url}", headers=headers, data=st)

    def churn_system(self, tid):
        """
        Leave the system.

        :return:
        """
        st = json.dumps({"user_id": self.user_id, "left_on": tid})

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        api_url = f"{self.base_url}/churn"
        response = post(f"{api_url}", headers=headers, data=st)

        return response.__dict__["_content"].decode("utf-8")

    @log_execution_time
    def select_action(self, tid, actions, max_length_thread_reading=10):
        """
        Post a message to the service.

        :param actions: The list of actions to select from.
        :param tid: The time id.
        :param max_length_thread_reading: The maximum length of the thread to read.
        """
        # Forum experiments do not allow repost/share of other users' content.
        actions = [a for a in actions if str(a).upper() != "SHARE"]
        if not actions:
            actions = ["NONE"]

        np.random.shuffle(actions)
        acts = ",".join(actions)

        u1 = AssistantAgent(
            name=f"{self.name}",
            llm_config=self.llm_config,
            system_message=self.__effify(self.prompts["agent_roleplay_base"]),
            max_consecutive_auto_reply=1,
        )

        u2 = AssistantAgent(
            name=f"Handler",
            llm_config=self.llm_config,
            system_message=self.__effify(self.prompts["handler_instructions_simple"]),
            max_consecutive_auto_reply=0,
        )

        u2.initiate_chat(
            u1,
            message=self.__effify(self.prompts["handler_action"], actions=acts),
            silent=True,
            max_turns=1,
        )

        text = u1.chat_messages[u2][-1]["content"].replace("!", "").upper()
        u1.reset()
        u2.reset()

        if "COMMENT" in text.split():
            candidates = json.loads(self.read())
            if len(candidates) > 0:
                seed_post_id = int(random.sample(candidates, min(1, len(candidates)))[0])
                target = self._select_comment_target_via_thread_browse(seed_post_id, tid=tid)

                # If the agent decided not to comment after reading, still register an engagement vote.
                if target is None:
                    self.reaction(int(seed_post_id), check_follow=False, tid=tid)
                else:
                    target_post_id = int(target.get("post_id", seed_post_id))
                    comment_success = self.comment(
                        int(target_post_id),
                        max_length_threads=max_length_thread_reading,
                        tid=tid,
                        reply_to_username=target.get("username"),
                        reply_to_text=target.get("text"),
                        reply_target_depth=target.get("depth"),
                        reply_target_meta=target.get("quality_meta"),
                    )
                    if comment_success:
                        self.reaction(int(target_post_id), check_follow=False, tid=tid)
                    else:
                        self.reaction(int(seed_post_id), check_follow=False, tid=tid)

        elif "POST" in text.split():
            self.post(tid=tid)

        elif "READ" in text.split():
            candidates = json.loads(self.read())
            try:
                selected_post = random.sample(candidates, min(1, len(candidates)))
                self.reaction(int(selected_post[0]), tid=tid)
            except:
                pass

        # elif "REPLY" in text.split():
        #    selected_post = json.loads(self.read_mentions())
        #    if "status" not in selected_post:
        #        self.comment(
        #            int(selected_post[0]),
        #            max_length_threads=max_length_thread_reading,
        #            tid=tid,
        #        )

        elif "SEARCH" in text.split():
            candidates = json.loads(self.search())
            if "status" not in candidates and len(candidates) > 0:
                seed_post_id = int(random.sample(candidates, min(1, len(candidates)))[0])
                target = self._select_comment_target_via_thread_browse(seed_post_id, tid=tid)

                if target is None:
                    self.reaction(int(seed_post_id), check_follow=False, tid=tid)
                else:
                    target_post_id = int(target.get("post_id", seed_post_id))
                    comment_success = self.comment(
                        int(target_post_id),
                        max_length_threads=max_length_thread_reading,
                        tid=tid,
                        reply_to_username=target.get("username"),
                        reply_to_text=target.get("text"),
                        reply_target_depth=target.get("depth"),
                        reply_target_meta=target.get("quality_meta"),
                    )
                    if comment_success:
                        self.reaction(int(target_post_id), check_follow=False, tid=tid)
                    else:
                        self.reaction(int(seed_post_id), check_follow=False, tid=tid)

        elif "FOLLOW" in text.split():
            candidates = self.search_follow()
            if len(candidates) > 0:
                tot = sum([float(v) for v in candidates.values()])
                probs = [v / tot for v in candidates.values()]
                selected = np.random.choice(
                    [int(c) for c in candidates],
                    p=probs,
                    size=1,
                )[0]
                self.follow(tid=tid, target=selected, action="follow")

        # demanded to page agents
        # elif "NEWS" in text.split():
        #    news, website = self.select_news()
        #    if not isinstance(news, str):
        #        self.news(tid=tid, article=news, website=website)

        elif "SHARE_LINK" in text.split():
            article, website = self.select_link(tid=tid)
            if article and website and not isinstance(article, str):
                self.share_link(tid=tid, article=article, website=website)
            else:
                logging.info(
                    "Agent %s: No suitable article found for SHARE_LINK; skipping action.",
                    self.name,
                )

        elif "SHARE_IMAGE" in text.split():
            image_post = self.select_standalone_image(tid=tid)
            if image_post:
                self.share_image(tid=tid, image_post=image_post)
            else:
                logging.info(
                    "Agent %s: No suitable image found for SHARE_IMAGE; skipping action.",
                    self.name,
                )

        elif "SHARE" in text.split():
            logging.info(
                "Agent %s: LLM selected SHARE, but share is disabled; skipping.",
                self.name,
            )

        elif "CAST" in text.split():
            candidates = json.loads(self.read())
            try:
                selected_post = random.sample(candidates, min(1, len(candidates)))
                self.cast(int(selected_post[0]), tid=tid)
            except:
                pass

        elif "IMAGE" in text.split():
            image, article_id = self.select_image(tid=tid)
            if image is not None:
                self.comment_image(image, tid=tid, article_id=article_id)

        return

    @log_execution_time
    def reply(self, tid: int, max_length_thread_reading: int = 10):
        """
        Handle a mention notification (reply vs upvote/downvote/ignore) with guardrails to prevent
        infinite A-B-A loops.

        :param tid: current round/time id
        :param max_length_thread_reading: maximum depth of thread context to read
        """
        selected_post = json.loads(self.read_mentions())

        post_id = None
        if isinstance(selected_post, dict):
            if selected_post.get("status") == 404:
                return
            post_id = selected_post.get("post_id")
        elif isinstance(selected_post, list):
            if not selected_post:
                return
            post_id = selected_post[0]
        else:
            return

        if post_id is None:
            return

        try:
            post_id = int(post_id)
        except (TypeError, ValueError):
            return

        # Fetch mention/comment details for decision + accurate reply targeting.
        mention_text = None
        try:
            mention_text = self.__get_post(post_id)
        except Exception:
            mention_text = None

        mention_user_id, mention_username = (None, None)
        try:
            mention_user_id, mention_username = self.get_username_from_post(post_id)
        except Exception:
            mention_user_id, mention_username = (None, None)

        if not mention_username:
            try:
                mention_username = self.get_user_from_post(post_id)
            except Exception:
                mention_username = None

        # Thread root id for depth guardrail (fallback to post_id).
        thread_root_id = None
        try:
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            api_url = f"{self.base_url}/get_thread_root"
            response = get(
                f"{api_url}",
                headers=headers,
                data=json.dumps({"post_id": int(post_id)}),
            )
            thread_root_id = json.loads(response.__dict__["_content"].decode("utf-8"))
            thread_root_id = int(thread_root_id)
        except Exception:
            thread_root_id = int(post_id)

        other_user_key = mention_user_id if mention_user_id is not None else mention_username
        subtle_memory_mode = self._memory_use_subtle_prompt_mode()

        # Hard guardrails for replying (votes/ignore still allowed).
        budget_ok = self.replies_this_round < self.max_replies_per_round

        cooldown_ok = True
        if other_user_key is not None:
            last_reply_round = self.last_replied_to.get(other_user_key, -999)
            cooldown_ok = (tid - last_reply_round) >= self.reply_cooldown_rounds

        depth_ok = True
        if other_user_key is not None and thread_root_id is not None:
            depth_key = (int(thread_root_id), other_user_key)
            depth_ok = self.reply_chain_depths.get(depth_key, 0) < int(
                self.max_reply_chain_depth
            )

        # Absolute depth limit and thread comment cap: both require the thread tree.
        abs_depth_ok = True
        thread_cap_ok = True
        tree_data = None
        max_abs_depth = int(getattr(self, "max_absolute_reply_depth", 0) or 0)
        max_comments = int(getattr(self, "max_comments_per_thread", 0) or 0)
        if (max_abs_depth > 0 or max_comments > 0) and thread_root_id is not None:
            try:
                tree_data = self.__get_thread_tree(int(thread_root_id), limit=200)
            except Exception:
                tree_data = None

            if isinstance(tree_data, dict) and tree_data.get("posts"):
                posts_list = tree_data["posts"]

                # Thread comment cap
                if max_comments > 0 and len(posts_list) >= max_comments:
                    thread_cap_ok = False

                # Absolute depth computation
                if max_abs_depth > 0:
                    try:
                        parent_map = {}
                        for p in posts_list:
                            try:
                                pid = int(p.get("post_id"))
                                ct = p.get("comment_to")
                                parent_map[pid] = int(ct) if ct is not None and int(ct) > 0 else None
                            except Exception:
                                pass
                        current = int(post_id)
                        post_depth = 0
                        visited = set()
                        while current in parent_map and parent_map[current] is not None and current not in visited:
                            visited.add(current)
                            current = parent_map[current]
                            post_depth += 1
                        if post_depth >= max_abs_depth:
                            abs_depth_ok = False
                    except Exception:
                        pass

        reply_allowed = budget_ok and cooldown_ok and depth_ok and abs_depth_ok and thread_cap_ok

        reply_block_reasons = []
        if not budget_ok:
            reply_block_reasons.append("per-round reply budget reached")
        if not cooldown_ok:
            reply_block_reasons.append("cooldown active for this user")
        if not depth_ok:
            reply_block_reasons.append("max reply chain depth reached for this thread+user")
        if not abs_depth_ok:
            reply_block_reasons.append("absolute reply depth limit reached")
        if not thread_cap_ok:
            reply_block_reasons.append("thread comment cap reached")

        # Build a compact thread context for the decision model.
        try:
            thread_lines = self.__get_thread(post_id, max_tweets=max_length_thread_reading)
        except Exception:
            thread_lines = []

        if mention_username and mention_text:
            thread_lines = list(thread_lines)
            thread_lines.append(f"@{mention_username} - {str(mention_text).strip()}\n")

        thread_context = "".join(thread_lines)
        max_chars = (
            getattr(self, "max_thread_context_chars", DEFAULT_MAX_THREAD_CONTEXT_CHARS)
            or DEFAULT_MAX_THREAD_CONTEXT_CHARS
        )
        if max_chars > 0 and len(thread_context) > max_chars:
            thread_context = thread_context[-max_chars:]

        # Memory context can influence whether we reply vs upvote/downvote/ignore.
        memory_injected = False
        memory_preview = ""
        memory_meta = {}
        memory_cues = {}
        memory_cues_block = ""
        try:
            if getattr(self, "memory_enabled", False):
                query_text = self._memory_build_query_text(
                    "decide mention action",
                    mention_text or "",
                    thread_context,
                    f"mention_author={mention_username or ''}",
                )
                if subtle_memory_mode:
                    mem_text, memory_meta = self._memory_build_reply_context(
                        query_text=query_text,
                        other_user_id=mention_user_id,
                        thread_root_id=thread_root_id,
                        other_username=mention_username,
                        round_id=tid,
                    )
                else:
                    mem_text, memory_meta = self._memory_build_tiered_context(
                        query_text=query_text,
                        other_user_id=mention_user_id,
                        thread_root_id=thread_root_id,
                        other_username=mention_username,
                        round_id=tid,
                        uncertainty_score=0.75,
                    )
                if mem_text:
                    memory_injected = True
                    memory_preview = mem_text
                    if not subtle_memory_mode:
                        thread_context = mem_text + "\n\n" + thread_context
        except Exception:
            pass

        try:
            if getattr(self, "memory_nuance_enabled", True):
                memory_cues = self._memory_build_conversation_cues(
                    memory_text=memory_preview,
                    memory_meta=memory_meta,
                    target_username=mention_username,
                    mode="reply_decision",
                )
                memory_cues_block = self._memory_format_conversation_cues(memory_cues)
                if memory_cues_block:
                    memory_preview = (
                        memory_cues_block + ("\n\n" + memory_preview if memory_preview else "")
                    )
                    memory_injected = True
                    if not subtle_memory_mode:
                        thread_context = memory_cues_block + "\n\n" + thread_context
        except Exception:
            memory_cues = {}
            memory_cues_block = ""

        try:
            self._decision_log(
                {
                    "decision_type": "reply_memory_mode",
                    "run_id": getattr(self, "memory_run_id", None),
                    "agent_user_id": int(getattr(self, "user_id", -1) or -1),
                    "agent_name": getattr(self, "name", None),
                    "tid": int(tid),
                    "post_id": int(post_id),
                    "mode": "mention_decision",
                    "memory_prompt_mode": "subtle_forum" if subtle_memory_mode else "legacy",
                    "cross_thread_callback_used": bool(
                        memory_meta.get("cross_thread_callback_candidate")
                    ),
                    "continuity_text": self._decision_compact_text(
                        memory_meta.get("continuity_text"), 320
                    ),
                }
            )
        except Exception:
            pass

        # Proactive affect detection for mention/reply path
        proactive_signal = {
            "is_proactive_high_affect": False,
            "mode": None,
            "confidence": 0.0,
            "reasons": [],
            "dominant_view_detected": False,
        }
        proactive_affect_block = ""
        proactive_triggered = False
        try:
            if getattr(self, "proactive_affect_enabled", True):
                proactive_signal = self._detect_proactive_high_affect_signal(
                    thread_context=thread_context,
                    round_id=tid,
                )
                if proactive_signal.get("is_proactive_high_affect"):
                    proactive_affect_block = self._format_proactive_affect_block(proactive_signal)
                    proactive_triggered = True
        except Exception:
            proactive_affect_block = ""

        # Get current interests as additional signal for the decision model.
        try:
            current_interests, _ = self.__get_interests(tid)
        except Exception:
            current_interests = []

        action = "ignore"
        decision_raw = ""
        decision_choice = None
        options_for_log = []
        decision_prompt_key = "handler_mention_action_decision"
        decision_fallback_reason = ""

        # LLM chooses between reply/upvote/downvote/ignore.
        if not self.prompts:
            action = "ignore"
            decision_fallback_reason = "no_prompts"
        else:
            prompt = self.prompts.get("handler_mention_action_decision")
            if not prompt:
                prompt = (
                    "You have a Reddit notification: someone mentioned you (or replied to you).\n\n"
                    "Decide how to respond based on the content, your persona, and your interests.\n"
                    "General guidance: if you agree but have nothing to add: upvote; if you strongly disagree or think it's misinformation: downvote; if you don't care: ignore.\n"
                    "Output ONLY the number of your chosen action.\n\n"
                    "MENTION AUTHOR: {mention_author}\n"
                    "MENTION TEXT:\n{mention_text}\n\n"
                    "THREAD CONTEXT:\n{thread_context}\n\n"
                    "YOUR INTERESTS: {interests_str}\n"
                    "REPLY ALLOWED: {reply_allowed}\n"
                    "REPLY BLOCK REASONS: {reply_block_reasons}\n\n"
                    "ACTIONS:\n{options}\n\n"
                    "Remember: output ONLY a single number."
                )

            options = []
            if reply_allowed:
                options.append(("reply", "Reply with a comment addressing the mention author."))
            options.extend(
                [
                    ("upvote", "Upvote the mention comment."),
                    ("downvote", "Downvote the mention comment."),
                    ("ignore", "Ignore it and clear the notification."),
                ]
            )
            random.shuffle(options)

            options_text = "\n".join(
                [f"{i}. {opt[0].upper()} - {opt[1]}" for i, opt in enumerate(options, start=1)]
            )
            options_for_log = [{"action": opt[0], "desc": opt[1]} for opt in options]

            interests_str = ", ".join(current_interests) if current_interests else "none"
            reply_block_reasons_str = (
                ", ".join(reply_block_reasons) if reply_block_reasons else "none"
            )

            decision_config = self.__get_fresh_llm_config()
            decision_config["temperature"] = 0.2
            decision_config["max_tokens"] = 30

            u1 = AssistantAgent(
                name=f"{self.name}",
                llm_config=decision_config,
                system_message=self.__effify(
                    self.prompts.get(
                        "agent_roleplay_comments_share",
                        self.prompts.get("agent_roleplay_simple", ""),
                    ),
                    interests=current_interests,
                ),
                # Allow a single LLM response for the decision.
                max_consecutive_auto_reply=1,
            )

            u2 = AssistantAgent(
                name="Handler",
                llm_config=decision_config,
                system_message=self.__effify(
                    self.prompts.get(
                        "handler_instructions_simple",
                        "You are the Handler that specifies the actions to be taken.",
                    )
                ),
                max_consecutive_auto_reply=0,
            )

            u2.initiate_chat(
                u1,
                message=self.__effify(
                    prompt,
                    mention_author=mention_username or "unknown",
                    mention_text=mention_text or "",
                    thread_context=thread_context,
                    interests_str=interests_str,
                    reply_allowed=reply_allowed,
                    reply_block_reasons=reply_block_reasons_str,
                    options=options_text,
                    memory_cues_block=memory_cues_block,
                    memory_scope=(
                        memory_cues.get("scope") if isinstance(memory_cues, dict) else "none"
                    ),
                    memory_callback_hint=(
                        memory_cues.get("callback_hint") if isinstance(memory_cues, dict) else ""
                    ),
                    memory_argument_hint=(
                        memory_cues.get("argument_hint") if isinstance(memory_cues, dict) else ""
                    ),
                    memory_tone_hint=(
                        memory_cues.get("tone_hint") if isinstance(memory_cues, dict) else ""
                    ),
                    proactive_affect_block=proactive_affect_block,
                ),
                silent=True,
                max_turns=1,
            )

            raw = ""
            try:
                raw = u1.chat_messages[u2][-1]["content"].strip()
            except Exception:
                raw = ""
            decision_raw = raw

            u1.reset()
            u2.reset()

            # Prefer strict "number only" output, but be resilient to minor formatting (e.g. "2.").
            m = re.match(r"^\s*(\d+)\s*$", raw) or re.search(r"\b(\d+)\b", raw)
            if m:
                try:
                    choice = int(m.group(1))
                except Exception:
                    choice = None
                decision_choice = choice
                if choice is not None and 1 <= choice <= len(options):
                    action = options[choice - 1][0]
                else:
                    action = "ignore"
            else:
                action = "ignore"

        # Structured decision log (best-effort).
        try:
            self._decision_log(
                {
                    "decision_type": "mention_action_decision",
                    "run_id": getattr(self, "memory_run_id", None),
                    "agent_user_id": int(getattr(self, "user_id", -1) or -1),
                    "agent_name": getattr(self, "name", None),
                    "tid": int(tid),
                    "prompt_key": decision_prompt_key,
                    "fallback_reason": decision_fallback_reason,
                    "mention_post_id": int(post_id),
                    "mention_user_id": int(mention_user_id) if mention_user_id is not None else None,
                    "mention_username": mention_username,
                    "mention_text": mention_text if isinstance(mention_text, str) else (str(mention_text) if mention_text is not None else ""),
                    "thread_root_id": int(thread_root_id) if thread_root_id is not None else None,
                    "reply_allowed": bool(reply_allowed),
                    "reply_block_reasons": list(reply_block_reasons),
                    "memory_injected": bool(memory_injected),
                    "context_preview": memory_preview,
                    "memory_search_used": bool(memory_meta.get("search_used", False)) if isinstance(memory_meta, dict) else False,
                    "memory_tier_c_used": bool(memory_meta.get("tier_c_used", False)) if isinstance(memory_meta, dict) else False,
                    "memory_retrieved_item_count": int(memory_meta.get("retrieved_item_count", 0) or 0) if isinstance(memory_meta, dict) else 0,
                    "memory_scope": memory_cues.get("scope") if isinstance(memory_cues, dict) else "none",
                    "memory_callback_hint": memory_cues.get("callback_hint") if isinstance(memory_cues, dict) else "",
                    "memory_argument_hint": memory_cues.get("argument_hint") if isinstance(memory_cues, dict) else "",
                    "memory_tone_hint": memory_cues.get("tone_hint") if isinstance(memory_cues, dict) else "",
                    "memory_callback_enabled": bool(memory_cues.get("should_callback", False)) if isinstance(memory_cues, dict) else False,
                    "options": options_for_log,
                    "llm_raw": decision_raw,
                    "parsed_choice": decision_choice,
                    "selected_action": action,
                }
            )
        except Exception:
            pass

        if action == "reply":
            comment_success = self.comment(
                post_id,
                max_length_threads=max_length_thread_reading,
                tid=tid,
                reply_to_username=mention_username,
                reply_to_text=mention_text,
            )
            if not comment_success:
                return

            self.replies_this_round += 1

            if other_user_key is not None:
                self.last_replied_to[other_user_key] = tid
                if thread_root_id is not None:
                    depth_key = (int(thread_root_id), other_user_key)
                    self.reply_chain_depths[depth_key] = self.reply_chain_depths.get(depth_key, 0) + 1

            return

        if action == "upvote":
            vote_success = self.vote(post_id=post_id, tid=tid, vote_type="like")
            if vote_success and proactive_triggered:
                self._consume_proactive_affect_budget()
            return

        if action == "downvote":
            vote_success = self.vote(post_id=post_id, tid=tid, vote_type="dislike")
            if vote_success and proactive_triggered:
                self._consume_proactive_affect_budget()
            return

        # ignore (or any unknown action): the server already cleared the notification on read
        return

    @log_execution_time
    def read(self, article=False):
        """
        Read n_posts from the service.

        :param article: whether to read an article or not
        :return: the response from the service
        """
        return self.content_rec_sys.read(self.base_url, self.user_id, article)

    def read_mentions(self):
        """
        Read n_posts from the service.

        :return: the response from the service
        """
        return self.content_rec_sys.read_mentions(self.base_url)

    @log_execution_time
    def search(self):
        """
        Read n_posts from the service.

        :return: the response from the service
        """
        return self.content_rec_sys.search(self.base_url)

    @log_execution_time
    def search_follow(self):
        """
        Read n_posts from the service.

        :return: the response from the service
        """
        return self.follow_rec_sys.follow_suggestions(self.base_url)

    def select_news(self):
        """
        Select a news article from the service by directly querying the database.

        :return: a tuple containing (News object, Website object)
        """
        from y_client.news_feeds.feed_reader import News

        # First try to find websites with the same leaning as the agent
        website_ids = session.query(Websites.id).filter(Websites.leaning == self.leaning).all()

        # If no matching leaning, get all website IDs
        if len(website_ids) == 0:
            website_ids = session.query(Websites.id).all()

        if len(website_ids) == 0:
            return "", ""

        # Extract IDs from query result
        website_ids = [w[0] for w in website_ids]

        # Query for articles from these websites
        articles = session.query(Articles).filter(
            Articles.website_id.in_(website_ids)
        ).order_by(func.random()).limit(10).all()

        if not articles:
            return "", ""

        # Select a random article from the results
        article = random.choice(articles)

        # Get the website for this article
        website = session.query(Websites).filter(Websites.id == article.website_id).first()

        if not website:
            return "", ""

        # Create a News object from the article data
        news_obj = News(
            title=article.title,
            summary=article.summary,
            link=article.link,
            published=article.fetched_on
        )

        # Check if the article has an image and add it
        image = session.query(Images).filter(Images.article_id == article.id).first()
        if image:
            news_obj.image_url = image.url

        return news_obj, website

    def select_link(self, tid: int = None):
        """
        Select a link (article) that matches agent's interests by directly querying the database.
        No filtering by political leaning is applied.

        :param tid: optional round id used for decision logging
        :return: The selected article and website objects, or empty strings if none found
        """
        from y_client.news_feeds.feed_reader import News
        from sqlalchemy import or_
        try:
            tid_for_log = int(tid) if tid is not None else None
        except Exception:
            tid_for_log = None

        def _article_to_log(article_obj, source: str):
            if article_obj is None:
                return None
            return {
                "article_id": int(getattr(article_obj, "id", -1))
                if getattr(article_obj, "id", None) is not None
                else None,
                "website_id": int(getattr(article_obj, "website_id", -1))
                if getattr(article_obj, "website_id", None) is not None
                else None,
                "title": self._decision_compact_text(getattr(article_obj, "title", ""), 140),
                "summary_snippet": self._decision_compact_text(
                    getattr(article_obj, "summary", ""), 240
                ),
                "link": self._decision_compact_text(getattr(article_obj, "link", ""), 220),
                "source": source,
            }

        def _log_link_decision(
            selection_mode: str,
            fallback_reason: str = "",
            selected_article=None,
            selected_website=None,
        ):
            selected_for_log = _article_to_log(selected_article, selection_mode) if selected_article else None
            if selected_for_log is not None:
                selected_for_log["website_name"] = self._decision_compact_text(
                    getattr(selected_website, "name", ""), 80
                )
            try:
                self._decision_log(
                    {
                        "decision_type": "link_share_candidate_decision",
                        "run_id": getattr(self, "memory_run_id", None),
                        "agent_user_id": int(getattr(self, "user_id", -1) or -1),
                        "agent_name": getattr(self, "name", None),
                        "tid": tid_for_log,
                        "selection_mode": selection_mode,
                        "fallback_reason": fallback_reason,
                        "interests": interests_for_log,
                        "persona": self._decision_persona_snapshot(),
                        "candidates": candidates_for_log[:6],
                        "selected": selected_for_log,
                    }
                )
            except Exception:
                pass

        candidates_for_log = []
        interests_for_log = []

        # Get the session from global scope (same fix as in add_feeds method)
        try:
            from y_client.clients.client_web import session as global_session
            if global_session is not None:
                current_session = global_session
            else:
                # Fallback to module-level session
                current_session = session
        except ImportError:
            current_session = session
            
        if current_session is None:
            import logging
            logging.error("Database session is None in select_link! Cannot query articles.")
            _log_link_decision("error", "db_session_none")
            return "", ""

        # Get agent's interests
        interests, _ = self.__get_interests(-1)
        interests_for_log = [
            self._decision_compact_text(interest, 48)
            for interest in (interests or [])[:10]
            if interest
        ]

        if not interests:
            # Fallback to select_news if no interests available
            article, website = self.select_news()
            if article and not isinstance(article, str):
                selected = type(
                    "LogArticle",
                    (),
                    {
                        "id": None,
                        "website_id": getattr(website, "id", None),
                        "title": getattr(article, "title", ""),
                        "summary": getattr(article, "summary", ""),
                        "link": getattr(article, "link", ""),
                    },
                )()
                _log_link_decision(
                    selection_mode="fallback_select_news",
                    fallback_reason="no_interests",
                    selected_article=selected,
                    selected_website=website,
                )
            else:
                _log_link_decision("fallback_select_news", "no_interests_no_article")
            return article, website

        # Get all website IDs without filtering by leaning
        website_ids = current_session.query(Websites.id).all()

        if len(website_ids) == 0:
            _log_link_decision("none_available", "no_websites")
            return "", ""

        # Extract IDs from query result
        website_ids = [w[0] for w in website_ids]

        # Build search conditions for each interest
        search_conditions = []
        for interest in interests:
            if interest and len(interest) > 2:  # Avoid very short search terms
                search_term = f"%{interest.lower()}%"
                search_conditions.append(Articles.title.ilike(search_term))
                search_conditions.append(Articles.summary.ilike(search_term))

        # If we have search conditions, query for matching articles
        matching_articles = []
        if search_conditions:
            matching_articles = current_session.query(Articles).filter(
                Articles.website_id.in_(website_ids),
                or_(*search_conditions)
            ).order_by(func.random()).limit(6).all()
            candidates_for_log = [
                _article_to_log(article, "interest_match")
                for article in matching_articles
            ]

        # If no matching articles found, fall back to random selection
        if not matching_articles:
            fallback_articles = current_session.query(Articles).filter(
                Articles.website_id.in_(website_ids)
            ).order_by(func.random()).limit(6).all()
            candidates_for_log = [
                _article_to_log(article, "random_fallback")
                for article in fallback_articles
            ]

            if not fallback_articles:
                _log_link_decision("none_available", "no_articles")
                return "", ""

            random_article = random.choice(fallback_articles)

            website = current_session.query(Websites).filter(Websites.id == random_article.website_id).first()

            if not website:
                _log_link_decision(
                    "random_fallback",
                    "website_not_found",
                    selected_article=random_article,
                )
                return "", ""

            news_obj = News(
                title=random_article.title,
                summary=random_article.summary,
                link=random_article.link,
                published=random_article.fetched_on
            )

            # Get image if available (handle schema differences between YClientReddit and Y_Web)
            try:
                image = current_session.query(Images).filter(Images.article_id == random_article.id).first()
                if image:
                    news_obj.image_url = image.url
            except Exception as e:
                # Handle schema mismatch - Y_Web database doesn't have remote_article_id column
                import logging
                logging.warning(f"Could not query images due to schema mismatch: {e}")
                # Continue without image

            _log_link_decision(
                selection_mode="random_fallback",
                fallback_reason="no_interest_match",
                selected_article=random_article,
                selected_website=website,
            )
            return news_obj, website

        # If we found matching articles, choose one randomly
        selected_article = random.choice(matching_articles)
        website = current_session.query(Websites).filter(Websites.id == selected_article.website_id).first()

        if not website:
            _log_link_decision(
                "interest_match_random",
                "website_not_found",
                selected_article=selected_article,
            )
            return "", ""

        news_obj = News(
            title=selected_article.title,
            summary=selected_article.summary,
            link=selected_article.link,
            published=selected_article.fetched_on
        )

        # Get image if available (handle schema differences between YClientReddit and Y_Web)
        try:
            image = current_session.query(Images).filter(Images.article_id == selected_article.id).first()
            if image:
                news_obj.image_url = image.url
        except Exception as e:
            # Handle schema mismatch - Y_Web database doesn't have remote_article_id column
            import logging
            logging.warning(f"Could not query images due to schema mismatch: {e}")
            # Continue without image

        _log_link_decision(
            selection_mode="interest_match_random",
            selected_article=selected_article,
            selected_website=website,
        )
        return news_obj, website

    def select_image(self, tid):
        """
        Select an image

        :return: the response from the service
        """
        # randomly select an image from database
        image = session.query(Images).order_by(func.random()).first()

        # @Todo: add the case of no news sharing enabled
        if (
            "news" not in self.actions_likelihood
            or self.actions_likelihood["news"] == 0
        ):
            if image is None:
                # where to get the image from??
                return None, None
            else:
                if image.description is not None:
                    return image, None

                else:
                    # annotate the image with a description
                    an = Annotator(config=self.llm_v_config)
                    description = an.annotate(image.url)
                    image.description = description
                    session.commit()

                    return image, None

        # the news module is active: images will be selected among RSS shared articles
        else:
            # no image available, select a news article and extract image from it
            if image is None:
                news, website = self.select_news()

                if news == "":
                    return None, None

                res = self.news(tid=tid, article=news, website=website)
                article_id = int(
                    json.loads(res.__dict__["_content"].decode("utf-8"))["article_id"]
                )

                # get image given article id and set the remote id
                image = (
                    session.query(Images)
                    .filter(Images.article_id == article_id)
                    .first()
                )

                if image is None:
                    return None, None
                else:
                    image.remote_article_id = article_id
                    session.commit()

                    # annotate the image with a description
                    an = Annotator(self.llm_v_config)
                    description = an.annotate(image.url)
                    image.description = description
                    session.commit()

                    return image, article_id

            # images available, check if they have a description
            else:
                # check if the image has a remote article id
                if image.remote_article_id is None:
                    # get local article linked to the image
                    article = (
                        session.query(Articles)
                        .filter(Articles.id == image.article_id)
                        .first()
                    )
                    # get the website linked to the article
                    website = (
                        session.query(Websites)
                        .filter(Websites.id == article.website_id)
                        .first()
                    )

                    # save the website and article on the server
                    st = json.dumps(
                        {
                            "user_id": self.user_id,
                            "tweet": "",
                            "emotions": [],
                            "hashtags": [],
                            "mentions": [],
                            "tid": tid,
                            "title": article.title,
                            "summary": article.summary,
                            "link": article.link,
                            "publisher": website.name,
                            "rss": website.rss,
                            "leaning": website.leaning,
                            "country": website.country,
                            "language": website.language,
                            "category": website.category,
                            "fetched_on": website.last_fetched,
                        }
                    )

                    headers = {"Content-Type": "application/x-www-form-urlencoded"}

                    api_url = f"{self.base_url}/news"
                    res = post(f"{api_url}", headers=headers, data=st)
                    remote_article_id = int(
                        json.loads(res.__dict__["_content"].decode("utf-8"))[
                            "article_id"
                        ]
                    )
                    image.remote_article_id = remote_article_id
                    session.commit()

                if image.description is not None:
                    return image, image.remote_article_id

                else:
                    # annotate the image with a description
                    an = Annotator(config=self.llm_v_config)
                    description = an.annotate(image.url)
                    image.description = description
                    session.commit()

                    return image, image.remote_article_id

    @log_execution_time
    def comment_image(self, image: object, tid: int, article_id: int = None):
        """
        Comment on an image

        :param image:
        :param tid:
        :param article_id:
        :return:
        """
        # obtain the most recent (and frequent) interests of the agent
        interests, _ = self.__get_interests(tid)

        self.topics_opinions = ""

        fresh_config = self._get_llm_config_for_write_action()

        u1 = AssistantAgent(
            name=f"{self.name}",
            llm_config=fresh_config,
            system_message=self.__effify(
                self.prompts["agent_roleplay_comments_share"], interests=interests
            ),
            max_consecutive_auto_reply=1,
        )

        u2 = AssistantAgent(
            name=f"Handler",
            llm_config=fresh_config,
            system_message=self.__effify(self.prompts["handler_instructions"]),
            max_consecutive_auto_reply=1,
        )

        comment_image_prompt = self.__effify(
            self.prompts["handler_comment_image"], descr=image.description
        )
        u2.initiate_chat(
            u1,
            message=comment_image_prompt,
            silent=True,
            max_turns=1,
        )

        emotion_raw = self._extract_emotion_chat_content(u2, u1)
        emotion_eval = self.__clean_emotion(emotion_raw)

        post_text = self._extract_generated_chat_content(
            u2, u1, prompt_hint=comment_image_prompt, skip_emotion_like=True
        )

        # cleaning the post text of some unwanted characters
        post_text = self.__clean_text(post_text)

        # avoid posting empty messages
        if len(post_text) < 3:
            return

        hashtags = self.__extract_components(post_text, c_type="hashtags")

        st = json.dumps(
            {
                "user_id": self.user_id,
                "text": post_text.replace('"', "")
                .replace(f"{self.name}", "")
                .replace(":", "")
                .replace("*", ""),
                "emotions": emotion_eval,
                "hashtags": hashtags,
                "tid": tid,
                "image_url": image.url,
                "image_description": image.description,
                "article_id": article_id,
            }
        )

        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        api_url = f"{self.base_url}/comment_image"
        post(f"{api_url}", headers=headers, data=st)
        self._record_writing_action()

    def __str__(self):
        """
        Return a string representation of the Agent object.

        :return: the string representation
        """
        return f"Name: {self.name}, Age: {self.age}, Type: {self.type}"

    def __dict__(self):
        """
        Return a dictionary representation of the Agent object.

        :return: the dictionary representation
        """

        interests = self.__get_interests(-1)

        return {
            "name": self.name,
            "email": self.email,
            "password": self.pwd,
            "age": self.age,
            "type": self.type,
            "leaning": self.leaning,
            "interests": interests,
            "oe": self.oe,
            "co": self.co,
            "ex": self.ex,
            "ag": self.ag,
            "ne": self.ne,
            "rec_sys": self.content_rec_sys_name,
            "frec_sys": self.follow_rec_sys_name,
            "language": self.language,
            "owner": self.owner,
            "education_level": self.education_level,
            "round_actions": self.round_actions,
            "gender": self.gender,
            "nationality": self.nationality,
            "toxicity": self.toxicity,
            "joined_on": self.joined_on,
            "is_page": self.is_page,
        }

    def _is_prompt_scaffold(self, text_value):
        text = str(text_value or "").strip()
        if not text:
            return False
        normalized = re.sub(r"\s+", " ", text).strip().lower()
        if not normalized:
            return False
        for pattern in _PROMPT_SCAFFOLD_PATTERNS:
            try:
                if pattern.search(normalized):
                    return True
            except Exception:
                continue
        return False

    def _strip_prompt_scaffold(self, text_value):
        if not isinstance(text_value, str):
            return ""

        lines = []
        for raw_line in text_value.splitlines():
            line = raw_line.strip()
            if not line:
                if lines and lines[-1] != "":
                    lines.append("")
                continue
            if self._is_prompt_scaffold(line):
                continue
            if line.lower().startswith("previous bad attempt:"):
                continue
            lines.append(raw_line)

        cleaned = "\n".join(lines).strip()
        if cleaned and self._is_prompt_scaffold(cleaned):
            return ""
        return cleaned

    def _looks_like_emotion_payload(self, text_value):
        text = str(text_value or "").strip()
        if not text:
            return False

        tokens = [t for t in re.split(r"[\s,\[\]\(\)\{\}:;]+", text.lower()) if t]
        if not tokens or len(tokens) > 8:
            return False
        allowed = {
            str(e).strip().lower()
            for e in (self.emotions or [])
            if str(e).strip()
        }
        if not allowed:
            return False
        return all(t in allowed for t in tokens)

    def _extract_chat_messages(self, chat_owner, peer_agent):
        if chat_owner is None or peer_agent is None:
            return []
        try:
            messages = chat_owner.chat_messages.get(peer_agent, [])
            if isinstance(messages, list):
                return messages
        except Exception:
            pass
        try:
            messages = chat_owner.chat_messages[peer_agent]
            if isinstance(messages, list):
                return messages
        except Exception:
            pass
        return []

    def _extract_emotion_chat_content(self, chat_owner, peer_agent):
        messages = self._extract_chat_messages(chat_owner, peer_agent)
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            if self.__clean_emotion(content):
                return content.lower()

        try:
            last = chat_owner.last_message(peer_agent)
            if isinstance(last, dict):
                content = last.get("content")
                if isinstance(content, str):
                    return content.lower()
        except Exception:
            pass
        return ""

    def _extract_generated_chat_content(
        self,
        chat_owner,
        peer_agent,
        *,
        prompt_hint=None,
        skip_emotion_like=False,
    ):
        prompt_norm = re.sub(r"\s+", " ", str(prompt_hint or "").strip()).lower()
        messages = self._extract_chat_messages(chat_owner, peer_agent)

        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, str) or not content.strip():
                continue

            norm = re.sub(r"\s+", " ", content.strip()).lower()
            if prompt_norm and norm == prompt_norm:
                continue

            cleaned = self._strip_prompt_scaffold(content)
            if not cleaned:
                continue
            if skip_emotion_like and self._looks_like_emotion_payload(cleaned):
                continue
            return cleaned

        try:
            last = chat_owner.last_message(peer_agent)
            if isinstance(last, dict):
                content = last.get("content")
                if isinstance(content, str):
                    cleaned = self._strip_prompt_scaffold(content)
                    if cleaned and not (
                        skip_emotion_like and self._looks_like_emotion_payload(cleaned)
                    ):
                        return cleaned
        except Exception:
            pass
        return ""

    def __clean_emotion(self, text):
        try:
            emotion_eval = [
                e.strip()
                for e in text.replace("'", " ")
                .replace('"', " ")
                .replace("*", "")
                .replace(":", " ")
                .replace("[", " ")
                .replace("]", " ")
                .replace(",", " ")
                .split(" ")
                if e.strip() in self.emotions
            ]
        except:
            emotion_eval = []
        return emotion_eval

    def __clean_text(self, text):
        if not isinstance(text, str):
            return ""

        text = (
            text.split("##")[-1]
            .replace("@ ", "")
            .replace("  ", " ")
            .replace(" ,", ",")
            .replace("[", "")
            .replace("]", "")
            .replace("@,", "")
            .strip("()[]{}'")
            .lstrip()
        )
        text = self._strip_prompt_scaffold(text)
        text = text.replace(f"@{self.name}", "")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text


class Agents(object):
    def __init__(self):
        """
        Initialize the Agent object.
        """
        self.agents = []

    def add_agent(self, agent: Agent):
        """
        Add a profile to the Agents object.

        :param agent: The Profile object to add.
        """
        self.agents.append(agent)

    def remove_agent(self, agent: Agent):
        """
        Remove a profile from the Agents object.

        :param agent: The Profile object to remove.
        """
        self.agents.remove(agent)

    def remove_agent_by_ids(self, agent_ids: list):
        """
        Remove a profile from the Agents object.

        :param agent: The Profile object to remove.
        """
        for agent in self.agents:
            if agent.user_id in agent_ids:
                self.agents.remove(agent)

    def get_agents(self):
        return self.agents

    def agents_iter(self):
        """
        Iterate over the agents.
        """
        for agent in self.agents:
            yield agent

    def __str__(self):
        """
        Return a string representation of the Agents object.

        :return: the string representation
        """
        return "".join([p.__str__() for p in self.agents])

    def __dict__(self):
        """
        Return a dictionary representation of the Agents object.

        :return: the dictionary representation
        """
        return {"agents": [p.__dict__() for p in self.agents]}

    def __eq__(self, other):
        """
        Return True if the Agents objects are equal.

        :param other: The other agent object to compare.
        :return: True if the Agents objects are equal.
        """
        return self.__dict__() == other.__dict__()
