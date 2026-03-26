import json
import sys
import os
import shutil
import logging
import traceback
import time
import random
from pathlib import Path
from sqlalchemy.ext.declarative import declarative_base
import sqlalchemy as db
from requests import post
from sqlalchemy import orm



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
session = None
engine = None
base = None

from y_client.classes.base_agent import Agent, Agents
from y_client.classes.fake_base_agent import FakeAgent
from y_client.classes.time import SimulationSlot
from y_client.news_feeds import Feeds
from y_client.news_feeds.client_modals import ImagePosts
from y_client.clients.logging_utils import resolve_log_file_path


class YClientWeb(object):
    def __init__(
        self,
        config_file,
        data_base_path,
        agents_filename=None,
        agents_output="agents.json",
        owner="admin",
        first_run=False,
        network=None,
        log_file="agent_execution.log",
        llm=True,
    ):
        """
        Initialize the YClient object

        :param config_filename: the configuration file for the simulation in JSON format
        :param prompts_filename: the LLM prompts file for the simulation in JSON format
        :param agents_filename: the file containing the agents in JSON format
        :param graph_file: the file containing the graph of the agents in CSV format, where the number of nodes is equal to the number of agents
        :param agents_output: the file to save the generated agents in JSON format
        :param owner: the owner of the simulation
        :param first_run: if it is the first run of the simulation
        :param log_file: path to the log file for agent execution time tracking
        :param llm: whether to use LLM for agent behaviors
        """
        from y_client.logger import set_logger

        # Configure the logger with the specified log file
        resolved_log_file = resolve_log_file_path(data_base_path, log_file)
        set_logger(resolved_log_file)

        self.first_run = first_run
        self.llm = llm
        self.base_path = data_base_path
        self.log_file = resolved_log_file
        self.config = config_file

        self.prompts = self._load_prompts_with_defaults(data_base_path)

        self.agents_owner = owner
        self.agents_filename = agents_filename
        self.agents_output = agents_output

        self.days = int(self.config["simulation"]["days"])
        self.slots = int(self.config["simulation"]["slots"])
        self.percentage_new_agents_iteration = float(self.config["simulation"][
            "percentage_new_agents_iteration"
        ])
        self.hourly_activity = self.config["simulation"]["hourly_activity"]
        self.percentage_removed_agents_iteration = float(
            self.config["simulation"]["percentage_removed_agents_iteration"]
        )
        self.actions_likelihood = {
            a.upper(): float(v)
            for a, v in self.config["simulation"]["actions_likelihood"].items()
        }
        # Forum experiments do not allow repost/share of other users' content.
        if "SHARE" in self.actions_likelihood:
            self.actions_likelihood["SHARE"] = 0.0
        tot = sum(self.actions_likelihood.values())
        if tot <= 0:
            self.actions_likelihood = {"NONE": 1.0}
        else:
            self.actions_likelihood = {
                k: v / tot for k, v in self.actions_likelihood.items()
            }

        # users' parameters
        self.fratio = float(self.config["agents"]["reading_from_follower_ratio"])
        self.max_length_thread_reading = int(self.config["agents"][
            "max_length_thread_reading"
        ])

        # posts' parameters
        self.visibility_rd = int(self.config["posts"]["visibility_rounds"])

        ##############
        BASE_DIR = os.path.dirname(os.path.abspath(__file__)).split("y_client")[0]

        # Check for PostgreSQL via environment variable or data_base_path
        database_url = os.environ.get("DATABASE_URL")

        if database_url and "postgresql" in database_url:
            # PostgreSQL mode
            db_uri = database_url
            logging.info(f"Using PostgreSQL database: {db_uri}")
        elif data_base_path and os.path.exists(f"{data_base_path}database_server.db"):
            # Use the Y_Web experiment database (SQLite)
            db_path = f"{data_base_path}database_server.db"
            db_uri = f"sqlite:////{db_path}"
            logging.info(f"Using Y_Web experiment database: {db_path}")
        else:
            # Fallback to standalone database (for backwards compatibility)
            db_path = f"{BASE_DIR}experiments/{self.config['simulation']['name']}.db"
            if not os.path.exists(db_path):
                # copy the clean database to the experiments folder
                shutil.copyfile(
                    f"{BASE_DIR}data_schema/database_clean_client.db",
                    db_path,
                )
            db_uri = f"sqlite:////{db_path}"
            logging.info(f"Using standalone database: {db_path}")

        global session, engine, base
        base = declarative_base()

        # Create engine with appropriate options
        if "postgresql" in db_uri:
            from sqlalchemy.pool import NullPool
            engine = db.create_engine(db_uri, poolclass=NullPool)
        else:
            engine = db.create_engine(db_uri, connect_args={"check_same_thread": False, "timeout": 30})
        base.metadata.bind = engine
        session = orm.scoped_session(orm.sessionmaker())(bind=engine)

        globals()["session"] = session
        globals()["engine"] = engine
        globals()["base"] = base
        ##############

        # Ensure images are populated for all experiments
        self._ensure_images_populated(data_base_path, engine)

        yclient_path = os.path.dirname(os.path.abspath(__file__)).split("y_web")[0]
        sys.path.append(f'{yclient_path}{os.sep}external{os.sep}YClient/')

        # initialize simulation clock
        self.sim_clock = SimulationSlot(self.config)

        self.agents = Agents()
        self.feed = Feeds()
        self.content_recsys = None
        self.follow_recsys = None
        self.network = network
        self.pages = []  # Reddit doesn't have page agents, all are regular users

    def _rule_based_agents_enabled(self):
        llm_agents = self.config.get("agents", {}).get("llm_agents")
        return (
            isinstance(llm_agents, list)
            and len(llm_agents) == 1
            and llm_agents[0] is None
        )

    def _agent_class_for_payload(self, payload):
        if self._rule_based_agents_enabled():
            return FakeAgent
        # Backward compatibility for forum rule-based clients created while the
        # writer incorrectly persisted llm_agents=[] and empty per-agent types.
        if not str((payload or {}).get("type") or "").strip():
            return FakeAgent
        return Agent

    def _coerce_legacy_rule_based_config(self, payloads):
        if self._rule_based_agents_enabled():
            return
        non_page_payloads = [
            payload for payload in (payloads or []) if payload.get("is_page", 0) != 1
        ]
        if not non_page_payloads:
            return
        if all(
            not str(payload.get("type") or "").strip()
            for payload in non_page_payloads
        ):
            self.config.setdefault("agents", {})["llm_agents"] = [None]

    def _load_prompts_with_defaults(self, data_base_path: str):
        exp_prompts_path = Path(data_base_path) / "prompts.json"
        with open(exp_prompts_path, "r") as f:
            exp_prompts = json.load(f)
        if not isinstance(exp_prompts, dict):
            exp_prompts = {}

        default_prompts = {}
        candidate_defaults = [
            # Package-local fallback defaults.
            Path(__file__).resolve().parents[2] / "config_files" / "prompts.json",
            # YSocial forum defaults used when creating new forum clients.
            # These should override package-local defaults for forum mode.
            Path(__file__).resolve().parents[4] / "data_schema" / "prompts_forum.json",
        ]

        for candidate in candidate_defaults:
            try:
                if not candidate.exists():
                    continue
                with open(candidate, "r") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    default_prompts.update(loaded)
            except Exception as exc:
                logging.warning(f"Could not load default prompts from {candidate}: {exc}")

        if not default_prompts:
            return exp_prompts

        merged = dict(default_prompts)
        merged.update(exp_prompts)
        missing = [k for k in default_prompts.keys() if k not in exp_prompts]
        if missing:
            logging.info(
                "Merged %d missing default prompts into experiment prompts at %s",
                len(missing),
                exp_prompts_path,
            )
        return merged

    def _ensure_images_populated(self, data_base_path, engine):
        """
        Ensure image_posts table is populated for PostgreSQL experiments.

        Checks if the image_posts table is empty. If empty and image_feeds.json
        exists in the experiment folder, populates the table from Reddit RSS feeds.
        This runs automatically on client startup for PostgreSQL experiments.
        """
        import time
        import re
        import json
        import datetime
        from sqlalchemy import text
        from y_client.news_feeds.feed_reader import parse_feed_with_retry

        try:
            ImagePosts.__table__.create(bind=engine, checkfirst=True)
        except Exception as exc:
            logging.warning("Could not ensure image_posts table exists: %s", exc)

        IMAGE_EXTENSIONS = r"\.(jpg|jpeg|png|gif|webp)(\?.*)?$"
        def is_nsfw(entry) -> bool:
            """Check if RSS entry is NSFW."""
            if hasattr(entry, "over_18") and entry.over_18:
                return True
            if hasattr(entry, "tags"):
                for tag in entry.tags:
                    if hasattr(tag, "term") and tag.term.lower() in ["nsfw", "over_18"]:
                        return True
            title = entry.title.lower() if hasattr(entry, "title") else ""
            if "[nsfw]" in title or "(nsfw)" in title:
                return True
            return False

        def extract_image_url(entry):
            """Extract image URL from RSS entry, preserving signed params for Reddit preview URLs."""
            candidates = []

            if hasattr(entry, "link"):
                if re.search(IMAGE_EXTENSIONS, entry.link, re.IGNORECASE):
                    url = entry.link
                    if "preview.redd.it" not in url and "external-preview" not in url:
                        url = url.split("?")[0]
                    candidates.append(url)

            if hasattr(entry, "media_content"):
                for media in entry.media_content:
                    if "url" in media and re.search(IMAGE_EXTENSIONS, media["url"], re.IGNORECASE):
                        url = media["url"]
                        if "preview.redd.it" not in url and "external-preview" not in url:
                            url = url.split("?")[0]
                        candidates.append(url)

            if hasattr(entry, "media_thumbnail"):
                for thumb in entry.media_thumbnail:
                    if "url" in thumb:
                        candidates.append(thumb["url"])

            if hasattr(entry, "summary"):
                img_match = re.search(r'<img[^>]+src="([^"]+)"', entry.summary)
                if img_match:
                    url = img_match.group(1).replace("&amp;", "&")
                    candidates.append(url)

            if not candidates:
                return None

            # Avoid tiny thumbnails (thumbs.redditmedia.com) when possible.
            non_thumb_candidates = [
                u for u in candidates if "thumbs.redditmedia.com" not in u
            ] or candidates

            # Prefer direct/fullsize domains first.
            preferred_domains = ["i.redd.it", "imgur.com", "i.imgur.com"]
            for url in non_thumb_candidates:
                if any(domain in url for domain in preferred_domains):
                    return url

            for url in non_thumb_candidates:
                if "preview.redd.it" in url and "?" in url:
                    return url

            # If all we found are thumbnails, skip this entry entirely.
            if all("thumbs.redditmedia.com" in u for u in candidates):
                return None

            return non_thumb_candidates[0]

        try:
            # Check if image_posts table is empty
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM image_posts"))
                count = result.scalar()

            if count > 0:
                logging.info(f"image_posts table already has {count} images, skipping population")
                return

            # Check for image_feeds.json
            image_feeds_path = os.path.join(data_base_path, "image_feeds.json")
            if not os.path.exists(image_feeds_path):
                logging.info("No image_feeds.json found, skipping image population")
                return

            logging.info(f"image_posts table is empty, populating from {image_feeds_path}")

            # Import download functions
            from y_client.news_feeds.image_feed_reader import (
                download_image_checked, extract_high_res_url, generate_filename,
                IMAGE_STORAGE_DIR
            )

            with open(image_feeds_path) as f:
                feeds = json.load(f)

            timestamp = datetime.datetime.now().strftime("%Y%m%d")
            total_added = 0
            total_downloaded = 0

            # Setup storage directory (in y_web/static)
            storage_path = Path(data_base_path).parent.parent / "static" / IMAGE_STORAGE_DIR
            storage_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Image storage directory: {storage_path}")

            for i, feed in enumerate(feeds):
                subreddit = feed.get("subreddit")
                if not subreddit:
                    continue

                # Rate limiting for Reddit RSS feeds (~5-7 seconds between requests)
                if i > 0:  # Don't delay before first feed
                    delay = 5.0 + random.uniform(0, 2)  # 5-7 second jitter
                    logging.info(f"Rate limiting: waiting {delay:.1f}s before next image feed...")
                    time.sleep(delay)

                logging.info(f"Fetching images from r/{subreddit}...")

                try:
                    feed_url = f"https://www.reddit.com/r/{subreddit}.rss"
                    rss_feed, used_url, error = parse_feed_with_retry(
                        feed_url,
                        timeout=20,
                        max_retries=3,
                        backoff_seconds=2.0,
                        require_entries=True,
                    )
                    if not rss_feed:
                        logging.warning(f"Failed to fetch r/{subreddit}: {error}")
                        continue
                    if used_url and used_url != feed_url:
                        logging.info(f"Using fallback feed URL: {used_url}")
                    if hasattr(rss_feed, "bozo") and rss_feed.bozo:
                        logging.warning(
                            f"Warning: Feed might be malformed: {rss_feed.bozo_exception}"
                        )

                    added = 0
                    downloaded = 0
                    for entry in rss_feed.entries[:25]:
                        if is_nsfw(entry):
                            continue

                        image_url = extract_image_url(entry)
                        if not image_url:
                            continue

                        # Check for duplicates
                        with engine.connect() as conn:
                            result = conn.execute(
                                text("SELECT id FROM image_posts WHERE url = :url"),
                                {"url": image_url}
                            )
                            if result.fetchone():
                                continue

                        # Get high-res URL (with rate limiting delay)
                        source_url = entry.link if hasattr(entry, "link") else None
                        high_res_url = None
                        if source_url:
                            high_res_url = extract_high_res_url(source_url, image_url)
                            # Small delay between Reddit API requests (1-2s)
                            time.sleep(1.0 + random.uniform(0, 1))

                        # Insert record to get ID
                        with engine.begin() as conn:
                            result = conn.execute(
                                text("""
                                    INSERT INTO image_posts (url, source_url, title, subreddit, fetched_on, used, high_res_url)
                                    VALUES (:url, :source_url, :title, :subreddit, :fetched_on, false, :high_res_url)
                                    RETURNING id
                                """),
                                {
                                    "url": image_url,
                                    "source_url": source_url,
                                    "title": entry.title[:300] if hasattr(entry, "title") else "",
                                    "subreddit": subreddit,
                                    "fetched_on": timestamp,
                                    "high_res_url": high_res_url,
                                }
                            )
                            row = result.fetchone()
                            image_id = row[0] if row else None

                        if image_id:
                            added += 1

                            # Download image locally
                            download_url = high_res_url or image_url
                            filename = generate_filename(download_url, image_id)
                            filepath = storage_path / filename
                            relative_path = f"{IMAGE_STORAGE_DIR}/{filename}"

                            logging.info(f"  Downloading: {download_url[:60]}...")
                            ok, reason = download_image_checked(download_url, str(filepath))
                            if ok:
                                with engine.begin() as conn:
                                    conn.execute(
                                        text("UPDATE image_posts SET local_path = :path WHERE id = :id"),
                                        {"path": relative_path, "id": image_id}
                                    )
                                downloaded += 1
                                logging.info(f"    -> Saved to {relative_path}")
                            else:
                                logging.warning(f"    -> Download failed ({reason})")
                                # If we only got a tiny thumbnail, delete the row so it won't be selected for posts.
                                if reason == "too_small":
                                    with engine.begin() as conn:
                                        conn.execute(
                                            text("DELETE FROM image_posts WHERE id = :id"),
                                            {"id": image_id},
                                        )
                                    try:
                                        if filepath.exists():
                                            filepath.unlink()
                                    except Exception:
                                        pass

                    total_added += added
                    total_downloaded += downloaded
                    logging.info(f"  Added {added} images from r/{subreddit}, downloaded {downloaded}")

                except Exception as e:
                    logging.warning(f"Error fetching r/{subreddit}: {e}")
                    continue

            logging.info(f"Image population complete: {total_added} images added, {total_downloaded} downloaded")

            # Annotate newly populated images with VLM descriptions
            if total_added > 0:
                from y_client.news_feeds.image_feed_reader import annotate_pending_images
                from y_client.classes.annotator import Annotator

                logging.info("Starting image annotation with VLM...")
                try:
                    llm_v_url = os.getenv("LLM_URL", "http://127.0.0.1:11434/v1")
                    annotator = Annotator(config={
                        "url": llm_v_url,
                        "api_key": "NULL",
                        "model": "minicpm-v",
                        "temperature": 0.5,
                        "max_tokens": 300,
                    })
                    annotated = annotate_pending_images(annotator, batch_size=50, engine=engine)
                    logging.info(f"Annotated {annotated} images with VLM descriptions")
                except Exception as e:
                    logging.error(f"Image annotation failed: {e}")

        except Exception as e:
            logging.warning(f"Error during image population: {e}")
            # Don't crash the client if image population fails

    def read_agents(self):
        """
        Read the agents from the file

        :return:
        """
        import y_client.recsys as recsys
        import y_client.recsys as frecsys

        # population filename
        population_name = str(self.config["simulation"]["population"])
        population_candidates = [
            Path(self.base_path) / f"{population_name}.json",
            Path(self.base_path) / f"{population_name.replace(' ', '')}.json",
        ]
        chosen_population = next((path for path in population_candidates if path.exists()), None)
        if chosen_population is None:
            chosen_population = population_candidates[1]
        self.agents_filename = str(chosen_population)
        data = json.load(open(self.agents_filename, "r"))
        self._coerce_legacy_rule_based_config(data.get("agents", []))
        skipped_pages = 0
        for ag in data["agents"]:
            is_page = ag.get("is_page", 0)
            if is_page != 1:
                AgentClass = self._agent_class_for_payload(ag)
                content_recsys = getattr(recsys, ag["rec_sys"])()
                follow_recsys = getattr(frecsys, ag["frec_sys"])(leaning_bias=1.5)
                agent = AgentClass(
                    name=ag["name"],
                    email=ag["email"],
                    pwd=ag["password"],
                    ag_type=ag["type"],
                    leaning=ag["leaning"],
                    interests=ag["interests"][0],
                    oe=ag["oe"],
                    co=ag["co"],
                    ex=ag["ex"],
                    ag=ag["ag"],
                    ne=ag["ne"],
                    education_level=ag.get("education_level", ""),
                    round_actions=ag.get("round_actions", []),
                    nationality=ag.get("nationality", ""),
                    profession=ag.get("profession", ""),
                    toxicity=ag.get("toxicity", 0),
                    gender=ag.get("gender", ""),
                    age=ag.get("age", 0),
                    recsys=content_recsys,
                    frecsys=follow_recsys,
                    language=ag.get("language", "en"),
                    owner=ag.get("owner", ""),
                    config=self.config,
                    load=not self.first_run,
                    web=True,
                    prompt=ag.get("prompts", {}),
                    daily_activity_level=ag.get("daily_activity_level", 1),
                    activity_profile=ag.get("activity_profile", "Always On"),
                    opinions=ag.get("opinions"),
                    experiment_db_path=os.path.join(self.base_path, "database_server.db"),
                )
                agent.set_prompts(self.prompts)
                self.agents.add_agent(agent)
            else:
                skipped_pages += 1
        if skipped_pages:
            logging.info(
                "Skipped %s legacy page agent definitions; standard agents now handle link sharing.",
                skipped_pages,
            )

    def add_feeds(self):
        """
        Load and process RSS feeds if configured.
        Bridge method for Y_Web compatibility.
        """
        import logging
        
        # Ensure session is properly set in the global scope for feed_reader
        global session, engine, base
        if session is not None:
            # Import feed_reader modules and update their session reference
            try:
                import y_client.news_feeds.feed_reader as feed_reader_module
                import y_client.news_feeds.client_modals as client_modals_module
                
                # Explicitly set the session and related objects in the modules
                feed_reader_module.session = session
                client_modals_module.session = session
                client_modals_module.engine = engine  
                client_modals_module.base = base
                
                logging.info("Database session properly initialized for RSS processing")
            except ImportError as e:
                logging.warning(f"Could not update session in feed_reader modules: {e}")
        else:
            logging.error("Database session is None! RSS feeds cannot be processed.")
        
        # Check if RSS feeds are configured in the experiment
        rss_file = os.path.join(self.base_path, "rss_feeds.json")
        if not os.path.exists(rss_file):
            logging.info("No RSS feeds configured for this experiment (rss_feeds.json not found)")
            return
        logging.info(f"Loading RSS feeds from {rss_file}")
        try:
            self.load_feeds(rss_file)
        except Exception as e:
            logging.error(f"Failed to load RSS feeds: {e}")
            import traceback
            traceback.print_exc()
            # Don't crash the simulation if RSS feeds fail to load
    def set_interests(self):
        """
        Set the interests of the agents
        """
        api_url = f"{self.config['servers']['api']}set_interests"

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        data = self.config["agents"]["interests"]

        post(f"{api_url}", headers=headers, data=json.dumps(data))

    def set_recsys(self, c_recsys, f_recsys):
        """
        Set the recommendation systems

        :param c_recsys: the content recommendation system
        :param f_recsys: the follower recommendation system
        """
        self.content_recsys = c_recsys
        self.follow_recsys = f_recsys

    def save_agents(self, agent_file):
        """
        Save the agents to a file
        """
        res = self.agents.__dict__()

        json.dump(res, open(agent_file, "w"), indent=4)

    def load_existing_agents(self, a_file):
        """
        Load existing agents from a file
        :param a_file: the JSON file containing the agents
        """
        agents = json.load(open(a_file, "r"))
        self._coerce_legacy_rule_based_config(agents.get("agents", []))
        skipped_pages = 0
        for a in agents["agents"]:
            try:
                if a.get("is_page", 0) != 1:
                    AgentClass = self._agent_class_for_payload(a)
                    ag = AgentClass(
                        name=a["name"],
                        email=a["email"],
                        load=True,
                        config=self.config,
                        web=True,
                        opinions=a.get("opinions"),
                        experiment_db_path=os.path.join(self.base_path, "database_server.db"),
                    )
                    ag.set_prompts(self.prompts)
                    ag.set_rec_sys(self.content_recsys, self.follow_recsys)
                    self.agents.add_agent(ag)
                else:
                    skipped_pages += 1
            except Exception:
                logging.exception(f"Error loading agent: {a['name']}")
        if skipped_pages:
            logging.info(
                "Skipped %s legacy page agents from existing roster; link sharing is handled by regular users.",
                skipped_pages,
            )

    def add_network(self):
        """
        Add network relationships between agents.
        Not implemented for Reddit-style simulations - follower relationships
        are handled differently in forum-style platforms.
        """
        logging.info("add_network called but not implemented for Reddit-style simulations")
        pass

    def churn(self, tid):
        """
        Evaluate churn

        :param tid:
        :return:
        """

        if self.percentage_removed_agents_iteration > 0:
            n_users = max(
                1,
                int(len(self.agents.agents) * self.percentage_removed_agents_iteration),
            )
            st = json.dumps({"n_users": n_users, "left_on": tid})

            headers = {"Content-Type": "application/x-www-form-urlencoded"}

            api_url = f"{self.config['servers']['api']}/churn"
            response = post(f"{api_url}", headers=headers, data=st)

            data = json.loads(response.__dict__["_content"].decode("utf-8"))["removed"]

            self.agents.remove_agent_by_ids(data)

    def add_agent(self, agent=None, username_type=None):
        """
        Add an agent to the simulation

        :param agent: the agent to add
        """
        from y_client.utils import generate_user

        if agent is None:
            max_attempts = 8
            used_names = {a.name for a in self.agents.agents if getattr(a, "name", None)}
            for attempt in range(max_attempts):
                try:
                    agent = generate_user(
                        self.config,
                        owner=self.agents_owner,
                        username_type=username_type,
                        used_names=used_names,
                    )
                    if agent is None:
                        continue
                    agent.set_prompts(self.prompts)
                    agent.set_rec_sys(self.content_recsys, self.follow_recsys)
                    break
                except Exception as e:
                    logging.error(
                        "Error generating agent (attempt %s/%s): %s",
                        attempt + 1,
                        max_attempts,
                        e,
                        exc_info=True,
                    )
                    traceback.print_exc()
                    agent = None
            if agent is None:
                logging.warning(
                    "Failed to generate a new agent after %s attempts",
                    max_attempts,
                )
                return
        if agent is not None:
            self.agents.add_agent(agent)

    def load_feeds(self, filename):
        """
        Load RSS feeds from a file, add them to self.feed, and process them.
        Adds optional fields: language (default 'en'), country (default '').
        Logs statistics about feeds and articles.
        :param filename: the file containing the RSS feeds (JSON)
        :return: True if feeds were loaded successfully, False otherwise
        """
        import logging
        import traceback
        try:
            logging.info(f"Loading RSS feed definitions from {filename}...")
            with open(filename, 'r') as f:
                data = json.load(f)
            logging.info(f"Found {len(data)} feed definitions in file")

            for fdef in data:
                self.feed.add_feed(
                    name=fdef.get("name", ""),
                    url_feed=fdef.get("feed_url", ""),
                    category=fdef.get("category", ""),
                    leaning=fdef.get("leaning", ""),
                    language=fdef.get("language", "en"),
                    country=fdef.get("country", ""),
                    fetch_images_from_url=fdef.get("fetch_images_from_url", False),
                    fetch_images_timeout=fdef.get("fetch_images_timeout", 10),
                )

            # Process all feeds manually
            total_stats = {
                "total_feeds": len(self.feed.feeds),
                "successful_feeds": 0,
                "total_articles": 0,
                "feeds_with_articles": 0
            }

            logging.info("====== RSS Feed Processing ======")
            logging.info(f"Processing {len(self.feed.feeds)} feeds...")

            # Rate limiting for Reddit RSS feeds (~5-7 seconds between requests)
            for i, feed in enumerate(self.feed.feeds):
                if i > 0:  # Don't delay before first feed
                    delay = 5.0 + random.uniform(0, 2)  # 5-7 second jitter
                    logging.info(f"Rate limiting: waiting {delay:.1f}s before next feed...")
                    time.sleep(delay)

                initial_article_count = len(feed.news)
                feed.read_feed()
                articles_found = len(feed.news) - initial_article_count
                if articles_found > 0:
                    total_stats["feeds_with_articles"] += 1
                    total_stats["total_articles"] += articles_found
                    total_stats["successful_feeds"] += 1

            logging.info("====== RSS Feed Processing Summary ======")
            logging.info(f"Total feeds processed: {total_stats['total_feeds']}")
            logging.info(f"Successful feeds: {total_stats['successful_feeds']}")
            logging.info(f"Feeds with articles: {total_stats['feeds_with_articles']}")
            logging.info(f"Total articles collected: {total_stats['total_articles']}")

            if total_stats['total_articles'] == 0:
                logging.warning("No articles were found in any feeds!")
                return False
            return True
        except Exception as e:
            logging.error(f"Error loading RSS feeds: {str(e)}")
            traceback.print_exc()
            return False

    def load_news_from_urls(self, urls_file, max_urls=1000):
        """
        Load news articles from a list of URLs in a text file.
        Uses URLReader to process URLs and logs statistics.
        :param urls_file: Path to the file containing URLs (one per line)
        :param max_urls: Maximum number of URLs to process (randomly sampled)
        :return: True if articles were loaded successfully, False otherwise
        """
        import logging
        import traceback
        import random
        try:
            logging.info(f"Loading URLs from file: {urls_file}")
            with open(urls_file, "r") as f:
                urls = [line.strip() for line in f if line.strip()]
            if not urls:
                logging.warning("No URLs found in the file.")
                return False
            if len(urls) > max_urls:
                logging.info(f"Sampling {max_urls} URLs from {len(urls)} available...")
                urls = random.sample(urls, min(max_urls, len(urls)))
            from y_client.news_feeds.url_reader import URLReader
            url_reader = URLReader(urls)
            stats = url_reader.process_urls()
            if stats.get('processed', 0) == 0:
                logging.warning("No articles were successfully processed from URLs!")
                return False
            logging.info(f"Articles processed from URLs: {stats['processed']}")
            return True
        except Exception as e:
            logging.error(f"Error loading news from URLs: {str(e)}")
            traceback.print_exc()
            return False
