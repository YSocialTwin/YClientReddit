import json
import sys
import os
import shutil
import logging
import traceback
import datetime
from sqlalchemy.ext.declarative import declarative_base
import sqlalchemy as db
from requests import post
from sqlalchemy import orm



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
session = None
engine = None
base = None

from y_client.classes import Agent, Agents, SimulationSlot
from y_client.news_feeds import Feeds
from y_client.news_feeds.client_modals import ImagePosts


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
        set_logger(log_file)

        self.first_run = first_run
        self.llm = llm
        self.base_path = data_base_path
        self.config = config_file

        self.prompts = json.load(open(f"{data_base_path}prompts.json", "r"))

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
        tot = sum(self.actions_likelihood.values())
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

        database_url = os.environ.get("DATABASE_URL")
        if database_url and "postgresql" in database_url:
            db_uri = database_url
            logging.info(f"Using PostgreSQL database: {db_uri}")
        elif data_base_path and os.path.exists(f"{data_base_path}database_server.db"):
            db_path = f"{data_base_path}database_server.db"
            db_uri = f"sqlite:////{db_path}"
            logging.info(f"Using Y_Web experiment database: {db_path}")
        else:
            db_path = f"{BASE_DIR}experiments/{self.config['simulation']['name']}.db"
            if not os.path.exists(db_path):
                shutil.copyfile(
                    f"{BASE_DIR}data_schema/database_clean_client.db",
                    db_path,
                )
            db_uri = f"sqlite:////{db_path}"
            logging.info(f"Using standalone database: {db_path}")

        global session, engine, base
        base = declarative_base()

        if "postgresql" in db_uri:
            from sqlalchemy.pool import NullPool

            engine = db.create_engine(db_uri, poolclass=NullPool)
        else:
            engine = db.create_engine(
                db_uri,
                connect_args={"check_same_thread": False, "timeout": 30},
            )
        base.metadata.bind = engine
        session = orm.scoped_session(orm.sessionmaker())(bind=engine)

        globals()["session"] = session
        globals()["engine"] = engine
        globals()["base"] = base
        ##############

        self._ensure_image_posts_populated(data_base_path)

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

    def _ensure_image_posts_populated(self, data_base_path):
        """Populate the standalone image pool from image_feeds.json when present."""
        import feedparser
        import re
        from sqlalchemy import text

        image_feeds_path = os.path.join(data_base_path, "image_feeds.json")
        if not os.path.exists(image_feeds_path):
            return

        create_table_sql = """
            CREATE TABLE IF NOT EXISTS image_posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url VARCHAR(500) NOT NULL,
                source_url VARCHAR(500),
                title VARCHAR(300),
                subreddit VARCHAR(100),
                description TEXT,
                fetched_on VARCHAR(20),
                used BOOLEAN DEFAULT 0
            )
        """
        if engine.dialect.name == "postgresql":
            create_table_sql = """
                CREATE TABLE IF NOT EXISTS image_posts (
                    id SERIAL PRIMARY KEY,
                    url VARCHAR(500) NOT NULL,
                    source_url VARCHAR(500),
                    title VARCHAR(300),
                    subreddit VARCHAR(100),
                    description TEXT,
                    fetched_on VARCHAR(20),
                    used BOOLEAN DEFAULT FALSE
                )
            """

        with engine.begin() as conn:
            conn.execute(text(create_table_sql))
            existing_count = conn.execute(text("SELECT COUNT(*) FROM image_posts")).scalar()
            if existing_count and int(existing_count) > 0:
                return

        try:
            with open(image_feeds_path, "r") as f:
                feeds = json.load(f)
        except Exception as exc:
            logging.warning("Could not read image feeds from %s: %s", image_feeds_path, exc)
            return

        image_pattern = re.compile(r"\.(jpg|jpeg|png|gif|webp)(\?.*)?$", re.IGNORECASE)
        timestamp = datetime.datetime.now().strftime("%Y%m%d")

        def is_nsfw(entry):
            if hasattr(entry, "over_18") and entry.over_18:
                return True
            title = entry.title.lower() if hasattr(entry, "title") else ""
            return "[nsfw]" in title or "(nsfw)" in title

        def extract_image_url(entry):
            candidates = []

            if hasattr(entry, "link") and image_pattern.search(entry.link):
                url = entry.link
                if "preview.redd.it" not in url and "external-preview" not in url:
                    url = url.split("?")[0]
                candidates.append(url)

            if hasattr(entry, "media_content"):
                for media in entry.media_content:
                    url = media.get("url", "")
                    if image_pattern.search(url):
                        if "preview.redd.it" not in url and "external-preview" not in url:
                            url = url.split("?")[0]
                        candidates.append(url)

            if hasattr(entry, "media_thumbnail"):
                for thumb in entry.media_thumbnail:
                    url = thumb.get("url", "")
                    if url:
                        candidates.append(url)

            if hasattr(entry, "summary"):
                match = re.search(r'<img[^>]+src="([^"]+)"', entry.summary)
                if match:
                    candidates.append(match.group(1).replace("&amp;", "&"))

            if not candidates:
                return None

            for url in candidates:
                if "i.redd.it" in url or "imgur.com" in url:
                    return url
            for url in candidates:
                if "preview.redd.it" in url and "?" in url:
                    return url
            return candidates[0]

        added = 0
        with engine.begin() as conn:
            for item in feeds:
                if not isinstance(item, dict):
                    continue
                subreddit = str(item.get("subreddit", "")).strip().lower()
                subreddit = subreddit[2:] if subreddit.startswith("r/") else subreddit
                if not subreddit:
                    continue

                parsed = feedparser.parse(f"https://www.reddit.com/r/{subreddit}.rss")
                for entry in parsed.entries[:25]:
                    if is_nsfw(entry):
                        continue
                    image_url = extract_image_url(entry)
                    if not image_url:
                        continue
                    exists = conn.execute(
                        text("SELECT id FROM image_posts WHERE url = :url"),
                        {"url": image_url},
                    ).fetchone()
                    if exists:
                        continue
                    conn.execute(
                        text(
                            """
                            INSERT INTO image_posts (url, source_url, title, subreddit, fetched_on, used)
                            VALUES (:url, :source_url, :title, :subreddit, :fetched_on, false)
                            """
                        ),
                        {
                            "url": image_url,
                            "source_url": getattr(entry, "link", None),
                            "title": getattr(entry, "title", "")[:300],
                            "subreddit": subreddit,
                            "fetched_on": timestamp,
                        },
                    )
                    added += 1

        if added:
            logging.info("Seeded %s standalone images from %s", added, image_feeds_path)

    def read_agents(self):
        """
        Read the agents from the file

        :return:
        """
        import y_client.recsys as recsys
        import y_client.recsys as frecsys

        # population filename
        self.agents_filename = f"{self.base_path}{self.config['simulation']['population']}.json"
        data = json.load(open(self.agents_filename, "r"))
        skipped_pages = 0
        for ag in data["agents"]:
            is_page = ag.get("is_page", 0)
            if is_page != 1:
                content_recsys = getattr(recsys, ag["rec_sys"])()
                follow_recsys = getattr(frecsys, ag["frec_sys"])(leaning_bias=1.5)
                agent = Agent(
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

        skipped_pages = 0
        for a in agents["agents"]:
            try:
                if a.get("is_page", 0) != 1:
                    ag = Agent(
                        name=a["name"], email=a["email"], load=True, config=self.config, web=True
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

    def add_agent(self, agent=None):
        """
        Add an agent to the simulation

        :param agent: the agent to add
        """
        from y_client.utils import generate_user

        if agent is None:
            try:
                agent = generate_user(self.config, owner=self.agents_owner)

                if agent is None:
                    return
                agent.set_prompts(self.prompts)
                agent.set_rec_sys(self.content_recsys, self.follow_recsys)
            except Exception as e:
                logging.error(f"Error generating agent: {e}", exc_info=True)
                traceback.print_exc()
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
                    country=fdef.get("country", "")
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

            for feed in self.feed.feeds:
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
