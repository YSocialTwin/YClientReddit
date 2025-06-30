import json
import sys
import os
import shutil
import logging
import traceback
from sqlalchemy.ext.declarative import declarative_base
import sqlalchemy as db
from requests import post
from sqlalchemy import orm



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
session = None
engine = None
base = None


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
        """

        self.first_run = first_run
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
        
        # Use the Y_Web experiment database instead of creating our own
        if data_base_path and os.path.exists(f"{data_base_path}database_server.db"):
            # Use the Y_Web experiment database
            db_path = f"{data_base_path}database_server.db"
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
            logging.info(f"Using standalone database: {db_path}")

        global session, engine, base
        base = declarative_base()

        engine = db.create_engine(f"sqlite:////{db_path}")
        base.metadata.bind = engine
        session = orm.scoped_session(orm.sessionmaker())(bind=engine)

        globals()["session"] = session
        globals()["engine"] = engine
        globals()["base"] = base
        ##############

        yclient_path = os.path.dirname(os.path.abspath(__file__)).split("y_web")[0]
        sys.path.append(f'{yclient_path}{os.sep}external{os.sep}YClient/')

        from y_client.classes import Agent, Agents, SimulationSlot
        from y_client.news_feeds import Feeds

        # initialize simulation clock
        self.sim_clock = SimulationSlot(self.config)

        self.agents = Agents()
        self.feed = Feeds()
        self.content_recsys = None
        self.follow_recsys = None
        self.network = network

    def read_agents(self):
        """
        Read the agents from the file

        :return:
        """
        from y_client.classes import Agent
        import y_client.recsys as recsys
        import y_client.recsys as frecsys

        # population filename
        self.agents_filename = f"{self.base_path}{self.config['simulation']['population']}.json"
        data = json.load(open(self.agents_filename, "r"))
        for ag in data['agents']:
            if ag["is_page"] == 0:
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
                    education_level=ag["education_level"],
                    round_actions=ag["round_actions"],
                    nationality=ag["nationality"],
                    profession=ag["profession"],
                    toxicity=ag["toxicity"],
                    gender=ag["gender"],
                    age=ag["age"],
                    recsys=content_recsys,
                    frecsys=follow_recsys,
                    language=ag["language"],
                    owner=ag["owner"],
                    config=self.config,
                    load=not self.first_run,
                    web=True,
                    prompt=ag["prompts"],
                )
                agent.set_prompts(self.prompts)
                self.agents.add_agent(agent)

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
        from y_client.classes import Agent

        for a in agents["agents"]:
            try:
                if a["is_page"] == 0:
                    ag = Agent(
                        name=a["name"], email=a["email"], load=True, config=self.config, web=True
                    )
                    ag.set_prompts(self.prompts)
                    ag.set_rec_sys(self.content_recsys, self.follow_recsys)
                    self.agents.add_agent(ag)
            except Exception:
                logging.exception(f"Error loading agent: {a['name']}")
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