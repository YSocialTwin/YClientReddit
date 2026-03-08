from sqlalchemy import orm
from sqlalchemy.ext.declarative import declarative_base
import sqlalchemy as db
import os
import os.path
import json
import shutil


try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    database_url = os.environ.get("DATABASE_URL")

    if database_url and "postgresql" in database_url:
        base = declarative_base()
        from sqlalchemy.pool import NullPool

        engine = db.create_engine(database_url, poolclass=NullPool)
        base.metadata.bind = engine
        session = orm.scoped_session(orm.sessionmaker())(bind=engine)
    else:
        import glob

        config_files = glob.glob(f"{BASE_DIR}/../../experiments/client_*.json")
        if config_files:
            config_path = config_files[0]
            config = json.load(open(config_path))
            db_name = config["simulation"]["name"]
        else:
            config = None
            db_name = None

        if config and not os.path.exists(f"{BASE_DIR}/../../experiments/{db_name}.db"):
            shutil.copyfile(
                f"{BASE_DIR}/../../data_schema/database_clean_client.db",
                f"{BASE_DIR}/../../experiments/{db_name}.db",
            )

        base = declarative_base()
        if db_name:
            engine = db.create_engine(f"sqlite:///experiments/{db_name}.db")
            base.metadata.bind = engine
            session = orm.scoped_session(orm.sessionmaker())(bind=engine)
        else:
            engine = None
            session = None
except Exception:
    from y_client.clients.client_web import base, session
    pass


class Articles(base):
    __tablename__ = "articles"
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    summary = db.Column(db.String(800))
    website_id = db.Column(db.Integer, nullable=False)
    fetched_on = db.Column(db.Integer, nullable=False)
    link = db.Column(db.String(200), nullable=False)


class Websites(base):
    __tablename__ = "websites"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    rss = db.Column(db.String(200), nullable=False)
    country = db.Column(db.String(50), nullable=False)
    language = db.Column(db.String(50), nullable=False)
    leaning = db.Column(db.String(50), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    last_fetched = db.Column(db.Integer, nullable=False)


class Images(base):
    __tablename__ = "images"
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(200), nullable=True)
    description = db.Column(db.String(400), nullable=True)
    article_id = db.Column(db.Integer, db.ForeignKey("articles.id"), nullable=True)
    remote_article_id = db.Column(db.Integer, nullable=True)


class ImagePosts(base):
    __tablename__ = "image_posts"

    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(500), nullable=False)
    source_url = db.Column(db.String(500), nullable=True)
    title = db.Column(db.String(300), nullable=True)
    subreddit = db.Column(db.String(100), nullable=True)
    description = db.Column(db.Text, nullable=True)
    fetched_on = db.Column(db.String(20), nullable=True)
    used = db.Column(db.Boolean, default=False)


class Agent_Custom_Prompt(base):
    __tablename__ = "agent_custom_prompt"
    id = db.Column(db.Integer, primary_key=True)
    agent_name = db.Column(db.TEXT, nullable=False)
    prompt = db.Column(db.TEXT, nullable=False)
