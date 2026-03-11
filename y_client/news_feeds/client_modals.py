from sqlalchemy import orm, text
from sqlalchemy.ext.declarative import declarative_base
import sqlalchemy as db
import os
import os.path
import json
import shutil


try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Check for PostgreSQL via DATABASE_URL environment variable first
    database_url = os.environ.get("DATABASE_URL")

    if database_url and "postgresql" in database_url:
        # PostgreSQL mode - use DATABASE_URL
        base = declarative_base()
        from sqlalchemy.pool import NullPool
        engine = db.create_engine(database_url, poolclass=NullPool)
        base.metadata.bind = engine
        session = orm.scoped_session(orm.sessionmaker())(bind=engine)
    else:
        # SQLite mode - try to find config file
        import glob
        config_files = glob.glob(f"{BASE_DIR}/../../experiments/client_*.json")
        if config_files:
            config_path = config_files[0]
            config = json.load(open(config_path))
            db_name = config['simulation']['name']
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
    engine = getattr(session, "bind", None) if session is not None else None
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
    fetch_images_from_url = db.Column(db.Boolean, default=False)
    fetch_images_timeout = db.Column(db.Integer, default=10)


class Images(base):
    __tablename__ = "images"
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(200), nullable=True)
    description = db.Column(db.String(400), nullable=True)
    article_id = db.Column(db.Integer, db.ForeignKey("articles.id"), nullable=True)
    remote_article_id = db.Column(db.Integer, nullable=True)


class ImagePosts(base):
    """Standalone images from image-focused feeds (Reddit RSS, etc.)"""
    __tablename__ = "image_posts"
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(500), nullable=False)
    source_url = db.Column(db.String(500), nullable=True)
    title = db.Column(db.String(300), nullable=True)
    subreddit = db.Column(db.String(100), nullable=True)
    description = db.Column(db.Text, nullable=True)
    fetched_on = db.Column(db.String(20), nullable=True)
    used = db.Column(db.Boolean, default=False)
    local_path = db.Column(db.String(500), nullable=True)
    high_res_url = db.Column(db.String(500), nullable=True)


class Agent_Custom_Prompt(base):
    __tablename__ = "agent_custom_prompt"
    id = db.Column(db.Integer, primary_key=True)
    agent_name = db.Column(db.TEXT, nullable=False)
    prompt = db.Column(db.TEXT, nullable=False)


def _ensure_sqlite_schema_compatibility():
    if engine is None or engine.dialect.name != "sqlite":
        return

    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS image_posts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        url VARCHAR(500) NOT NULL,
                        source_url VARCHAR(500),
                        title VARCHAR(300),
                        subreddit VARCHAR(100),
                        description TEXT,
                        fetched_on VARCHAR(20),
                        used BOOLEAN DEFAULT 0,
                        local_path VARCHAR(500),
                        high_res_url VARCHAR(500)
                    )
                    """
                )
            )

            website_columns = {
                row[1] for row in conn.execute(text("PRAGMA table_info(websites)")).fetchall()
            }
            if website_columns:
                if "fetch_images_from_url" not in website_columns:
                    conn.execute(
                        text(
                            "ALTER TABLE websites ADD COLUMN fetch_images_from_url BOOLEAN DEFAULT 0"
                        )
                    )
                if "fetch_images_timeout" not in website_columns:
                    conn.execute(
                        text(
                            "ALTER TABLE websites ADD COLUMN fetch_images_timeout INTEGER DEFAULT 10"
                        )
                    )

            image_post_columns = {
                row[1] for row in conn.execute(text("PRAGMA table_info(image_posts)")).fetchall()
            }
            if "local_path" not in image_post_columns:
                conn.execute(
                    text("ALTER TABLE image_posts ADD COLUMN local_path VARCHAR(500)")
                )
            if "high_res_url" not in image_post_columns:
                conn.execute(
                    text("ALTER TABLE image_posts ADD COLUMN high_res_url VARCHAR(500)")
                )
    except Exception:
        pass


_ensure_sqlite_schema_compatibility()
