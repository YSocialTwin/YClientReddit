import json
import os
import os.path
import shutil

import sqlalchemy as db
from sqlalchemy import orm, text
from sqlalchemy.ext.declarative import declarative_base


base = declarative_base()
engine = None
session = None


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


class StressReward(base):
    __tablename__ = "stress_reward"
    __table_args__ = (
        db.CheckConstraint(
            "variable IN ('stress', 'reward')", name="ck_stress_reward_variable"
        ),
        db.CheckConstraint(
            "type IN ('aggregate', 'variation')", name="ck_stress_reward_type"
        ),
        db.CheckConstraint("value >= 0 AND value <= 1", name="ck_stress_reward_value"),
    )

    id = db.Column(db.String(36), primary_key=True)
    uid = db.Column(db.Integer, db.ForeignKey("user_mgmt.id"), nullable=False, index=True)
    variable = db.Column(db.String(16), nullable=False)
    value = db.Column(db.Float, nullable=False)
    type = db.Column(db.String(16), nullable=False)
    tid = db.Column(db.Integer, db.ForeignKey("rounds.id"), nullable=False, index=True)


def _legacy_default_db_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    database_url = os.environ.get("CONTENT_DATABASE_URL")
    if database_url:
        return None, database_url

    try:
        import glob

        config_files = glob.glob(f"{base_dir}/../../experiments/client_*.json")
        if config_files:
            config = json.load(open(config_files[0]))
            db_name = config["simulation"]["name"]
            db_path = f"{base_dir}/../../experiments/{db_name}.db"
            return db_path, None
    except Exception:
        pass

    return None, None


def _ensure_sqlite_seed_db(target_path):
    if target_path is None or os.path.exists(target_path):
        return
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    shutil.copyfile(
        f"{base_dir}/../../data_schema/database_clean_client.db",
        target_path,
    )


def _ensure_sqlite_schema_compatibility():
    if engine is None or engine.dialect.name != "sqlite":
        return

    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS stress_reward (
                        id VARCHAR(36) PRIMARY KEY,
                        uid INTEGER NOT NULL,
                        variable VARCHAR(16) NOT NULL,
                        value FLOAT NOT NULL,
                        type VARCHAR(16) NOT NULL,
                        tid INTEGER NOT NULL,
                        FOREIGN KEY(uid) REFERENCES user_mgmt(id),
                        FOREIGN KEY(tid) REFERENCES rounds(id),
                        CONSTRAINT ck_stress_reward_variable
                            CHECK (variable IN ('stress', 'reward')),
                        CONSTRAINT ck_stress_reward_type
                            CHECK (type IN ('aggregate', 'variation')),
                        CONSTRAINT ck_stress_reward_value
                            CHECK (value >= 0 AND value <= 1)
                    )
                    """
                )
            )

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
                    conn.execute(text("ALTER TABLE websites ADD COLUMN fetch_images_from_url BOOLEAN DEFAULT 0"))
                if "fetch_images_timeout" not in website_columns:
                    conn.execute(text("ALTER TABLE websites ADD COLUMN fetch_images_timeout INTEGER DEFAULT 10"))

            image_post_columns = {
                row[1] for row in conn.execute(text("PRAGMA table_info(image_posts)")).fetchall()
            }
            if "local_path" not in image_post_columns:
                conn.execute(text("ALTER TABLE image_posts ADD COLUMN local_path VARCHAR(500)"))
            if "high_res_url" not in image_post_columns:
                conn.execute(text("ALTER TABLE image_posts ADD COLUMN high_res_url VARCHAR(500)"))
    except Exception:
        pass


def initialize_client_db(*, db_path=None, database_url=None):
    global engine, session

    if database_url is None and db_path is None:
        db_path, database_url = _legacy_default_db_path()

    if database_url:
        engine = db.create_engine(database_url)
    elif db_path:
        _ensure_sqlite_seed_db(db_path)
        engine = db.create_engine(
            f"sqlite:////{os.path.abspath(db_path)}",
            connect_args={"check_same_thread": False, "timeout": 30},
        )
    else:
        engine = None
        session = None
        return None, None, base

    base.metadata.bind = engine
    session = orm.scoped_session(orm.sessionmaker())(bind=engine)
    _ensure_sqlite_schema_compatibility()
    return session, engine, base


def get_session():
    return session


def get_engine():
    return engine


initialize_client_db()
