import os
import random

from sqlalchemy import or_, text
from sqlalchemy.sql.expression import func

from y_client.news_feeds.client_modals import (
    Agent_Custom_Prompt,
    Articles,
    ImagePosts,
    Images,
    Websites,
    base,
    get_engine,
    get_session,
    initialize_client_db,
)


def initialize_content_store(*, data_base_path=None, experiment_name=None):
    database_url = os.environ.get("CONTENT_DATABASE_URL")
    if database_url:
        return initialize_client_db(database_url=database_url)

    db_path = None
    if data_base_path:
        db_path = os.path.join(data_base_path, "client_content.db")
    elif experiment_name:
        db_path = os.path.join("experiments", f"{experiment_name}.db")
    return initialize_client_db(db_path=db_path)


def get_bindings():
    return get_session(), get_engine(), base


def reset_content_db():
    session = get_session()
    if session is None:
        return
    session.query(Agent_Custom_Prompt).delete()
    session.query(ImagePosts).delete()
    session.query(Images).delete()
    session.query(Articles).delete()
    session.query(Websites).delete()
    session.commit()


def save_agent_custom_prompt(agent_name, prompt):
    session = get_session()
    if session is None or prompt is None:
        return
    existing = session.query(Agent_Custom_Prompt).filter_by(agent_name=agent_name).first()
    if existing is None:
        existing = Agent_Custom_Prompt(agent_name=agent_name, prompt=prompt)
        session.add(existing)
    else:
        existing.prompt = prompt
    session.commit()


def get_agent_custom_prompt(agent_name):
    session = get_session()
    if session is None:
        return None
    return session.query(Agent_Custom_Prompt).filter_by(agent_name=agent_name).first()


def get_website(name=None, rss=None, website_id=None):
    session = get_session()
    if session is None:
        return None
    query = session.query(Websites)
    if website_id is not None:
        return query.filter(Websites.id == website_id).first()
    if name is not None:
        query = query.filter(Websites.name == name)
    if rss is not None:
        query = query.filter(Websites.rss == rss)
    return query.first()


def website_exists(name, rss):
    return get_website(name=name, rss=rss) is not None


def ensure_website(
    *,
    name,
    rss,
    country,
    language,
    leaning,
    category,
    last_fetched,
    fetch_images_from_url=False,
    fetch_images_timeout=10,
):
    session = get_session()
    if session is None:
        return None
    website = get_website(name=name, rss=rss)
    if website is None:
        website = Websites(
            name=name,
            rss=rss,
            country=country,
            language=language,
            leaning=leaning,
            category=category,
            last_fetched=last_fetched,
            fetch_images_from_url=fetch_images_from_url,
            fetch_images_timeout=fetch_images_timeout,
        )
        session.add(website)
        session.commit()
    return website


def update_website_last_fetched(name, rss, timestamp):
    website = get_website(name=name, rss=rss)
    if website is None:
        return
    website.last_fetched = timestamp
    get_session().commit()


def save_article(*, website_name, rss, title, summary, published, link, image_url=None):
    session = get_session()
    if session is None:
        return None
    website = get_website(name=website_name, rss=rss)
    if website is None:
        return None
    article = (
        session.query(Articles)
        .filter(Articles.link == link, Articles.website_id == website.id)
        .first()
    )
    if article is None:
        article = Articles(
            title=title,
            summary=summary,
            website_id=website.id,
            fetched_on=published,
            link=link,
        )
        session.add(article)
        session.commit()
    if image_url is not None:
        ensure_article_image(image_url, article.id)
    return article


def get_recent_articles_for_feed(name, rss, limit=10):
    session = get_session()
    website = get_website(name=name, rss=rss)
    if session is None or website is None:
        return []
    return (
        session.query(Articles)
        .filter(Articles.website_id == website.id)
        .order_by(Articles.id.desc())
        .limit(limit)
        .all()
    )


def ensure_article_image(url, article_id):
    session = get_session()
    if session is None or url is None:
        return None
    image = session.query(Images).filter(Images.url == url).first()
    if image is None:
        image = Images(url=url, article_id=article_id)
        session.add(image)
        session.commit()
    return image


def get_image_by_article_id(article_id):
    session = get_session()
    if session is None:
        return None
    return session.query(Images).filter(Images.article_id == article_id).first()


def get_random_website_by_leaning(leaning):
    session = get_session()
    if session is None:
        return None
    candidates = session.query(Websites).filter(Websites.leaning == leaning).all()
    if not candidates:
        candidates = session.query(Websites).all()
    if not candidates:
        return None
    return random.choice(candidates)


def get_random_news_article_for_leaning(leaning):
    session = get_session()
    if session is None:
        return None, None, None
    website_ids = [row[0] for row in session.query(Websites.id).filter(Websites.leaning == leaning).all()]
    if not website_ids:
        website_ids = [row[0] for row in session.query(Websites.id).all()]
    if not website_ids:
        return None, None, None
    articles = (
        session.query(Articles)
        .filter(Articles.website_id.in_(website_ids))
        .order_by(func.random())
        .limit(10)
        .all()
    )
    if not articles:
        return None, None, None
    article = random.choice(articles)
    website = get_website(website_id=article.website_id)
    image = get_image_by_article_id(article.id)
    return article, website, image


def find_articles_matching_interests(interests, limit=6):
    session = get_session()
    if session is None:
        return []
    search_conditions = []
    for interest in interests or []:
        if interest and len(interest) > 2:
            pattern = f"%{str(interest).lower()}%"
            search_conditions.append(Articles.title.ilike(pattern))
            search_conditions.append(Articles.summary.ilike(pattern))
    if not search_conditions:
        return []
    return (
        session.query(Articles)
        .filter(or_(*search_conditions))
        .order_by(func.random())
        .limit(limit)
        .all()
    )


def get_random_articles(limit=6):
    session = get_session()
    if session is None:
        return []
    return session.query(Articles).order_by(func.random()).limit(limit).all()


def get_article(article_id):
    session = get_session()
    if session is None:
        return None
    return session.query(Articles).filter(Articles.id == article_id).first()


def get_random_image():
    session = get_session()
    if session is None:
        return None
    return session.query(Images).order_by(func.random()).first()


def save_image_description(image_id, description):
    image = get_image(image_id)
    if image is None:
        return None
    image.description = description
    get_session().commit()
    return image


def save_image_remote_article(image_id, remote_article_id):
    image = get_image(image_id)
    if image is None:
        return None
    image.remote_article_id = remote_article_id
    get_session().commit()
    return image


def delete_image(image_id):
    session = get_session()
    image = get_image(image_id)
    if session is None or image is None:
        return
    session.delete(image)
    session.commit()


def get_image(image_id):
    session = get_session()
    if session is None:
        return None
    return session.query(Images).filter(Images.id == image_id).first()


def get_image_with_article_and_website(image_id):
    image = get_image(image_id)
    if image is None:
        return None, None, None
    article = get_article(image.article_id)
    website = get_website(website_id=article.website_id) if article is not None else None
    return image, article, website


def count_image_posts():
    engine = get_engine()
    if engine is None:
        return 0
    with engine.connect() as conn:
        return int(conn.execute(text("SELECT COUNT(*) FROM image_posts")).scalar() or 0)


def image_post_exists(url):
    session = get_session()
    if session is None:
        return False
    return session.query(ImagePosts).filter(ImagePosts.url == url).first() is not None


def create_image_post(**kwargs):
    session = get_session()
    if session is None:
        return None
    row = ImagePosts(**kwargs)
    session.add(row)
    session.commit()
    return row


def update_image_post_local_path(image_post_id, local_path):
    session = get_session()
    if session is None:
        return
    row = session.query(ImagePosts).filter(ImagePosts.id == image_post_id).first()
    if row is None:
        return
    row.local_path = local_path
    session.commit()


def delete_image_post(image_post_id):
    session = get_session()
    if session is None:
        return
    row = session.query(ImagePosts).filter(ImagePosts.id == image_post_id).first()
    if row is None:
        return
    session.delete(row)
    session.commit()


def get_candidate_image_posts(*, subreddits=None, require_description=None, exclude_ids=None, limit=6):
    session = get_session()
    if session is None:
        return []
    query = session.query(ImagePosts).filter(ImagePosts.used == False)
    if subreddits:
        query = query.filter(ImagePosts.subreddit.in_(list(subreddits)))
    if require_description is True:
        query = query.filter(ImagePosts.description.isnot(None))
    elif require_description is False:
        query = query.filter(ImagePosts.description.is_(None))
    if exclude_ids:
        query = query.filter(~ImagePosts.id.in_(list(exclude_ids)))
    return query.order_by(func.random()).limit(limit).all()


def mark_image_post_used(image_post_id):
    return set_image_post_used(image_post_id, True)


def set_image_post_used(image_post_id, used):
    session = get_session()
    if session is None:
        return None
    row = session.query(ImagePosts).filter(ImagePosts.id == image_post_id).first()
    if row is None:
        return None
    row.used = bool(used)
    session.commit()
    return row


def update_image_post_description(image_post_id, description):
    session = get_session()
    if session is None:
        return None
    row = session.query(ImagePosts).filter(ImagePosts.id == image_post_id).first()
    if row is None:
        return None
    row.description = description
    session.commit()
    return row
