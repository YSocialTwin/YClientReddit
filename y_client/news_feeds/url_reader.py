import requests
import json
import re
from bs4 import BeautifulSoup
import datetime
from urllib.parse import urlparse

from y_client import content_store

EXCLUDED_DOMAINS = [
    'youtube.com', 'youtu.be', 'facebook.com', 'twitter.com', 't.co', 'reddit.com',
    'instagram.com', 'linkedin.com', 'pinterest.com', 'tiktok.com', 'discord.com',
    'telegram.org', 'vk.com', 'snapchat.com', 'tumblr.com', 'wechat.com', 'weibo.com',
]
EXCLUDED_EXTENSIONS = ['.gif', '.jpg', '.jpeg', '.png', '.webp', '.svg']

def is_valid_external_url(url):
    try:
        parsed = urlparse(url)
        if not parsed.scheme.startswith('http'):
            return False
        domain = parsed.netloc.lower()
        for ex in EXCLUDED_DOMAINS:
            if ex in domain:
                return False
        for ext in EXCLUDED_EXTENSIONS:
            if parsed.path.lower().endswith(ext):
                return False
        return True
    except Exception:
        return False

class NewsFromURL(object):
    def __init__(self, title, summary, link, published, image_url=None):
        self.title = title
        self.summary = summary
        self.link = link
        self.published = published
        self.image_url = image_url

    def save(self, website_name):
        website = content_store.ensure_website(
            name=website_name,
            rss="",
            country="Unknown",
            language="en",
            leaning="center",
            category="general",
            last_fetched=self.published,
        )
        if website is None:
            return
        content_store.save_article(
            website_name=website.name,
            rss=website.rss,
            title=self.title,
            summary=self.summary,
            published=self.published,
            link=self.link,
            image_url=self.image_url,
        )

class URLReader(object):
    def __init__(self, urls, website_name_prefix="Website"):  # urls: list of URLs
        self.urls = [u for u in urls if is_valid_external_url(u)]
        self.website_name_prefix = website_name_prefix
        self.articles = []

    def process_urls(self):
        today = datetime.datetime.now()
        timestamp = int(today.strftime("%Y%m%d"))
        stats = {
            "total_urls": len(self.urls),
            "processed": 0,
            "errors": 0,
            "images_added": 0,
            "skipped": 0
        }
        for idx, url in enumerate(self.urls):
            try:
                print(f"Processing URL {idx+1}/{len(self.urls)}: {url}")
                resp = requests.get(url, timeout=10)
                if resp.status_code != 200 or "text/html" not in resp.headers.get("content-type", ""):
                    print(f"Skipping URL (bad response or not HTML): {url}")
                    stats["skipped"] += 1
                    continue
                soup = BeautifulSoup(resp.text, features="html.parser")
                # Extract title
                title = soup.title.string.strip() if soup.title and soup.title.string else url
                # Extract summary/description
                summary = ""
                desc_tag = soup.find("meta", attrs={"name": "description"})
                if desc_tag and desc_tag.get("content"):
                    summary = desc_tag["content"].strip()
                elif soup.find("p"):
                    summary = soup.find("p").get_text().strip()
                # Extract published date
                published = timestamp
                date_tag = soup.find("meta", attrs={"property": "article:published_time"})
                if date_tag and date_tag.get("content"):
                    try:
                        published = int(date_tag["content"].split("T")[0].replace("-", ""))
                    except:
                        published = timestamp
                # Extract image URL
                image_url = None
                img_tag = soup.find("meta", attrs={"property": "og:image"})
                if img_tag and img_tag.get("content"):
                    image_url = img_tag["content"].split("?")[0]
                elif soup.find("img"):
                    image_url = soup.find("img").get("src")
                # Validate article details
                if not title or len(title) < 5:
                    print(f"Skipping URL (no valid title found): {url}")
                    stats["skipped"] += 1
                    continue
                if not summary or len(summary) < 10:
                    print(f"Skipping URL (no valid summary found): {url}")
                    stats["skipped"] += 1
                    continue
                website_name = f"{self.website_name_prefix}_{idx+1}"
                article = NewsFromURL(title, summary, url, published, image_url)
                article.save(website_name)
                self.articles.append(article)
                stats["processed"] += 1
                if image_url:
                    stats["images_added"] += 1
            except Exception as e:
                print(f"Error processing URL {url}: {str(e)}")
                stats["errors"] += 1
        print("\n====== URL News Processing Summary ======")
        print(f"Total URLs processed: {stats['total_urls']}")
        print(f"Successfully processed: {stats['processed']}")
        print(f"Skipped (invalid, social, or not news): {stats['skipped']}")
        print(f"Errors encountered: {stats['errors']}")
        print(f"Images added: {stats['images_added']}")
        print(f"Total articles in memory: {len(self.articles)}")
        return stats
