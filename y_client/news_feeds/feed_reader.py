import feedparser
import numpy as np
import json
import requests, re
import time
from urllib.parse import urlparse, urlunparse
from bs4 import BeautifulSoup
from typing import Optional, Dict
try:
    from .client_modals import Websites, Articles, Images, session
except:
    from y_client.clients.client_web import session
    from .client_modals import Websites, Articles, Images
import datetime

# Domains to skip when fetching OG metadata (social media that requires auth or has no useful OG)
SKIP_DOMAINS_FOR_OG_FETCH = [
    'facebook.com', 'twitter.com', 't.co', 'x.com',
    'instagram.com', 'linkedin.com', 'pinterest.com', 'tiktok.com',
    'reddit.com', 'discord.com', 'telegram.org'
]

# Video platforms with special handling
YOUTUBE_DOMAINS = ['youtube.com', 'youtu.be', 'www.youtube.com', 'm.youtube.com']
VIMEO_DOMAINS = ['vimeo.com', 'www.vimeo.com', 'player.vimeo.com']

USER_AGENT = "YSocial/1.0 RSS Feed Reader"
REDDIT_DOMAINS = {"reddit.com", "www.reddit.com", "redd.it", "redditmedia.com", "external-preview.redd.it"}

# Domains considered external (not Reddit) for OG metadata fetching
REDDIT_INTERNAL_DOMAINS = [
    'reddit.com', 'www.reddit.com', 'old.reddit.com',
    'i.redd.it', 'v.redd.it', 'preview.redd.it',
    'external-preview.redd.it', 'redditmedia.com'
]


def _is_reddit_url(url: str) -> bool:
    if not url:
        return False
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return host in REDDIT_DOMAINS or host.endswith(".reddit.com") or host.endswith(".redd.it")


def is_external_url(url: str) -> bool:
    """
    Check if URL points to an external site (not Reddit).

    Args:
        url: URL to check

    Returns:
        True if the URL is external (not Reddit), False otherwise
    """
    if not url:
        return False
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        return not any(rd in domain for rd in REDDIT_INTERNAL_DOMAINS)
    except Exception:
        return False


def clean_reddit_formatting(text: str) -> str:
    """
    Remove Reddit-specific formatting from text.

    Cleans up RSS entry text that contains Reddit submission metadata
    like "submitted by /u/username" and "[link] [comments]" patterns.

    Args:
        text: Text to clean

    Returns:
        Cleaned text with Reddit formatting removed
    """
    if not text:
        return ''
    # Remove "submitted by /u/username" patterns (allow hyphens/underscores)
    text = re.sub(r'submitted by\s+/u/[A-Za-z0-9_-]+', '', text, flags=re.IGNORECASE)
    # Remove [link] [comments] patterns
    text = re.sub(r'\[link\]|\[comments\]', '', text, flags=re.IGNORECASE)
    # Remove Reddit HTML artifacts and CDATA sections
    text = re.sub(r'<!\[CDATA\[.*?\]\]>', '', text, flags=re.DOTALL)
    # Remove HTML table structures often found in Reddit RSS
    text = re.sub(r'<table>.*?</table>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_youtube_video_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from various YouTube URL formats.

    Supports:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://www.youtube.com/v/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID

    Args:
        url: YouTube URL

    Returns:
        Video ID string or None if not found
    """
    if not url:
        return None

    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})',
        r'[?&]v=([a-zA-Z0-9_-]{11})',
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def extract_youtube_metadata(url: str, timeout: int = 10) -> Optional[Dict[str, str]]:
    """
    Extract metadata from a YouTube video URL.

    Uses YouTube's oembed API to get title and author, and constructs
    thumbnail URLs using the standard YouTube thumbnail URL pattern.

    Args:
        url: YouTube video URL
        timeout: Request timeout in seconds

    Returns:
        Dict with 'title', 'description', 'image', 'site_name', 'video_id', 'embed_url'
        or None if extraction fails
    """
    video_id = extract_youtube_video_id(url)
    if not video_id:
        return None

    result = {
        'title': None,
        'description': None,
        'image': None,
        'site_name': 'YouTube',
        'video_id': video_id,
        'embed_url': f'https://www.youtube.com/embed/{video_id}',
        'content_type': 'video',
    }

    # Get high-quality thumbnail (try maxresdefault first, fall back to hqdefault)
    thumbnail_urls = [
        f'https://img.youtube.com/vi/{video_id}/maxresdefault.jpg',
        f'https://img.youtube.com/vi/{video_id}/sddefault.jpg',
        f'https://img.youtube.com/vi/{video_id}/hqdefault.jpg',
    ]

    # Check which thumbnail exists (maxresdefault might not exist for all videos)
    for thumb_url in thumbnail_urls:
        try:
            head_resp = requests.head(thumb_url, timeout=5)
            if head_resp.status_code == 200:
                result['image'] = thumb_url
                break
        except Exception:
            continue

    # If no thumbnail found via HEAD, use hqdefault as it always exists
    if not result['image']:
        result['image'] = f'https://img.youtube.com/vi/{video_id}/hqdefault.jpg'

    # Try to get title/description via oembed API
    try:
        oembed_url = f'https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json'
        headers = {'User-Agent': USER_AGENT}
        response = requests.get(oembed_url, timeout=timeout, headers=headers)

        if response.status_code == 200:
            data = response.json()
            result['title'] = data.get('title')
            result['description'] = f"Video by {data.get('author_name', 'Unknown')}"
            if data.get('thumbnail_url'):
                # oembed might return a better thumbnail
                result['image'] = data['thumbnail_url']
    except Exception:
        # oembed failed, but we still have the thumbnail
        pass

    return result


def extract_vimeo_metadata(url: str, timeout: int = 10) -> Optional[Dict[str, str]]:
    """
    Extract metadata from a Vimeo video URL.

    Uses Vimeo's oembed API to get title, description, and thumbnail.

    Args:
        url: Vimeo video URL
        timeout: Request timeout in seconds

    Returns:
        Dict with 'title', 'description', 'image', 'site_name', 'video_id', 'embed_url'
        or None if extraction fails
    """
    # Extract video ID from URL
    match = re.search(r'vimeo\.com/(?:video/)?(\d+)', url)
    if not match:
        return None

    video_id = match.group(1)

    result = {
        'title': None,
        'description': None,
        'image': None,
        'site_name': 'Vimeo',
        'video_id': video_id,
        'embed_url': f'https://player.vimeo.com/video/{video_id}',
        'content_type': 'video',
    }

    # Try oembed API
    try:
        oembed_url = f'https://vimeo.com/api/oembed.json?url={url}'
        headers = {'User-Agent': USER_AGENT}
        response = requests.get(oembed_url, timeout=timeout, headers=headers)

        if response.status_code == 200:
            data = response.json()
            result['title'] = data.get('title')
            result['description'] = data.get('description', f"Video by {data.get('author_name', 'Unknown')}")
            result['image'] = data.get('thumbnail_url')
            return result
    except Exception:
        pass

    return None


def is_video_url(url: str) -> bool:
    """
    Check if URL is a video platform URL (YouTube, Vimeo, etc.).

    Args:
        url: URL to check

    Returns:
        True if URL is from a video platform
    """
    if not url:
        return False
    try:
        domain = urlparse(url).netloc.lower()
        return (
            any(yd in domain for yd in YOUTUBE_DOMAINS) or
            any(vd in domain for vd in VIMEO_DOMAINS)
        )
    except Exception:
        return False


def extract_video_metadata(url: str, timeout: int = 10) -> Optional[Dict[str, str]]:
    """
    Extract metadata from video platform URLs (YouTube, Vimeo).

    Dispatcher function that routes to the appropriate extractor.

    Args:
        url: Video URL
        timeout: Request timeout in seconds

    Returns:
        Dict with video metadata or None if extraction fails
    """
    if not url:
        return None

    try:
        domain = urlparse(url).netloc.lower()

        if any(yd in domain for yd in YOUTUBE_DOMAINS):
            return extract_youtube_metadata(url, timeout)
        elif any(vd in domain for vd in VIMEO_DOMAINS):
            return extract_vimeo_metadata(url, timeout)
    except Exception:
        pass

    return None


def extract_og_metadata(url: str, timeout: int = 10) -> Optional[Dict[str, str]]:
    """
    Extract metadata from a URL using multiple strategies.

    Tries in order:
    1. OpenGraph meta tags (og:title, og:description, og:image)
    2. Twitter Card meta tags (twitter:title, twitter:description, twitter:image)
    3. JSON-LD schema.org structured data
    4. Standard HTML tags (<title>, <meta name="description">)

    Args:
        url: URL to fetch metadata from
        timeout: Request timeout in seconds (default: 10)

    Returns:
        Dict with keys 'title', 'description', 'image', 'site_name',
        or None if extraction fails
    """
    try:
        # Check if URL domain should be skipped (social media that requires auth)
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        for skip_domain in SKIP_DOMAINS_FOR_OG_FETCH:
            # Use proper domain matching (not substring) to avoid false positives
            # e.g., "t.co" should not match "thedailybeast.com"
            if domain == skip_domain or domain.endswith('.' + skip_domain):
                return None

        # Video platforms are handled separately via extract_video_metadata()
        if is_video_url(url):
            return None

        # Fetch the page
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        response = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)

        # Skip non-HTML responses
        content_type = response.headers.get('content-type', '')
        if 'text/html' not in content_type.lower():
            return None

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, features="html.parser")

        result = {
            'title': None,
            'description': None,
            'image': None,
            'site_name': None,
        }

        # === TITLE ===
        # Try og:title first
        og_title = soup.find("meta", attrs={"property": "og:title"})
        if og_title and og_title.get("content"):
            result['title'] = og_title["content"].strip()
        else:
            # Try twitter:title
            twitter_title = soup.find("meta", attrs={"name": "twitter:title"})
            if twitter_title and twitter_title.get("content"):
                result['title'] = twitter_title["content"].strip()
            else:
                # Fallback to <title> tag
                title_tag = soup.find("title")
                if title_tag and title_tag.string:
                    result['title'] = title_tag.string.strip()

        # === DESCRIPTION ===
        # Try og:description first
        og_desc = soup.find("meta", attrs={"property": "og:description"})
        if og_desc and og_desc.get("content"):
            result['description'] = og_desc["content"].strip()
        else:
            # Try twitter:description
            twitter_desc = soup.find("meta", attrs={"name": "twitter:description"})
            if twitter_desc and twitter_desc.get("content"):
                result['description'] = twitter_desc["content"].strip()
            else:
                # Fallback to <meta name="description">
                meta_desc = soup.find("meta", attrs={"name": "description"})
                if meta_desc and meta_desc.get("content"):
                    result['description'] = meta_desc["content"].strip()

        # === IMAGE ===
        # Try og:image first
        og_image = soup.find("meta", attrs={"property": "og:image"})
        if og_image and og_image.get("content"):
            result['image'] = og_image["content"]
        else:
            # Try twitter:image
            twitter_image = soup.find("meta", attrs={"name": "twitter:image"})
            if twitter_image and twitter_image.get("content"):
                result['image'] = twitter_image["content"]

        # === SITE NAME ===
        og_site = soup.find("meta", attrs={"property": "og:site_name"})
        if og_site and og_site.get("content"):
            result['site_name'] = og_site["content"].strip()
        else:
            # Try twitter:site
            twitter_site = soup.find("meta", attrs={"name": "twitter:site"})
            if twitter_site and twitter_site.get("content"):
                result['site_name'] = twitter_site["content"].strip()

        # === JSON-LD FALLBACK ===
        # If we're missing title, description, or image, try JSON-LD schema.org data
        if not result['title'] or not result['description'] or not result['image']:
            json_ld_scripts = soup.find_all("script", attrs={"type": "application/ld+json"})
            for script in json_ld_scripts:
                try:
                    if script.string:
                        data = json.loads(script.string)
                        # Handle both single objects and arrays
                        items = data if isinstance(data, list) else [data]
                        for item in items:
                            # Look for Article, NewsArticle, BlogPosting, WebPage types
                            item_type = item.get("@type", "")
                            if item_type in ["Article", "NewsArticle", "BlogPosting", "WebPage", "Product"]:
                                if not result['title'] and item.get("headline"):
                                    result['title'] = item["headline"]
                                if not result['title'] and item.get("name"):
                                    result['title'] = item["name"]
                                if not result['description'] and item.get("description"):
                                    result['description'] = item["description"]
                                if not result['image']:
                                    img_data = item.get("image")
                                    if img_data:
                                        if isinstance(img_data, str):
                                            result['image'] = img_data
                                        elif isinstance(img_data, dict) and img_data.get("url"):
                                            result['image'] = img_data["url"]
                                        elif isinstance(img_data, list) and len(img_data) > 0:
                                            first_img = img_data[0]
                                            if isinstance(first_img, str):
                                                result['image'] = first_img
                                            elif isinstance(first_img, dict) and first_img.get("url"):
                                                result['image'] = first_img["url"]
                                if not result['site_name'] and item.get("publisher"):
                                    publisher = item["publisher"]
                                    if isinstance(publisher, dict) and publisher.get("name"):
                                        result['site_name'] = publisher["name"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

        # === FIRST IMAGE FALLBACK ===
        # If still no image, find the first <img> tag in the article body
        if not result['image']:
            # Look for images in article/main content areas first
            content_areas = soup.find_all(['article', 'main', 'div'], class_=lambda c: c and any(
                x in str(c).lower() for x in ['content', 'article', 'post', 'entry', 'story']
            ))

            img_tag = None
            for area in content_areas:
                img_tag = area.find('img', src=True)
                if img_tag:
                    break

            # Fallback to any img in body
            if not img_tag:
                body = soup.find('body')
                if body:
                    img_tag = body.find('img', src=True)

            if img_tag:
                img_src = img_tag.get('src', '')
                # Skip tiny images (likely icons/trackers), data URIs, and SVGs
                if img_src and not img_src.startswith('data:') and '.svg' not in img_src.lower():
                    # Check for width/height attributes to skip small images
                    width = img_tag.get('width', '')
                    height = img_tag.get('height', '')
                    is_small = False
                    try:
                        if width and int(width) < 100:
                            is_small = True
                        if height and int(height) < 100:
                            is_small = True
                    except (ValueError, TypeError):
                        pass

                    if not is_small:
                        # Make relative URLs absolute
                        if img_src.startswith('//'):
                            img_src = 'https:' + img_src
                        elif img_src.startswith('/'):
                            img_src = f"{parsed.scheme}://{parsed.netloc}{img_src}"
                        result['image'] = img_src

        # Only return if we found at least title or description
        if result['title'] or result['description']:
            return result

        return None

    except Exception as e:
        # Silently fail - OG metadata extraction is optional
        return None


def _replace_netloc(url: str, netloc: str) -> str:
    parsed = urlparse(url)
    return urlunparse(parsed._replace(netloc=netloc))


def _build_feed_fallback_urls(feed_url: str) -> list[str]:
    urls = [feed_url]
    if _is_reddit_url(feed_url):
        parsed = urlparse(feed_url)
        for netloc in ["www.reddit.com", "old.reddit.com", "reddit.com"]:
            if parsed.netloc != netloc:
                urls.append(urlunparse(parsed._replace(netloc=netloc)))
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


def parse_feed_with_retry(
    feed_url: str,
    timeout: int = 20,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
    require_entries: bool = False,
):
    """
    Fetch and parse an RSS feed with retries and Reddit fallbacks.

    Returns (feed, used_url, error). If feed is None, error contains the last
    exception or error message.
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/rss+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.5",
    }
    urls = _build_feed_fallback_urls(feed_url)
    last_error = None

    for attempt in range(1, max_retries + 1):
        for url in urls:
            try:
                response = requests.get(url, timeout=timeout, headers=headers)
                if response.status_code == 429:
                    # Rate limited - check headers for wait time
                    reset_seconds = response.headers.get('X-Ratelimit-Reset')
                    retry_after = response.headers.get('Retry-After')

                    if reset_seconds:
                        wait_time = float(reset_seconds) + 1
                    elif retry_after:
                        wait_time = int(retry_after) + 1
                    else:
                        # Fallback: exponential backoff starting at 60s, cap at 900s (15 min)
                        wait_time = min(60 * (2 ** (attempt - 1)), 900)

                    import logging
                    logging.warning(f"Rate limited (429) for {url}, waiting {wait_time:.0f}s (attempt {attempt}/{max_retries})...")
                    print(f"Rate limited (429) for {url}, waiting {wait_time:.0f}s (attempt {attempt}/{max_retries})...")
                    time.sleep(wait_time)
                    last_error = f"HTTP 429 (rate limited)"
                    continue
                if response.status_code != 200:
                    last_error = f"HTTP {response.status_code}"
                    continue
                if not response.text:
                    last_error = "Empty response"
                    continue
                feed = feedparser.parse(response.text)
                if getattr(feed, "bozo", False) and not getattr(feed, "entries", None):
                    last_error = getattr(feed, "bozo_exception", "bozo_error")
                    continue
                if require_entries and (not getattr(feed, "entries", None) or len(feed.entries) == 0):
                    last_error = "No entries"
                    continue
                return feed, url, None
            except Exception as e:
                last_error = e
                continue

        if attempt < max_retries:
            time.sleep(backoff_seconds * attempt)

    # Final fallback: let feedparser handle the request with headers
    for url in urls:
        try:
            feed = feedparser.parse(url, request_headers=headers)
            if getattr(feed, "bozo", False) and not getattr(feed, "entries", None):
                last_error = getattr(feed, "bozo_exception", "bozo_error")
                continue
            if require_entries and (not getattr(feed, "entries", None) or len(feed.entries) == 0):
                last_error = "No entries"
                continue
            return feed, url, last_error
        except Exception as e:
            last_error = e

    return None, None, last_error


def _normalize_reddit_href(href: str) -> str | None:
    """Normalize RSS anchor hrefs to absolute URLs when possible."""
    if not href:
        return None
    href = href.strip()
    if href.startswith("//"):
        return f"https:{href}"
    if href.startswith("/"):
        return f"https://www.reddit.com{href}"
    return href


def _extract_external_link(entry) -> str | None:
    """
    Extract the external (non-Reddit) link from a Reddit RSS entry.

    Reddit RSS entries include a [link] anchor that points to the submission
    target (external URL for link posts, Reddit for self posts). Prefer that
    anchor and only fall back if the markup is non-standard.
    """
    html_sources = []
    content = getattr(entry, "content", None)
    if content:
        for item in content:
            value = getattr(item, "value", None)
            if value:
                html_sources.append(value)

    summary = getattr(entry, "summary", None)
    if summary:
        html_sources.append(summary)

    for html in html_sources:
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            continue

        found_link_anchor = False
        # Prefer the explicit [link] anchor when present
        for anchor in soup.find_all("a", href=True):
            text = anchor.get_text(strip=True).lower()
            if text == "[link]":
                found_link_anchor = True
                href = _normalize_reddit_href(anchor["href"])
                if href and not _is_reddit_url(href):
                    return href

        # If [link] anchor exists but is Reddit-hosted, treat as self post
        if found_link_anchor:
            return None

        # Fallback: first non-Reddit link that isn't the comments link
        for anchor in soup.find_all("a", href=True):
            text = anchor.get_text(strip=True).lower()
            if text == "[comments]":
                continue
            href = _normalize_reddit_href(anchor["href"])
            if href and not _is_reddit_url(href):
                return href

    return None


def extract_image_from_url(url: str, timeout: int = 10) -> str | None:
    """
    Extract image URL from an article or video page.

    For video platforms (YouTube, Vimeo), extracts thumbnail.
    For articles, attempts to extract image URL from:
    1. og:image meta tag (primary source)
    2. First img tag in the article (fallback)

    Args:
        url: Article or video URL to fetch
        timeout: Request timeout in seconds (default: 10)

    Returns:
        Image URL if found, None otherwise
    """
    try:
        # Check if URL domain should be skipped (social media with auth)
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        for skip_domain in SKIP_DOMAINS_FOR_OG_FETCH:
            # Use proper domain matching (not substring) to avoid false positives
            if domain == skip_domain or domain.endswith('.' + skip_domain):
                return None

        # For video platforms, use video metadata to get thumbnail
        if is_video_url(url):
            video_data = extract_video_metadata(url, timeout)
            if video_data and video_data.get('image'):
                return video_data['image']
            return None

        # Fetch the article page
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, timeout=timeout, headers=headers)

        # Skip non-HTML responses
        content_type = response.headers.get('content-type', '')
        if 'text/html' not in content_type.lower():
            return None

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, features="html.parser")

        # Helper to validate image URLs
        def is_valid_image_url(img_url: str) -> bool:
            if not img_url or len(img_url) < 10:
                return False
            # Reject data: URIs (placeholder/lazy-load images)
            if img_url.startswith("data:"):
                return False
            # Reject common placeholder patterns
            if "placeholder" in img_url.lower() or "blank" in img_url.lower():
                return False
            return True

        # Try og:image meta tag first (most reliable)
        og_image = soup.find("meta", attrs={"property": "og:image"})
        if og_image and og_image.get("content"):
            img_url = og_image["content"].split("?")[0]  # Remove query params
            if is_valid_image_url(img_url):
                return img_url

        # Fallback: first img tag with src
        img_tag = soup.find("img", src=True)
        if img_tag:
            img_url = img_tag.get("src", "")
            # Handle relative URLs
            if img_url.startswith("/"):
                img_url = f"{parsed.scheme}://{parsed.netloc}{img_url}"
            img_url = img_url.split("?")[0]
            if is_valid_image_url(img_url):
                return img_url

        return None
    except Exception as e:
        # Silently fail - image extraction is optional
        return None


class News(object):
    def __init__(self, title, summary, link, published, image_url=None):
        """
        This class represents a news article.

        :param title: the title of the article
        :param summary: the summary of the article
        :param link: the link to the article
        :param published: the date the article was published
        :param image_url: the url of the image in the article
        """
        self.title = title
        self.summary = summary
        self.link = link
        self.published = published
        self.image_url = image_url

    def __str__(self):
        """
        String representation of the news article.
        :return: a string representation of the news article
        """
        return f"Title: {self.title}\nSummary: {self.summary}\nLink: {self.link}\nPublished: {self.published}"

    def __repr__(self):
        """
        Representation of the news article.
        :return: the string representation of the news article
        """
        return self.__str__()

    def to_dict(self):
        """
        Convert the news article to a dictionary.

        :return: the dictionary representation of the news article
        """
        return {
            "title": self.title,
            "summary": self.summary,
            "link": self.link,
            "published": self.published,
        }

    def to_json(self):
        """
        Convert the news article to a json string.

        :return: a json string representation of the news article
        """
        return json.dumps(self.to_dict())

    def save(self, name, rss):
        """
        Save the news article to the database.

        :param name: the name of the website
        :param rss: the rss feed of the website
        """
        website_id = (
            session.query(Websites)
            .filter(Websites.name == name, Websites.rss == rss)
            .first()
            .id
        )
        # check if article exists
        if (
            session.query(Articles)
            .filter(Articles.link == self.link, Articles.website_id == website_id)
            .first()
            is None
        ):
            art = Articles(
                title=self.title,
                summary=self.summary,
                website_id=website_id,
                fetched_on=self.published,
                link=self.link,
            )
            session.add(art)
            session.commit()

        # get the article id
        article_id = (
            session.query(Articles)
            .filter(Articles.link == self.link, Articles.website_id == website_id)
            .first()
            .id
        )

        if self.image_url is not None:
            img = Images(url=self.image_url, article_id=article_id)
            session.add(img)
            session.commit()


class NewsFeed(object):
    def __init__(
        self,
        name,
        feed_url,
        url_site=None,
        category=None,
        language=None,
        leaning=None,
        country=None,
        fetch_images_from_url=False,
        fetch_images_timeout=10,
    ):
        """
        This class represents a news feed.

        :param name: the name of the website
        :param feed_url: the rss feed url
        :param url_site: the website url
        :param category: the category of the website
        :param language: the language of the website
        :param leaning: the political leaning of the website
        :param country: the country of the website
        :param fetch_images_from_url: whether to extract images from article URLs
        :param fetch_images_timeout: timeout for image extraction requests
        """
        self.feed_url = feed_url
        self.name = name
        self.url_site = url_site
        self.category = category
        self.language = language
        self.leaning = leaning
        self.country = country
        self.fetch_images_from_url = fetch_images_from_url
        self.fetch_images_timeout = fetch_images_timeout
        self.news = []

    def read_feed(self):
        """
        Read the feed and store the news articles.
        """
        today = datetime.datetime.now()
        timestamp = int(today.strftime("%Y%m%d"))

        # Track statistics
        stats = {
            "total_entries": 0,
            "processed": 0,
            "errors": 0,
            "images_added": 0,
            "reddit_only_skipped": 0,
        }

        try:
            # get website id
            website = session.query(Websites).filter(Websites.name == self.name, Websites.rss == self.feed_url).first()
            if not website:
                print(f"Error: Website {self.name} with RSS {self.feed_url} not found in database")
                return

            website_id = website.id

            print(f"Processing feed: {self.name} ({self.feed_url})")

            # Fetch the feed with retry + fallback
            feed, used_url, error = parse_feed_with_retry(
                self.feed_url,
                timeout=20,
                max_retries=3,
                backoff_seconds=2.0,
                require_entries=False,
            )
            if not feed:
                print(f"Error fetching feed {self.feed_url}: {error}")
                return
            if used_url and used_url != self.feed_url:
                print(f"Using fallback feed URL: {used_url}")
            if hasattr(feed, "bozo") and feed.bozo:
                print(f"Warning: Feed might be malformed: {getattr(feed, 'bozo_exception', 'Unknown')}")

            # Process all entries in the feed
            stats["total_entries"] = len(feed.entries)
            print(f"Found {stats['total_entries']} entries in the feed")

            for entry in feed.entries:
                try:
                    entry_link = entry.link
                    entry_title = entry.title
                    entry_summary = getattr(entry, 'summary', '')
                    img_url = None
                    content_type = 'article'  # Default content type

                    # Check if the entry links to Reddit (submission page)
                    if _is_reddit_url(entry_link):
                        external_link = _extract_external_link(entry)
                        if external_link:
                            entry_link = external_link
                        else:
                            stats["reddit_only_skipped"] += 1
                            continue

                    # For external URLs (not Reddit), fetch metadata
                    # Priority: Video metadata > OpenGraph metadata (includes first img fallback)
                    if is_external_url(entry_link):
                        metadata = None

                        # First, check if it's a video URL (YouTube, Vimeo, etc.)
                        if is_video_url(entry_link):
                            metadata = extract_video_metadata(entry_link, timeout=self.fetch_images_timeout)
                            if metadata:
                                content_type = 'video'
                                print(f"  [VIDEO] Fetched metadata for: {entry_link[:60]}...")

                        # If not a video or video fetch failed, try OG metadata
                        if not metadata:
                            metadata = extract_og_metadata(entry_link, timeout=self.fetch_images_timeout)
                            if metadata:
                                print(f"  [OG] Fetched metadata for: {entry_link[:60]}...")

                        if metadata:
                            # Use fetched metadata instead of RSS entry data
                            if metadata.get('title'):
                                entry_title = metadata['title']
                            if metadata.get('description'):
                                entry_summary = metadata['description']
                            if metadata.get('image'):
                                img_url = metadata['image']
                        else:
                            # Metadata fetch failed, clean up Reddit formatting from RSS data
                            entry_title = clean_reddit_formatting(entry_title)
                            entry_summary = clean_reddit_formatting(entry_summary)

                    # Create news article with potentially enhanced metadata
                    art = News(entry_title, entry_summary, entry_link, timestamp)
                    art.save(name=self.name, rss=self.feed_url)

                    # get article id to save image
                    article_record = (
                        session.query(Articles)
                        .filter(
                            Articles.link == entry_link,
                            Articles.website_id == website_id,
                        )
                        .first()
                    )
                    if not article_record:
                        print(f"Warning: Article {entry_title} not found in database after save")
                        continue

                    article_id = article_record.id

                    # Save image if we found one from OG/video metadata extraction
                    if img_url is not None:
                        # check if image is already in the database
                        if session.query(Images).filter(Images.url == img_url).first() is None:
                            img_record = Images(url=img_url, article_id=article_id)
                            session.add(img_record)
                            session.commit()
                            stats["images_added"] += 1

                    self.news.append(art)
                    stats["processed"] += 1
                except Exception as e:
                    stats["errors"] += 1
                    print(f"Error processing article '{entry.title if hasattr(entry, 'title') else 'Unknown'}': {str(e)}")

            # If no new articles were processed, load existing ones from database
            if stats["processed"] == 0:
                # Get recent articles from this website (not just today)
                articles = session.query(Articles).filter(Articles.website_id == website_id).order_by(Articles.id.desc()).limit(10).all()

                if articles:
                    print(f"Loading {len(articles)} existing articles from database")
                    for art in articles:
                        self.news.append(News(art.title, art.summary, art.link, art.fetched_on))

            # Print summary
            print(f"Feed processing summary for {self.name}:")
            print(f"  - Total entries found: {stats['total_entries']}")
            print(f"  - Successfully processed: {stats['processed']}")
            print(f"  - Errors encountered: {stats['errors']}")
            print(f"  - Reddit-only skipped: {stats['reddit_only_skipped']}")
            print(f"  - Images added: {stats['images_added']}")
            print(f"  - Total articles in memory: {len(self.news)}")

            # Record successful refresh time for this website.
            try:
                website.last_fetched = timestamp
                session.commit()
            except Exception as e:
                print(f"Error updating website last_fetched for {self.name}: {str(e)}")
                session.rollback()

        except Exception as e:
            print(f"Critical error processing feed {self.name}: {str(e)}")
            import traceback
            traceback.print_exc()

    def __extract_image_url(self, art):
        """
        Extract the image url from the article.

        :param art:
        :return: img url
        """
        if "media_content" in art:
            image = art.media_content[0]["url"].split("?")[0]
            return image
        return None

    def get_random_news(self):
        """
        Get a random news article from the feed.

        :return: a random news article or error message if none available
        """
        if len(self.news) == 0:
            # Try to load from database first
            self.get_all_news()

        # Check again after potential database load
        if len(self.news) == 0:
            return "No news available"

        return np.random.choice(self.news)

    def get_news(self):
        """
        Get all the news articles from the feed.
        :return: a list of news articles
        """
        return self.news

    def get_all_news(self):
        """
        Get all the news articles from the feed.
        If no news is available in memory, try to fetch from database.
        :return: a list of news articles
        """
        if len(self.news) == 0:
            # Try to get articles from database
            try:
                # Get website id
                website = session.query(Websites).filter(Websites.name == self.name, Websites.rss == self.feed_url).first()
                if website:
                    website_id = website.id
                    # Get articles from this website
                    articles = session.query(Articles).filter(Articles.website_id == website_id).order_by(Articles.id.desc()).limit(10).all()

                    if articles:
                        for art in articles:
                            self.news.append(News(art.title, art.summary, art.link, art.fetched_on))
            except Exception as e:
                print(f"Error fetching articles from database: {str(e)}")

        return self.news

    def to_dict(self):
        """
        Convert the news feed to a dictionary.

        :return: the dictionary representation of the news feed
        """
        return {
            "name": self.name,
            "feed_url": self.feed_url,
            "url_site": self.url_site,
            "category": self.category,
            "language": self.language,
            "leaning": self.leaning,
            "country": self.country,
            "news": [n.to_dict() for n in self.news],
        }

    def to_json(self):
        """
        Convert the news feed to a json string.

        :return: a json string representation of the news feed
        """
        return json.dumps(self.to_dict())


class Feeds(object):
    def __init__(self):
        """
        This class represents a collection of news feeds.
        """
        self.feeds = []

    @staticmethod
    def __not_in_db(name: str, url: str) -> object:
        """
        Check if the feed is not in the database.

        :param name: the name of the website
        :param url: the rss feed url
        :return: whether the feed is not in the database
        """
        res = (
            session.query(Websites)
            .filter(Websites.name == name, Websites.rss == url)
            .first()
        )
        return res is None

    def add_feed(
        self,
        name,
        url_site=None,
        url_feed=None,
        category=None,
        language=None,
        leaning=None,
        country=None,
        fetch_images_from_url=False,
        fetch_images_timeout=10,
    ):
        """
        Add a feed to the collection.

        :param name: the name of the website
        :param url_site: the website url
        :param url_feed: the rss feed url
        :param category: the category of the website
        :param language: the language of the website
        :param leaning: the political leaning of the website
        :param country: the country of the website
        :param fetch_images_from_url: whether to extract images from article URLs
        :param fetch_images_timeout: timeout for image extraction requests
        """
        today = datetime.datetime.now()
        timestamp = int(today.strftime("%Y%m%d"))

        if url_feed is not None:
            if self.__not_in_db(name, url_feed):
                if self.__validate_feed(url_feed):
                    print(f"Adding feed: {name} ({url_feed})")
                    self.feeds.append(
                        NewsFeed(
                            name,
                            url_feed,
                            url_site,
                            category,
                            language,
                            leaning,
                            country,
                            fetch_images_from_url=fetch_images_from_url,
                            fetch_images_timeout=fetch_images_timeout,
                        )
                    )

                    # check if website exists
                    web = Websites(
                        name=name,
                        rss=url_feed,
                        country=country or "Unknown",
                        language=language or "en",
                        leaning=leaning or "center",
                        category=category or "general",
                        last_fetched=timestamp,
                        fetch_images_from_url=fetch_images_from_url,
                        fetch_images_timeout=fetch_images_timeout,
                    )
                    session.add(web)
                    session.commit()
                else:
                    print(f"Feed validation failed: {name} ({url_feed})")
                    try:
                        website = session.query(Websites).filter(Websites.name == name, Websites.rss == url_feed).first()
                        if website:
                            last_fetched = website.last_fetched
                            if timestamp > last_fetched:
                                session.query(Websites).filter(
                                    Websites.name == name, Websites.rss == url_feed
                                ).update({"last_fetched": timestamp})
                                session.commit()
                    except Exception as e:
                        print(f"Error updating last_fetched time: {str(e)}")
            else:
                print(f"Feed already in database: {name} ({url_feed})")
                try:
                    website = session.query(Websites).filter(Websites.name == name, Websites.rss == url_feed).first()
                    if website:
                        # Add to current feed collection even if it's already in the database
                        # Use settings from database or provided values
                        self.feeds.append(
                            NewsFeed(
                                name,
                                url_feed,
                                url_site,
                                website.category,
                                website.language,
                                website.leaning,
                                website.country,
                                fetch_images_from_url=getattr(website, 'fetch_images_from_url', False) or fetch_images_from_url,
                                fetch_images_timeout=getattr(website, 'fetch_images_timeout', 10) or fetch_images_timeout,
                            )
                        )
                except Exception as e:
                    print(f"Error retrieving website from database: {str(e)}")

        elif url_site is not None:
            print(f"Extracting RSS feeds from site: {url_site}")
            fex = FeedLinkExtractor(url_site)
            fex.extract_rss_url()
            rss_urls = fex.get_rss_urls()

            if not rss_urls:
                print(f"No RSS feeds found on site: {url_site}")

            for rss in rss_urls:
                if self.__not_in_db(name, rss):  # Fixed bug: was using url_feed here
                    if self.__validate_feed(rss):
                        print(f"Adding extracted feed: {name} ({rss})")
                        self.feeds.append(
                            NewsFeed(
                                name,
                                rss,
                                url_site,
                                category,
                                language,
                                leaning,
                                country,
                                fetch_images_from_url=fetch_images_from_url,
                                fetch_images_timeout=fetch_images_timeout,
                            )
                        )

                        web = Websites(
                            name=name,
                            rss=rss,  # Fixed bug: was using url_feed here
                            country=country or "Unknown",
                            language=language or "en",
                            leaning=leaning or "center",
                            category=category or "general",
                            last_fetched=timestamp,
                            fetch_images_from_url=fetch_images_from_url,
                            fetch_images_timeout=fetch_images_timeout,
                        )
                        session.add(web)
                        session.commit()
                    else:
                        print(f"Extracted feed validation failed: {name} ({rss})")
                else:
                    print(f"Extracted feed already in database: {name} ({rss})")
        else:
            print("Please provide a feed url or a site url")

    def get_feeds(self):
        """
        Get all the feeds in the collection.

        :return: a list of feeds
        """
        return self.feeds

    @staticmethod
    def __validate_feed(url):
        """
        Validate the rss feed.

        :param url: the rss feed url
        :return: whether the feed is valid
        """
        try:
            feed, used_url, error = parse_feed_with_retry(
                url,
                timeout=20,
                max_retries=3,
                backoff_seconds=2.0,
                require_entries=False,
            )
            if not feed:
                print(f"Error validating feed {url}: {error}")
                return False

            if used_url and used_url != url:
                print(f"Validation used fallback URL: {used_url}")

            # Check if the feed was successfully parsed
            if hasattr(feed, 'bozo') and feed.bozo:
                print(
                    f"Warning: Feed {url} might be malformed. Error: "
                    f"{feed.bozo_exception if hasattr(feed, 'bozo_exception') else 'Unknown'}"
                )

            # Check if the feed has entries
            if not hasattr(feed, 'entries') or len(feed.entries) == 0:
                print(f"Warning: Feed {url} has no entries")
                # Still return True as it might be a valid feed with no current entries
                return True

            return True
        except Exception as e:
            print(f"Error validating feed {url}: {str(e)}")
            return False


class FeedLinkExtractor(object):
    def __init__(self, url):
        """
        This class extracts rss feed urls from a website.

        :param url: the website url
        """
        self.url = url
        self.rss_urls = []

    def extract_rss_url(self):
        """
        Extract (or at least tries to) the rss feed urls from the website.
        """
        try:
            page = requests.get(self.url, timeout=5).text
            soup = BeautifulSoup(page, features="html.parser")

            for e in soup.select(
                'a[href*="rss"],a[href*="/feed"],a:-soup-contains-own("RSS")'
            ):
                if e.get("href").startswith("/"):
                    url = self.url.strip("/") + e.get("href")
                else:
                    url = e.get("href")

                base_url = re.search(
                    "^(?:https?:\/\/)?(?:[^@\/\n]+@)?(?:www\.)?([^:\/\n]+)", url
                ).group(0)
                r = requests.get(url)
                soup = BeautifulSoup(r.text, features="html.parser")

                for e1 in soup.select(
                    '[type="application/rss+xml"],a[href*=".rss"],a[href$="feed"]'
                ):
                    if e1.get("href").startswith("/"):
                        rss = base_url + e1.get("href")
                    else:
                        rss = e1.get("href")
                    if "xml" in requests.get(rss).headers.get("content-type"):
                        self.rss_urls.append(rss)
        except:
            pass

    def get_rss_urls(self):
        """
        Get the extracted rss feed urls.

        :return: the extracted rss feed urls
        """
        return self.rss_urls

    def to_dict(self):
        """
        Convert the rss feed urls to a dictionary.

        :return: the dictionary representation of the rss feed urls
        """
        return {"rss_urls": self.rss_urls}

    def to_json(self):
        """
        Convert the rss feed urls to a json string.

        :return: the json string representation of the rss feed urls
        """
        return json.dumps(self.to_dict())
