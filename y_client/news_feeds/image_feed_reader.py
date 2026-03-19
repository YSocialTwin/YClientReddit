import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from sqlalchemy import text

IMAGE_STORAGE_DIR = "uploads/forum_feed_images"


def generate_filename(url: str, image_id: int) -> str:
    parsed = urlparse(url or "")
    ext = os.path.splitext(parsed.path)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
        ext = ".jpg"
    digest = hashlib.sha1((url or str(image_id)).encode("utf-8")).hexdigest()[:12]
    return f"image_{image_id}_{digest}{ext}"


def download_image_checked(url: str, filepath: str, min_bytes: int = 8 * 1024):
    headers = {"User-Agent": "YSocial/1.0 Image Feed Reader"}
    try:
        response = requests.get(url, headers=headers, timeout=25, stream=True)
        response.raise_for_status()
        content_type = (response.headers.get("Content-Type") or "").lower()
        if content_type and not content_type.startswith("image/"):
            return False, "not_image"

        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with open(target, "wb") as out:
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                out.write(chunk)
                written += len(chunk)

        if written < min_bytes:
            try:
                target.unlink(missing_ok=True)
            except Exception:
                pass
            return False, "too_small"
        return True, None
    except Exception as exc:
        return False, str(exc)


def extract_high_res_url(source_url: str, fallback_url: str | None = None):
    headers = {"User-Agent": "YSocial/1.0 Image Feed Reader"}
    try:
        response = requests.get(source_url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for selector in [
            ("meta", {"property": "og:image"}),
            ("meta", {"name": "twitter:image"}),
        ]:
            tag = soup.find(*selector)
            if tag and tag.get("content"):
                return tag.get("content")
    except Exception:
        pass
    return fallback_url or source_url


def annotate_pending_images(annotator, batch_size=50, engine=None):
    if engine is None:
        return 0
    updated = 0
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                "SELECT id, title FROM image_posts WHERE description IS NULL OR TRIM(description) = '' LIMIT :limit"
            ),
            {"limit": int(batch_size)},
        ).fetchall()
        for row in rows:
            description = (row[1] or "").strip()
            if not description:
                description = "Image shared from configured feed"
            conn.execute(
                text("UPDATE image_posts SET description = :description WHERE id = :id"),
                {"description": description[:400], "id": row[0]},
            )
            updated += 1
    return updated
