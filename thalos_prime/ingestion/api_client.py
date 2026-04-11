from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Optional

import httpx

from thalos_prime.config import Settings
from thalos_prime.ingestion.models import IngestionResult
from thalos_prime.storage.db import Feed, RawDocument, get_session, init_db

logger = logging.getLogger(__name__)


class FeedClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _store_raw(self, text: str, url: str, source_type: str = "feed") -> IngestionResult:
        init_db(self.settings.database_url)
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with get_session(self.settings.database_url) as session:
            from sqlalchemy import select

            existing = session.execute(
                select(RawDocument).where(RawDocument.content_hash == content_hash)
            ).scalar_one_or_none()
            if existing:
                return IngestionResult(
                    doc_id=existing.id,
                    url=url,
                    status="duplicate",
                    word_count=len(text.split()),
                    message="Duplicate content hash",
                )
            doc = RawDocument(
                url=url,
                content_hash=content_hash,
                raw_text=text,
                source_type=source_type,
                robots_allowed=True,
                status="raw",
            )
            session.add(doc)
            session.flush()
            doc_id = doc.id
        return IngestionResult(
            doc_id=doc_id,
            url=url,
            status="ingested",
            word_count=len(text.split()),
        )

    def fetch_rss(self, url: str) -> list[IngestionResult]:
        try:
            import feedparser
            feed = feedparser.parse(url)
        except Exception as e:
            logger.error("RSS parse failed for %s: %s", url, e)
            return []

        results = []
        for entry in feed.entries:
            text_parts = []
            if hasattr(entry, "title") and entry.title:
                text_parts.append(entry.title)
            if hasattr(entry, "summary") and entry.summary:
                from bs4 import BeautifulSoup
                text_parts.append(BeautifulSoup(entry.summary, "lxml").get_text())
            if hasattr(entry, "content") and entry.content:
                from bs4 import BeautifulSoup
                for c in entry.content:
                    text_parts.append(BeautifulSoup(c.get("value", ""), "lxml").get_text())
            text = "\n".join(text_parts).strip()
            if not text:
                continue
            link = getattr(entry, "link", url)
            result = self._store_raw(text, link, "rss")
            results.append(result)
        return results

    def fetch_json_feed(self, url: str) -> list[IngestionResult]:
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(url, headers={"User-Agent": "ThalosBot/0.1"})
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.error("JSON feed fetch failed for %s: %s", url, e)
            return []

        results = []
        items = data if isinstance(data, list) else data.get("items", [])
        for item in items:
            if not isinstance(item, dict):
                continue
            text_parts = []
            for key in ("title", "summary", "content", "text", "body", "description"):
                if item.get(key):
                    text_parts.append(str(item[key]))
            text = "\n".join(text_parts).strip()
            if not text:
                continue
            link = item.get("url") or item.get("link") or url
            result = self._store_raw(text, link, "json_feed")
            results.append(result)
        return results

    def register_feed(self, url: str, feed_type: str = "rss") -> str:
        init_db(self.settings.database_url)
        with get_session(self.settings.database_url) as session:
            from sqlalchemy import select

            existing = session.execute(
                select(Feed).where(Feed.url == url)
            ).scalar_one_or_none()
            if existing:
                return existing.id
            feed = Feed(url=url, feed_type=feed_type, active=True)
            session.add(feed)
            session.flush()
            return feed.id
