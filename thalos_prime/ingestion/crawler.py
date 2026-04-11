from __future__ import annotations

import hashlib
import logging
import urllib.robotparser
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from thalos_prime.config import Settings
from thalos_prime.ingestion.models import IngestionResult
from thalos_prime.storage.db import RawDocument, get_session, init_db

logger = logging.getLogger(__name__)


class WebCrawler:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}

    def _check_robots(self, url: str) -> bool:
        if not self.settings.respect_robots:
            return True
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        robots_url = urljoin(base, "/robots.txt")
        if base not in self._robots_cache:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(robots_url)
            try:
                rp.read()
            except Exception:
                rp = None
            self._robots_cache[base] = rp
        rp = self._robots_cache[base]
        if rp is None:
            return True
        return rp.can_fetch("*", url)

    def _extract_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [ln.strip() for ln in text.splitlines()]
        lines = [ln for ln in lines if ln]
        return "\n".join(lines)

    def fetch(self, url: str) -> IngestionResult:
        init_db(self.settings.database_url)
        robots_ok = self._check_robots(url)
        if not robots_ok:
            logger.warning("Robots.txt disallows: %s", url)
            return IngestionResult(doc_id=None, url=url, status="error", message="robots_disallowed")

        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                response = client.get(url, headers={"User-Agent": "ThalosBot/0.1"})
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "html" in content_type:
                    raw_text = self._extract_text(response.text)
                else:
                    raw_text = response.text
        except Exception as e:
            logger.error("Fetch failed for %s: %s", url, e)
            return IngestionResult(doc_id=None, url=url, status="error", message=str(e))

        content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

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
                    word_count=len(raw_text.split()),
                    message="Duplicate content hash",
                )

            doc = RawDocument(
                url=url,
                content_hash=content_hash,
                raw_text=raw_text,
                source_type="web",
                robots_allowed=robots_ok,
                status="raw",
            )
            session.add(doc)
            session.flush()
            doc_id = doc.id

        return IngestionResult(
            doc_id=doc_id,
            url=url,
            status="ingested",
            word_count=len(raw_text.split()),
        )
