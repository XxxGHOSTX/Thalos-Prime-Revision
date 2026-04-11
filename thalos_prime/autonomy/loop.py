from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from thalos_prime.config import Settings

if TYPE_CHECKING:
    from thalos_prime.core import ThalosCore

logger = logging.getLogger(__name__)


class AutonomyLoop:
    def __init__(self, settings: Settings, core: "ThalosCore"):
        self.settings = settings
        self.core = core
        self._scheduler = None

    def start(self) -> None:
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger
            from apscheduler.triggers.interval import IntervalTrigger
        except ImportError:
            logger.error("APScheduler not installed. Autonomy loop unavailable.")
            return

        self._scheduler = BackgroundScheduler(daemon=True)

        cron_parts = self.settings.autonomy_cron.split()
        if len(cron_parts) == 5:
            minute, hour, day, month, dow = cron_parts
            self._scheduler.add_job(
                self.re_ingest_feeds,
                CronTrigger(minute=minute, hour=hour, day=day, month=month, day_of_week=dow),
                id="re_ingest_feeds",
                replace_existing=True,
            )
        else:
            self._scheduler.add_job(
                self.re_ingest_feeds,
                "interval",
                hours=1,
                id="re_ingest_feeds",
                replace_existing=True,
            )

        self._scheduler.add_job(
            self.re_verify_stale,
            "interval",
            hours=24,
            id="re_verify_stale",
            replace_existing=True,
        )
        self._scheduler.add_job(
            self.re_check_contradictions,
            "interval",
            hours=6,
            id="re_check_contradictions",
            replace_existing=True,
        )

        self._scheduler.start()
        logger.info("Autonomy loop started")

    def stop(self) -> None:
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("Autonomy loop stopped")

    def re_ingest_feeds(self) -> None:
        from sqlalchemy import select
        from thalos_prime.storage.db import Feed, get_session
        from datetime import datetime, timezone

        logger.info("Autonomy: re-ingesting active feeds")
        with get_session(self.settings.database_url) as session:
            feeds = session.execute(
                select(Feed).where(Feed.active == True)
            ).scalars().all()

        for feed in feeds:
            try:
                results = self.core.ingest_feed(feed.url)
                logger.info("Feed %s: ingested %d items", feed.url, len(results))
                with get_session(self.settings.database_url) as session:
                    from sqlalchemy import select as sel
                    f = session.execute(sel(Feed).where(Feed.id == feed.id)).scalar_one_or_none()
                    if f:
                        f.last_fetched = datetime.now(timezone.utc)
            except Exception as e:
                logger.error("Feed re-ingest failed for %s: %s", feed.url, e)

    def re_verify_stale(self) -> None:
        from sqlalchemy import select
        from thalos_prime.storage.db import Claim, get_session

        logger.info("Autonomy: re-verifying stale claims")
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.settings.autonomy_max_age_days)

        with get_session(self.settings.database_url) as session:
            stale_claims = session.execute(
                select(Claim).where(
                    Claim.updated_at < cutoff,
                    Claim.status.in_(["draft", "candidate", "verified"]),
                )
            ).scalars().all()

            from thalos_prime.verification.promoter import Promoter
            promoter = Promoter(self.settings)
            for claim in stale_claims:
                try:
                    promoter.evaluate(claim, session)
                except Exception as e:
                    logger.warning("Re-verify failed for %s: %s", claim.id, e)

    def re_check_contradictions(self) -> None:
        from sqlalchemy import select
        from thalos_prime.storage.db import Claim, get_session

        logger.info("Autonomy: checking new contradictions")
        cutoff = datetime.now(timezone.utc) - timedelta(hours=6)

        with get_session(self.settings.database_url) as session:
            new_claims = session.execute(
                select(Claim).where(Claim.created_at >= cutoff)
            ).scalars().all()

            if not new_claims:
                return

            from thalos_prime.contradiction.detector import ContradictionDetector
            detector = ContradictionDetector(self.settings, self.core.vector_store)
            detector.detect(new_claims, session)
