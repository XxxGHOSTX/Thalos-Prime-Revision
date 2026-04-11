from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from thalos_prime.config import Settings, get_settings
from thalos_prime.storage.db import (
    Claim,
    NormalizedDocument,
    RawDocument,
    VerificationRecord,
    get_session,
    init_db,
)

logger = logging.getLogger(__name__)


class ThalosCore:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._crawler = None
        self._file_loader = None
        self._feed_client = None
        self._vector_store = None
        self._claim_extractor = None
        self._evidence_linker = None
        self._contradiction_detector = None
        self._scorer = None
        self._promoter = None
        self._query_engine_cache = None
        self._answerer = None
        self._autonomy_loop = None
        self._initialized = False
        self._init_components()

    def _init_components(self) -> None:
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.settings.vector_store_path.mkdir(parents=True, exist_ok=True)
        init_db(self.settings.database_url)
        self._initialized = True
        logging.basicConfig(level=getattr(logging, self.settings.log_level, logging.INFO))

    @property
    def vector_store(self):
        if self._vector_store is None:
            from thalos_prime.storage.vector_store import VectorStore
            self._vector_store = VectorStore(
                self.settings.vector_store_path, self.settings.embedding_model
            )
        return self._vector_store

    @property
    def crawler(self):
        if self._crawler is None:
            from thalos_prime.ingestion.crawler import WebCrawler
            self._crawler = WebCrawler(self.settings)
        return self._crawler

    @property
    def file_loader(self):
        if self._file_loader is None:
            from thalos_prime.ingestion.file_loader import FileLoader
            self._file_loader = FileLoader(self.settings)
        return self._file_loader

    @property
    def feed_client(self):
        if self._feed_client is None:
            from thalos_prime.ingestion.api_client import FeedClient
            self._feed_client = FeedClient(self.settings)
        return self._feed_client

    @property
    def claim_extractor(self):
        if self._claim_extractor is None:
            from thalos_prime.extraction.claim_extractor import ClaimExtractor
            self._claim_extractor = ClaimExtractor(self.settings)
        return self._claim_extractor

    @property
    def evidence_linker(self):
        if self._evidence_linker is None:
            from thalos_prime.evidence.linker import EvidenceLinker
            self._evidence_linker = EvidenceLinker(self.settings, self.vector_store)
        return self._evidence_linker

    @property
    def contradiction_detector(self):
        if self._contradiction_detector is None:
            from thalos_prime.contradiction.detector import ContradictionDetector
            self._contradiction_detector = ContradictionDetector(self.settings, self.vector_store)
        return self._contradiction_detector

    @property
    def scorer(self):
        if self._scorer is None:
            from thalos_prime.verification.scorer import VerificationScorer
            self._scorer = VerificationScorer(self.settings)
        return self._scorer

    @property
    def promoter(self):
        if self._promoter is None:
            from thalos_prime.verification.promoter import Promoter
            self._promoter = Promoter(self.settings)
        return self._promoter

    @property
    def answerer(self):
        if self._answerer is None:
            from thalos_prime.query.answerer import Answerer
            self._answerer = Answerer(self.settings)
        return self._answerer

    def _run_normalization(self, raw_doc: RawDocument) -> NormalizedDocument:
        from thalos_prime.normalization.cleaner import clean
        from thalos_prime.normalization.lang_detect import detect_language
        from thalos_prime.normalization.translator import Translator

        cleaned = clean(raw_doc.raw_text)
        lang = detect_language(cleaned)

        translated_text = None
        if lang != "en":
            translator = Translator(self.settings)
            translated_text = translator.translate(cleaned, lang, "en")

        norm_doc = NormalizedDocument(
            raw_doc_id=raw_doc.id,
            language=lang,
            original_text=cleaned,
            translated_text=translated_text,
            word_count=len(cleaned.split()),
        )
        return norm_doc

    def _run_extraction(self, norm_doc: NormalizedDocument):
        return self.claim_extractor.extract(norm_doc)

    def _run_evidence_linking(self, norm_doc: NormalizedDocument, claims: list[Claim]) -> list:
        from sqlalchemy import select

        with get_session(self.settings.database_url) as session:
            all_norm_docs = session.execute(select(NormalizedDocument)).scalars().all()
            corpus: list[tuple[str, str]] = []
            for nd in all_norm_docs:
                text = nd.translated_text or nd.original_text
                if text:
                    sentences = text.split(".")
                    for s in sentences:
                        s = s.strip()
                        if len(s.split()) >= 5:
                            doc = session.execute(
                                select(RawDocument).where(RawDocument.id == nd.raw_doc_id)
                            ).scalar_one_or_none()
                            doc_id = doc.id if doc else "unknown"
                            corpus.append((doc_id, s))

            results = []
            for claim in claims:
                result = self.evidence_linker.link(claim, corpus)
                results.append(result)
        return results

    def _run_evidence_linking_by_id(self, norm_doc_id: str, claims_data: list[tuple]) -> list:
        """Link evidence for claims using pre-fetched claim data tuples."""
        from sqlalchemy import select

        corpus: list[tuple[str, str]] = []
        with get_session(self.settings.database_url) as session:
            all_norm_docs = session.execute(select(NormalizedDocument)).scalars().all()
            for nd in all_norm_docs:
                text = nd.translated_text or nd.original_text
                raw_doc_id_local = nd.raw_doc_id
                if text:
                    sentences = text.split(".")
                    for s in sentences:
                        s = s.strip()
                        if len(s.split()) >= 5:
                            corpus.append((raw_doc_id_local or "unknown", s))

        results = []
        for claim_id, claim_text, coord, lang, status in claims_data:
            claim_obj = Claim()
            claim_obj.id = claim_id
            claim_obj.text = claim_text
            claim_obj.coordinate = coord
            claim_obj.language = lang
            claim_obj.status = status
            result = self.evidence_linker.link(claim_obj, corpus)
            results.append(result)
        return results

    def _run_contradiction_detection_by_id(self, claims_data: list[tuple]):
        """Detect contradictions using pre-fetched claim data tuples."""
        from thalos_prime.contradiction.models import ContradictionResult

        if len(claims_data) < 2:
            return ContradictionResult()

        claim_objects = []
        for claim_id, claim_text, coord, lang, status in claims_data:
            c = Claim()
            c.id = claim_id
            c.text = claim_text
            c.coordinate = coord
            c.language = lang
            c.status = status
            claim_objects.append(c)

        with get_session(self.settings.database_url) as session:
            return self.contradiction_detector.detect(claim_objects, session)

    def _run_contradiction_detection(self, claims: list[Claim]):
        if len(claims) < 2:
            from thalos_prime.contradiction.models import ContradictionResult
            return ContradictionResult()
        with get_session(self.settings.database_url) as session:
            return self.contradiction_detector.detect(claims, session)

    def _run_verification(self) -> list:
        results = []
        with get_session(self.settings.database_url) as session:
            promotions = self.promoter.promote_all(session)
            results.extend(promotions)
        return results

    def ingest_url(self, url: str):
        from thalos_prime.ingestion.models import IngestionResult
        result = self.crawler.fetch(url)
        if result.status == "ingested" and result.doc_id:
            self.run_pipeline(result.doc_id)
        return result

    def ingest_file(self, path: str):
        result = self.file_loader.load(Path(path))
        if result.status == "ingested" and result.doc_id:
            self.run_pipeline(result.doc_id)
        return result

    def ingest_feed(self, url: str) -> list:
        results = self.feed_client.fetch_rss(url)
        if not results:
            results = self.feed_client.fetch_json_feed(url)
        for result in results:
            if result.status == "ingested" and result.doc_id:
                self.run_pipeline(result.doc_id)
        return results

    def run_pipeline(self, doc_id: str) -> dict:
        from sqlalchemy import select

        with get_session(self.settings.database_url) as session:
            raw_doc = session.execute(
                select(RawDocument).where(RawDocument.id == doc_id)
            ).scalar_one_or_none()

            if raw_doc is None:
                return {"error": "Document not found", "doc_id": doc_id}

            existing_norm = session.execute(
                select(NormalizedDocument).where(NormalizedDocument.raw_doc_id == doc_id)
            ).scalar_one_or_none()

            if existing_norm is None:
                norm_doc = self._run_normalization(raw_doc)
                session.add(norm_doc)
                session.flush()
                norm_doc_id = norm_doc.id
                lang = norm_doc.language
            else:
                norm_doc_id = existing_norm.id
                lang = existing_norm.language

        with get_session(self.settings.database_url) as session:
            norm_doc = session.execute(
                select(NormalizedDocument).where(NormalizedDocument.id == norm_doc_id)
            ).scalar_one_or_none()

            extraction_result = self._run_extraction(norm_doc)

        norm_original_text = ""
        claims_list = []
        with get_session(self.settings.database_url) as session:
            norm_doc = session.execute(
                select(NormalizedDocument).where(NormalizedDocument.id == norm_doc_id)
            ).scalar_one_or_none()
            if norm_doc:
                norm_original_text = norm_doc.original_text or ""
            claims = session.execute(
                select(Claim).where(Claim.norm_doc_id == norm_doc_id)
            ).scalars().all()
            claims_list = list(claims)
            # Eagerly load claim ids/texts before session closes
            claims_data = [(c.id, c.text, c.coordinate, c.language, c.status) for c in claims_list]

        self.vector_store.add(doc_id, norm_original_text[:10000])

        evidence_results = self._run_evidence_linking_by_id(norm_doc_id, claims_data)
        contradiction_result = self._run_contradiction_detection_by_id(claims_data)
        promotion_results = self._run_verification()

        return {
            "doc_id": doc_id,
            "norm_doc_id": norm_doc_id,
            "language": lang,
            "claims_extracted": extraction_result.claims_extracted,
            "claims_new": extraction_result.claims_new,
            "claims_duplicate": extraction_result.claims_duplicate,
            "evidence_links": sum(r.links_new for r in evidence_results),
            "contradictions_found": contradiction_result.pairs_found,
            "promotions": len(promotion_results),
        }

    def query(self, text: str, top_k: int = 10) -> dict:
        from sqlalchemy import select

        with get_session(self.settings.database_url) as session:
            from thalos_prime.query.engine import QueryEngine
            engine = QueryEngine(self.settings, self.vector_store, session)
            results = engine.search(text, top_k=top_k)

        return self.answerer.synthesize(text, results)

    def get_claim(self, coordinate: str) -> Optional[dict]:
        from sqlalchemy import select

        with get_session(self.settings.database_url) as session:
            claim = session.execute(
                select(Claim).where(Claim.coordinate == coordinate)
            ).scalar_one_or_none()

            if claim is None:
                return None

            record = session.execute(
                select(VerificationRecord).where(VerificationRecord.claim_id == claim.id)
            ).scalar_one_or_none()

            from thalos_prime.storage.db import EvidenceLink
            links = session.execute(
                select(EvidenceLink).where(EvidenceLink.claim_id == claim.id)
            ).scalars().all()

            return {
                "id": claim.id,
                "coordinate": claim.coordinate,
                "text": claim.text,
                "status": claim.status,
                "language": claim.language,
                "entities": json.loads(claim.entities_json) if claim.entities_json else [],
                "created_at": claim.created_at.isoformat() if claim.created_at else None,
                "updated_at": claim.updated_at.isoformat() if claim.updated_at else None,
                "verification": {
                    "score": record.composite_score if record else 0.0,
                    "signals": json.loads(record.signals_json) if (record and record.signals_json) else {},
                    "promoted_to": record.promoted_to if record else None,
                } if record else None,
                "evidence_count": len(links),
            }

    def retract_claim(self, coordinate: str) -> bool:
        from sqlalchemy import select

        with get_session(self.settings.database_url) as session:
            claim = session.execute(
                select(Claim).where(Claim.coordinate == coordinate)
            ).scalar_one_or_none()

            if claim is None:
                return False

            claim.status = "retracted"
            claim.updated_at = datetime.now(timezone.utc)
            return True

    def start_autonomy(self) -> None:
        from thalos_prime.autonomy.loop import AutonomyLoop
        self._autonomy_loop = AutonomyLoop(self.settings, self)
        self._autonomy_loop.start()

    def stop_autonomy(self) -> None:
        if self._autonomy_loop:
            self._autonomy_loop.stop()
