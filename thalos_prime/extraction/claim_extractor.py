from __future__ import annotations

import json
import logging
import subprocess
import sys
from typing import Optional

from thalos_prime.config import Settings
from thalos_prime.coordinates.scheme import make_coordinate
from thalos_prime.extraction.models import ExtractedClaim, ExtractionResult
from thalos_prime.storage.db import Claim, NormalizedDocument, get_session

logger = logging.getLogger(__name__)

_IMPERATIVE_STARTERS = {
    "do", "don't", "make", "take", "use", "try", "get", "let", "go",
    "see", "read", "note", "please", "ensure", "check", "click", "visit",
}


class ClaimExtractor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._nlp = None

    def _load_nlp(self):
        if self._nlp is not None:
            return self._nlp
        try:
            import spacy
            try:
                self._nlp = spacy.load(self.settings.spacy_model)
                logger.info("Loaded spaCy model: %s", self.settings.spacy_model)
            except OSError:
                logger.warning("spaCy model '%s' not found, attempting download...", self.settings.spacy_model)
                try:
                    subprocess.run(
                        [sys.executable, "-m", "spacy", "download", self.settings.spacy_model],
                        check=True,
                        capture_output=True,
                    )
                    self._nlp = spacy.load(self.settings.spacy_model)
                    logger.info("Downloaded and loaded spaCy model: %s", self.settings.spacy_model)
                except Exception as e:
                    logger.warning("Failed to download spaCy model: %s. Using blank model.", e)
                    self._nlp = spacy.blank("en")
        except ImportError:
            logger.warning("spaCy not installed. Claim extraction will be limited.")
            self._nlp = None
        return self._nlp

    def _is_factual_sentence(self, sent) -> bool:
        text = sent.text.strip()
        words = text.split()
        if len(words) < self.settings.min_claim_words:
            return False
        if len(words) > self.settings.max_claim_words:
            return False
        if text.endswith("?"):
            return False
        first_word = words[0].lower().rstrip(",")
        if first_word in _IMPERATIVE_STARTERS:
            return False
        has_verb = any(token.pos_ == "VERB" for token in sent)
        if not has_verb:
            return False
        has_subj = any(token.dep_ in ("nsubj", "nsubjpass") for token in sent)
        if not has_subj:
            return False
        return True

    def _is_factual_sentence_fallback(self, text: str) -> bool:
        words = text.split()
        if len(words) < self.settings.min_claim_words:
            return False
        if len(words) > self.settings.max_claim_words:
            return False
        if text.strip().endswith("?"):
            return False
        return True

    def extract(self, norm_doc: NormalizedDocument) -> ExtractionResult:
        text = norm_doc.translated_text or norm_doc.original_text
        lang = norm_doc.language
        result = ExtractionResult(doc_id=norm_doc.id)

        nlp = self._load_nlp()
        sentences: list[str] = []

        if nlp is not None:
            try:
                doc_nlp = nlp(text[:100000])
                for sent in doc_nlp.sents:
                    if self._is_factual_sentence(sent):
                        sentences.append(sent.text.strip())
                        result.claims_extracted += 1
            except Exception as e:
                logger.warning("spaCy processing failed: %s. Falling back to simple split.", e)
                nlp = None

        if nlp is None:
            import re
            raw_sents = re.split(r"(?<=[.!])\s+", text)
            for s in raw_sents:
                s = s.strip()
                if self._is_factual_sentence_fallback(s):
                    sentences.append(s)
                    result.claims_extracted += 1

        if not sentences:
            return result

        from sqlalchemy import select
        with get_session(self.settings.database_url) as session:
            for sent_text in sentences:
                coord = make_coordinate(sent_text, lang)
                existing = session.execute(
                    select(Claim).where(Claim.coordinate == coord)
                ).scalar_one_or_none()

                if existing:
                    result.claims_duplicate += 1
                    continue

                entities = self._extract_entities_from_text(sent_text)
                claim = Claim(
                    norm_doc_id=norm_doc.id,
                    text=sent_text,
                    coordinate=coord,
                    language=lang,
                    entities_json=json.dumps(entities),
                    status="draft",
                )
                session.add(claim)
                result.claims_new += 1

        return result

    def _extract_entities_from_text(self, text: str) -> list[dict]:
        nlp = self._load_nlp()
        if nlp is None:
            return []
        try:
            doc = nlp(text)
            return [
                {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
                for ent in doc.ents
            ]
        except Exception:
            return []
