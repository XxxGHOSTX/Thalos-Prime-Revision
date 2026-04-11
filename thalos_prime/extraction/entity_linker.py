from __future__ import annotations

import logging
from typing import Optional

from thalos_prime.config import Settings

logger = logging.getLogger(__name__)


class EntityLinker:
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
            except OSError:
                self._nlp = spacy.blank("en")
        except ImportError:
            self._nlp = None
        return self._nlp

    def extract_entities(self, text: str) -> list[dict]:
        nlp = self._load_nlp()
        if nlp is None:
            return []
        try:
            doc = nlp(text)
            return [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
                for ent in doc.ents
            ]
        except Exception as e:
            logger.warning("Entity extraction failed: %s", e)
            return []
