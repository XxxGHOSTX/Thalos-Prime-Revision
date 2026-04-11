from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def detect_language(text: str) -> str:
    try:
        from langdetect import detect
        sample = text[:2000]
        return detect(sample)
    except Exception as e:
        logger.debug("Language detection failed: %s", e)
        return "en"
