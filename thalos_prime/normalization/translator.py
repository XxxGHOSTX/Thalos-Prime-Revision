from __future__ import annotations

import logging
from typing import Optional

from thalos_prime.config import Settings

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 4500


def _chunk_text(text: str, size: int = _CHUNK_SIZE) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks = []
    while text:
        chunks.append(text[:size])
        text = text[size:]
    return chunks


class Translator:
    def __init__(self, settings: Settings):
        self.settings = settings

    def translate(self, text: str, source_lang: str, target_lang: str = "en") -> str:
        if source_lang == target_lang:
            return text
        if not text.strip():
            return text

        chunks = _chunk_text(text)
        translated_chunks: list[str] = []

        for chunk in chunks:
            try:
                translated = self._translate_chunk(chunk, source_lang, target_lang)
            except Exception as e:
                logger.warning("Translation chunk failed: %s. Returning original chunk.", e)
                translated = chunk
            translated_chunks.append(translated)

        return " ".join(translated_chunks)

    def _translate_chunk(self, text: str, source_lang: str, target_lang: str) -> str:
        if self.settings.use_cloud_translation and self.settings.deepl_api_key:
            try:
                from deep_translator import DeepL
                translator = DeepL(api_key=self.settings.deepl_api_key, source=source_lang, target=target_lang)
                return translator.translate(text)
            except Exception as e:
                logger.warning("DeepL translation failed: %s. Falling back to Google.", e)

        try:
            from deep_translator import GoogleTranslator
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            return translator.translate(text)
        except Exception as e:
            logger.warning("Google translation failed: %s. Returning original text.", e)
            return text
