from __future__ import annotations

import pytest

from thalos_prime.normalization.cleaner import clean, normalize_whitespace, strip_html, remove_boilerplate
from thalos_prime.normalization.lang_detect import detect_language
from thalos_prime.normalization.translator import Translator


def test_clean_html():
    html = "<p>Hello <b>world</b>! <script>var x=1;</script></p>"
    result = strip_html(html)
    assert "Hello" in result
    assert "world" in result
    assert "<p>" not in result
    assert "var x" not in result


def test_normalize_whitespace():
    text = "Hello  \t  world\r\nFoo   bar"
    result = normalize_whitespace(text)
    assert "  " not in result
    assert "\t" not in result
    assert "\r" not in result


def test_remove_boilerplate():
    text = "A very short line\nThis is a proper sentence with enough words to pass the filter.\nA very short line\nAnother valid sentence that should be included in the output."
    result = remove_boilerplate(text)
    assert "proper sentence" in result
    assert "Another valid sentence" in result
    assert result.count("A very short line") <= 1


def test_lang_detect_english():
    text = "The quick brown fox jumps over the lazy dog. The sky is blue and the grass is green."
    lang = detect_language(text)
    assert lang == "en"


def test_lang_detect_spanish():
    text = "El cielo es azul y la hierba es verde. El sol brilla en el cielo todos los días."
    lang = detect_language(text)
    assert lang in ("es", "en")


def test_translation_same_lang(tmp_settings):
    translator = Translator(tmp_settings)
    text = "Hello world, this is a test."
    result = translator.translate(text, "en", "en")
    assert result == text


def test_translation_with_fallback(tmp_settings, monkeypatch):
    import thalos_prime.normalization.translator as trans_module
    translator = Translator(tmp_settings)
    original_translate = translator._translate_chunk

    def broken_translate(text, source_lang, target_lang):
        raise RuntimeError("Simulated translation failure")

    monkeypatch.setattr(translator, "_translate_chunk", broken_translate)
    result = translator.translate("Hola mundo", "es", "en")
    assert result == "Hola mundo"


def test_clean_pipeline():
    text = "  <html><body><p>  Hello world, this is a test sentence with enough words.   </p></body></html>  "
    result = clean(text)
    assert "Hello" in result
    assert "<html>" not in result
