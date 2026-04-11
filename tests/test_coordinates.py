from __future__ import annotations

import pytest

from thalos_prime.coordinates.scheme import (
    make_coordinate,
    verify_coordinate,
    base58_encode,
    BASE58_ALPHABET,
)


def test_coordinate_deterministic():
    text = "The sky is blue and the grass is green."
    c1 = make_coordinate(text, "en")
    c2 = make_coordinate(text, "en")
    assert c1 == c2


def test_coordinate_uniqueness():
    c1 = make_coordinate("The sky is blue.", "en")
    c2 = make_coordinate("The grass is green.", "en")
    assert c1 != c2


def test_coordinate_format():
    coord = make_coordinate("This is a test sentence.", "en")
    assert coord.startswith("TP:")
    assert len(coord) == 15


def test_base58_encode():
    data = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f"
    result = base58_encode(data)
    assert isinstance(result, str)
    assert len(result) > 0
    for ch in result:
        assert ch in BASE58_ALPHABET


def test_verify_coordinate():
    text = "Python is a programming language."
    lang = "en"
    coord = make_coordinate(text, lang)
    assert verify_coordinate(text, lang, coord) is True
    assert verify_coordinate("Different text.", lang, coord) is False


def test_coordinate_lang_affects_result():
    text = "The sky is blue."
    c_en = make_coordinate(text, "en")
    c_es = make_coordinate(text, "es")
    assert c_en != c_es


def test_coordinate_case_insensitive():
    c1 = make_coordinate("The SKY is BLUE.", "en")
    c2 = make_coordinate("the sky is blue.", "en")
    assert c1 == c2
