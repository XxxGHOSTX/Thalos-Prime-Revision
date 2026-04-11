from __future__ import annotations

import hashlib
import re

BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def base58_encode(data: bytes) -> str:
    n = int.from_bytes(data, "big")
    result = []
    while n > 0:
        n, remainder = divmod(n, 58)
        result.append(BASE58_ALPHABET[remainder])
    for byte in data:
        if byte == 0:
            result.append(BASE58_ALPHABET[0])
        else:
            break
    return "".join(reversed(result))


def normalize_for_coordinate(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_coordinate(text: str, lang: str = "en") -> str:
    normalized = normalize_for_coordinate(text)
    payload = f"{lang}::{normalized}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    encoded = base58_encode(digest)
    short = encoded[:12]
    return f"TP:{short}"


def verify_coordinate(text: str, lang: str, coordinate: str) -> bool:
    expected = make_coordinate(text, lang)
    return expected == coordinate
