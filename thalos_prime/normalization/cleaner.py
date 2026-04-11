from __future__ import annotations

import re

from bs4 import BeautifulSoup


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_html(text: str) -> str:
    try:
        soup = BeautifulSoup(text, "lxml")
        return soup.get_text(separator="\n")
    except Exception:
        return re.sub(r"<[^>]+>", " ", text)


def remove_boilerplate(text: str) -> str:
    lines = text.splitlines()
    seen: set[str] = set()
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        words = stripped.split()
        if len(words) < 3:
            continue
        key = stripped.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(stripped)
    result = "\n".join(cleaned)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def clean(text: str) -> str:
    text = strip_html(text)
    text = normalize_whitespace(text)
    text = remove_boilerplate(text)
    return text
