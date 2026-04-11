from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

MINI_FEVER_SAMPLE = [
    {"id": 1, "claim": "Paris is the capital of France.", "label": "SUPPORTS", "evidence": "Paris is the capital and most populous city of France."},
    {"id": 2, "claim": "The Earth is the largest planet in the solar system.", "label": "REFUTES", "evidence": "Jupiter is the largest planet in the Solar System."},
    {"id": 3, "claim": "Water boils at 100 degrees Celsius at standard pressure.", "label": "SUPPORTS", "evidence": "At standard pressure, water boils at 100°C or 212°F."},
    {"id": 4, "claim": "The Great Wall of China is visible from the Moon.", "label": "REFUTES", "evidence": "The Great Wall of China cannot be seen from the Moon with the naked eye."},
    {"id": 5, "claim": "Shakespeare wrote Romeo and Juliet.", "label": "SUPPORTS", "evidence": "Romeo and Juliet is a tragedy written by William Shakespeare."},
    {"id": 6, "claim": "Humans have 208 bones.", "label": "REFUTES", "evidence": "The adult human body has 206 bones."},
    {"id": 7, "claim": "The speed of light in vacuum is approximately 300,000 km/s.", "label": "SUPPORTS", "evidence": "The speed of light in vacuum is exactly 299,792,458 metres per second."},
    {"id": 8, "claim": "Alexander Fleming discovered antibiotics.", "label": "SUPPORTS", "evidence": "Alexander Fleming is credited with discovering penicillin in 1928."},
    {"id": 9, "claim": "The Amazon River is the longest river in the world.", "label": "REFUTES", "evidence": "The Nile is often considered the world's longest river."},
    {"id": 10, "claim": "DNA carries genetic information in all living organisms.", "label": "SUPPORTS", "evidence": "DNA is the hereditary material in humans and almost all other organisms."},
]

SYNTHETIC_QA_PAIRS = [
    {
        "question": "What is the capital of France?",
        "expected_answer_contains": "Paris",
        "seed_text": "Paris is the capital city of France, known for the Eiffel Tower.",
    },
    {
        "question": "What is the boiling point of water?",
        "expected_answer_contains": "100",
        "seed_text": "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    },
    {
        "question": "Who wrote Hamlet?",
        "expected_answer_contains": "Shakespeare",
        "seed_text": "Hamlet is a tragedy written by William Shakespeare around 1600.",
    },
    {
        "question": "What planet is closest to the Sun?",
        "expected_answer_contains": "Mercury",
        "seed_text": "Mercury is the smallest planet and the closest to the Sun in the solar system.",
    },
    {
        "question": "What is the chemical symbol for gold?",
        "expected_answer_contains": "Au",
        "seed_text": "Gold has the chemical symbol Au, from the Latin word aurum.",
    },
    {
        "question": "How many continents are there?",
        "expected_answer_contains": "7",
        "seed_text": "Earth has seven continents: Africa, Antarctica, Asia, Australia, Europe, North America, and South America.",
    },
    {
        "question": "What is the speed of light?",
        "expected_answer_contains": "299",
        "seed_text": "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
    },
    {
        "question": "Who invented the telephone?",
        "expected_answer_contains": "Bell",
        "seed_text": "Alexander Graham Bell is credited with inventing the telephone in 1876.",
    },
    {
        "question": "What is the largest ocean?",
        "expected_answer_contains": "Pacific",
        "seed_text": "The Pacific Ocean is the largest and deepest ocean on Earth.",
    },
    {
        "question": "What gas do plants absorb?",
        "expected_answer_contains": "carbon dioxide",
        "seed_text": "Plants absorb carbon dioxide from the atmosphere and release oxygen during photosynthesis.",
    },
]


def load_fever_sample(limit: int = 100) -> list[dict]:
    sample = MINI_FEVER_SAMPLE[:limit]
    try:
        from datasets import load_dataset
        dataset = load_dataset("fever", "v1.0", split="paper_dev", streaming=True)
        items = []
        for item in dataset:
            items.append({
                "id": item.get("id"),
                "claim": item.get("claim"),
                "label": item.get("label"),
                "evidence": "",
            })
            if len(items) >= limit:
                break
        if items:
            return items
    except Exception as e:
        logger.debug("Could not load FEVER from HuggingFace: %s. Using built-in sample.", e)
    return sample[:limit]


def load_synthetic_qa() -> list[dict]:
    return SYNTHETIC_QA_PAIRS
