from __future__ import annotations

import pytest
from pathlib import Path


def test_file_loader_txt(tmp_settings, tmp_path):
    from thalos_prime.ingestion.file_loader import FileLoader
    loader = FileLoader(tmp_settings)
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("This is a test document. It contains several sentences about Python programming.")
    result = loader.load(txt_file)
    assert result.status == "ingested"
    assert result.doc_id is not None
    assert result.word_count > 0


def test_content_hash_dedup(tmp_settings, tmp_path):
    from thalos_prime.ingestion.file_loader import FileLoader
    loader = FileLoader(tmp_settings)
    content = "This is a test document for deduplication testing purposes."
    txt_file1 = tmp_path / "doc1.txt"
    txt_file1.write_text(content)
    txt_file2 = tmp_path / "doc2.txt"
    txt_file2.write_text(content)

    result1 = loader.load(txt_file1)
    result2 = loader.load(txt_file2)

    assert result1.status == "ingested"
    assert result2.status == "duplicate"
    assert result1.doc_id == result2.doc_id


def test_file_loader_different_files(tmp_settings, tmp_path):
    from thalos_prime.ingestion.file_loader import FileLoader
    loader = FileLoader(tmp_settings)
    txt_file1 = tmp_path / "doc_a.txt"
    txt_file1.write_text("Document A contains unique content about apples and oranges.")
    txt_file2 = tmp_path / "doc_b.txt"
    txt_file2.write_text("Document B has completely different content about bananas and grapes.")

    result1 = loader.load(txt_file1)
    result2 = loader.load(txt_file2)

    assert result1.status == "ingested"
    assert result2.status == "ingested"
    assert result1.doc_id != result2.doc_id
