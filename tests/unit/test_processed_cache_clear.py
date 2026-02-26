from __future__ import annotations

from pathlib import Path

import pytest

from nvap.cache.processed_cache import CACHE_DIRNAME, clear_processed_cache


def test_clear_processed_cache_only_deletes_processed_npz(tmp_path: Path) -> None:
    cache = tmp_path / CACHE_DIRNAME
    cache.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("keep", encoding="utf-8")

    target = cache / "processed_a123.npz"
    target.write_bytes(b"dummy")
    keep_txt = cache / "notes.txt"
    keep_txt.write_text("keep", encoding="utf-8")
    keep_npz = cache / "other_data.npz"
    keep_npz.write_bytes(b"keep")
    nested = cache / "nested"
    nested.mkdir()
    (nested / "processed_b456.npz").write_bytes(b"keep-nested")

    removed, cache_path = clear_processed_cache(tmp_path)

    assert removed == 1
    assert cache_path == cache.resolve()
    assert not target.exists()
    assert keep_txt.exists()
    assert keep_npz.exists()
    assert nested.exists()
    assert outside.exists()


def test_clear_processed_cache_missing_dir_is_safe(tmp_path: Path) -> None:
    removed, cache_path = clear_processed_cache(tmp_path)
    assert removed == 0
    assert cache_path == (tmp_path / CACHE_DIRNAME).resolve()


def test_clear_processed_cache_rejects_non_directory(tmp_path: Path) -> None:
    bad = tmp_path / CACHE_DIRNAME
    bad.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError):
        clear_processed_cache(tmp_path)
