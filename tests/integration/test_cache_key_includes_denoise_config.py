from __future__ import annotations

from dataclasses import replace

from nvap.cache.processed_cache import build_processed_cache_key
from nvap.config.types import DEFAULT_SPACING, PSFConfig, PreprocessConfig


def test_cache_key_changes_with_green_denoise_strategy() -> None:
    dataset_sig = "a" * 64
    base = build_processed_cache_key(
        dataset_signature=dataset_sig,
        spacing=DEFAULT_SPACING,
        psf_config=PSFConfig(iterations=1),
        preprocess_config=PreprocessConfig(green_denoise_strategy="hybrid_auto"),
    )
    changed = build_processed_cache_key(
        dataset_signature=dataset_sig,
        spacing=DEFAULT_SPACING,
        psf_config=PSFConfig(iterations=1),
        preprocess_config=PreprocessConfig(green_denoise_strategy="legacy_anisotropic"),
    )
    assert base != changed


def test_cache_key_changes_with_branch_protection() -> None:
    dataset_sig = "b" * 64
    cfg = PreprocessConfig()
    key_a = build_processed_cache_key(dataset_sig, DEFAULT_SPACING, PSFConfig(iterations=1), cfg)
    key_b = build_processed_cache_key(
        dataset_sig,
        DEFAULT_SPACING,
        PSFConfig(iterations=1),
        replace(cfg, green_branch_protection=cfg.green_branch_protection + 0.1),
    )
    assert key_a != key_b
