from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from nvap import samples
from nvap.ui.main_window import MainWindow
from nvap.ui.services.project_files import load_project_state, project_channel_sources
from nvap.ui.services.recent_projects import ProjectEntry, RecentProjectsStore


def entry(path, date="2026-01-01", **kwargs):
    return ProjectEntry(name=Path(path).name, path=str(path), last_opened_iso=date, **kwargs)


def test_old_temporary_sample_entries_are_migrated_and_deduplicated(tmp_path, monkeypatch):
    source = tmp_path / "repo" / "samples" / "cortex"
    source.mkdir(parents=True)
    user_project = tmp_path / "user_project" / "cortex"
    user_project.mkdir(parents=True)
    store = RecentProjectsStore(tmp_path / "state" / "recent_projects.json")
    store.replace_all([
        entry(tmp_path / "_MEI111" / "samples" / "cortex", "2026-10-01", pinned=True, tags=["keep"]),
        entry(tmp_path / "_MEI222" / "samples" / "cortex", "2026-09-01", samples=1),
        entry(source, status="Sample"), entry(user_project, status="Ready"),
    ])
    monkeypatch.setattr(samples, "discover_bundled_samples", lambda: [("cortex", source)])
    monkeypatch.setattr(sys, "frozen", False, raising=False)

    assert samples.register_bundled_samples(store) == 0
    assert store.count() == 2
    migrated = store.find(source)
    assert migrated.exists and migrated.sample_id == "cortex"
    assert migrated.last_opened_iso == "2026-10-01"
    assert migrated.pinned and migrated.samples == 1 and migrated.tags == ["keep"]
    assert store.find(user_project) is not None
    monkeypatch.setattr(store, "_save", lambda: pytest.fail("Unchanged startup rewrote the store"))
    assert samples.register_bundled_samples(store) == 0


def test_frozen_samples_survive_a_different_extraction_directory(tmp_path, monkeypatch):
    first = tmp_path / "_MEI111" / "samples" / "cortex"
    second = tmp_path / "_MEI222" / "samples" / "cortex"
    for source in (first, second):
        source.mkdir(parents=True)
        (source / "slice_z001c1.png").write_bytes(b"sample image")
    store_path = tmp_path / "persistent" / "recent_projects.json"
    store = RecentProjectsStore(store_path)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(samples, "discover_bundled_samples", lambda: [("cortex", first)])

    assert samples.register_bundled_samples(store) == 1
    persistent = Path(store.all()[0].path)
    assert persistent == tmp_path / "persistent" / "samples" / "cortex"
    (first / "slice_z001c1.png").unlink()
    assert (persistent / "slice_z001c1.png").read_bytes() == b"sample image"
    monkeypatch.setattr(samples, "discover_bundled_samples", lambda: [("cortex", second)])
    restarted = RecentProjectsStore(store_path)
    assert samples.register_bundled_samples(restarted) == 0
    assert restarted.count() == 1
    assert restarted.all()[0].exists and Path(restarted.all()[0].path) == persistent


def test_moved_project_rebases_internal_sources_and_preserves_external_sources(tmp_path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    external = tmp_path / "external.tif"
    new.mkdir()
    descriptor = {
        "version": 1, "root": str(old),
        "channel_sources": {"green": str(old / "green"), "red": str(external)},
    }
    path = new / ".nvap_project.json"
    path.write_text(json.dumps(descriptor))

    state = load_project_state(new)
    assert project_channel_sources(state) == {"green": str(new / "green"), "red": str(external)}
    assert json.loads(path.read_text()) == descriptor


@pytest.mark.parametrize("choice", ["cancel", "invalid", "valid"])
def test_missing_project_can_be_relocated_without_losing_history(qtbot, monkeypatch, tmp_path, choice):
    window = MainWindow()
    qtbot.addWidget(window)
    old = tmp_path / "old"
    new = tmp_path / "new"
    new.mkdir()
    store = RecentProjectsStore(tmp_path / "recents.json")
    store.replace_all([entry(old, pinned=True, tags=["important"], samples=2)])
    window._project_store = store
    calls = []
    errors = []
    monkeypatch.setattr("nvap.ui.main_window.QFileDialog.getExistingDirectory", lambda *a: "" if choice == "cancel" else str(new))
    monkeypatch.setattr(window, "_start_dataset_load", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(window, "_show_error", lambda *a: errors.append(a))
    monkeypatch.setattr(window._home_page, "refresh_projects", lambda: None)

    def resolve(*args, **kwargs):
        if choice == "invalid":
            raise ValueError("No channel images")
        return {"green": new, "red": new}

    monkeypatch.setattr("nvap.ui.main_window.resolve_channel_dirs", resolve)
    window._open_recent_project(str(old))

    if choice == "valid":
        restored = store.find(new)
        assert restored and restored.pinned and restored.tags == ["important"] and restored.samples == 2
        assert store.find(old) is None
        assert calls and not errors
    else:
        assert store.find(old) is not None
        assert not calls
        assert bool(errors) == (choice == "invalid")
