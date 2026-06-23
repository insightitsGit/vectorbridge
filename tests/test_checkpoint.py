"""Tests for checkpoint save/load/resume."""

import os
import tempfile
import pytest

from vectorbridge.checkpoint import Checkpoint, checkpoint_path


def test_checkpoint_save_and_load(tmp_path):
    cp = Checkpoint.new("job_abc", "chromadb", "qdrant", "full")
    cp.advance(100, 100, "v0100")
    path = str(tmp_path / "cp.json")
    cp.save(path)

    loaded = Checkpoint.load(path)
    assert loaded.job_id == "job_abc"
    assert loaded.vectors_transferred == 100
    assert loaded.last_offset == 100
    assert loaded.last_id == "v0100"
    assert loaded.completed is False


def test_checkpoint_mark_complete(tmp_path):
    cp = Checkpoint.new("job_xyz", "faiss", "pgvector", "full")
    cp.advance(50, 50)
    cp.mark_complete()
    path = str(tmp_path / "cp.json")
    cp.save(path)

    loaded = Checkpoint.load(path)
    assert loaded.completed is True


def test_checkpoint_advance_accumulates(tmp_path):
    cp = Checkpoint.new("job_multi", "chromadb", "qdrant", "full")
    cp.advance(100, 100)
    cp.advance(100, 200)
    cp.advance(100, 300)
    assert cp.vectors_transferred == 300
    assert cp.batches_completed == 3
    assert cp.last_offset == 300


def test_checkpoint_path_format():
    path = checkpoint_path("my_job_123")
    assert "my_job_123" in path
    assert path.endswith(".json")
