from pathlib import Path

import pytest

from volara.datasets import Raw


def _store(tmp_path) -> Path:
    p = tmp_path / "x.zarr"
    p.mkdir()
    (p / "zarr.json").write_text("{}")
    return p


def test_drop_removes_a_path_store(tmp_path):
    p = _store(tmp_path)
    Raw(store=p).drop()
    assert not p.exists()


def test_drop_removes_a_local_str_store(tmp_path):
    p = _store(tmp_path)
    Raw(store=str(p)).drop()
    assert not p.exists()


def test_drop_of_a_missing_local_store_is_a_noop(tmp_path):
    Raw(store=str(tmp_path / "never-created.zarr")).drop()


def test_drop_refuses_a_non_s3_url_rather_than_silently_skipping(tmp_path):
    with pytest.raises(ValueError, match="URL this method cannot delete"):
        Raw(store="gs://bucket/x.zarr").drop()
