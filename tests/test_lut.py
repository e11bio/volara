import fsspec
import numpy as np
import pytest

from volara.lut import LUT, LUTS


@pytest.fixture
def memory_fs():
    """A clean in-memory fsspec filesystem: `memory://...` paths exercise every
    remote code path without a bucket."""
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    fs.pseudo_dirs.clear()
    yield fs
    fs.store.clear()
    fs.pseudo_dirs.clear()


def test_lut_save_load(tmp_path):
    lut = LUT(path=tmp_path / "test_lut.npz")
    data = np.array([[1, 2, 3], [10, 20, 30]])
    lut.save(data)
    loaded = lut.load()
    assert loaded is not None
    np.testing.assert_array_equal(loaded, data)


def test_lut_load_missing(tmp_path):
    lut = LUT(path=tmp_path / "nonexistent.npz")
    assert lut.load() is None


def test_lut_drop(tmp_path):
    lut = LUT(path=tmp_path / "drop_test.npz")
    lut.save(np.array([[1], [2]]))
    assert lut.file.exists()
    lut.drop()
    assert not lut.file.exists()


def test_lut_path_extension(tmp_path):
    """`.npz` appended when missing from string path."""
    lut = LUT(path=str(tmp_path / "no_ext"))
    assert lut.file.suffix == ".npz"
    assert str(lut.file).endswith("no_ext.npz")

    # If already has .npz, don't double it
    lut2 = LUT(path=str(tmp_path / "has_ext.npz"))
    assert str(lut2.file).endswith("has_ext.npz")


def test_lut_add_creates_luts(tmp_path):
    lut_a = LUT(path=tmp_path / "a.npz")
    lut_b = LUT(path=tmp_path / "b.npz")
    result = lut_a + lut_b
    assert isinstance(result, LUTS)
    assert len(result.luts) == 2


def test_luts_load(tmp_path):
    """Concatenation of multiple LUTs."""
    lut_a = LUT(path=tmp_path / "a.npz")
    lut_b = LUT(path=tmp_path / "b.npz")
    lut_a.save(np.array([[1, 2], [10, 20]]))
    lut_b.save(np.array([[3, 4], [30, 40]]))
    luts = LUTS(luts=[lut_a, lut_b])
    loaded = luts.load()
    assert loaded.shape == (2, 4)
    np.testing.assert_array_equal(loaded, [[1, 2, 3, 4], [10, 20, 30, 40]])


def test_lut_uri_local(tmp_path):
    """`uri` is the normalized local path for non-s3 luts."""
    lut = LUT(path=str(tmp_path / "local"))
    assert lut.uri == str(tmp_path / "local.npz")
    assert lut.name == "local"


def test_lut_remote_save_load(memory_fs):
    from pathlib import Path

    lut = LUT(path="memory://bucket/path/to/data.zarr/lut.npz")
    assert not isinstance(lut.file, Path)  # a remote UPath, no local counterpart
    assert lut.uri == "memory://bucket/path/to/data.zarr/lut.npz"
    assert lut.name == "lut"
    assert not lut.exists()
    assert lut.load() is None

    data = np.array([[1, 2, 3], [10, 20, 30]])
    lut.save(data)
    assert lut.exists()
    assert memory_fs.exists("/bucket/path/to/data.zarr/lut.npz")
    np.testing.assert_array_equal(lut.load(), data)


def test_lut_remote_save_load_with_edges(memory_fs):
    lut = LUT(path="memory://bucket/lut.npz")
    data = np.array([[1, 2], [10, 20]])
    edges = np.array([[1, 2]])
    lut.save(data, edges=edges)
    np.testing.assert_array_equal(lut.load(), data)


def test_lut_remote_extension_appended(memory_fs):
    lut = LUT(path="memory://bucket/path/to/data.zarr/lut")
    assert lut.uri == "memory://bucket/path/to/data.zarr/lut.npz"
    lut.save(np.array([[1], [2]]))
    assert memory_fs.exists("/bucket/path/to/data.zarr/lut.npz")


def test_lut_remote_drop(memory_fs):
    lut = LUT(path="memory://bucket/lut.npz")
    lut.save(np.array([[1], [2]]))
    assert lut.exists()
    lut.drop()
    assert not lut.exists()
    # dropping a missing lut is a no-op
    lut.drop()


def test_luts_mixed_local_and_remote(tmp_path, memory_fs):
    """Local and remote luts can be combined."""
    local = LUT(path=tmp_path / "a.npz")
    remote = LUT(path="memory://bucket/b.npz")
    local.save(np.array([[1, 2], [10, 20]]))
    remote.save(np.array([[10, 20], [100, 200]]))

    luts = local + remote
    np.testing.assert_array_equal(luts.load(), [[1, 2, 10, 20], [10, 20, 100, 200]])
    iterated = luts.load_iterated()
    np.testing.assert_array_equal(iterated[0], [1, 2])
    np.testing.assert_array_equal(iterated[1], [100, 200])


def test_lut_config_round_trips_as_the_string_it_was_given(tmp_path):
    """`path` serialises as a plain string, so saved configs are unchanged, and
    `s3_kwargs` is still accepted as the former name of `storage_options`."""
    for given in (
        "s3://bucket/lut.npz",
        str(tmp_path / "lut.npz"),
        tmp_path / "lut.npz",
    ):
        lut = LUT(path=given)
        assert lut.model_dump()["path"] == str(given)
        assert LUT.model_validate_json(lut.model_dump_json()) == lut

    assert LUT.model_json_schema()["properties"]["path"]["type"] == "string"

    lut = LUT.model_validate(
        {"path": "memory://bucket/lut.npz", "s3_kwargs": {"anon": True}}
    )
    assert lut.storage_options == {"anon": True}
    assert lut.path.storage_options == {"anon": True}
    assert "s3_kwargs" not in lut.model_dump()


def test_luts_load_skips_missing(tmp_path):
    """Missing luts are skipped rather than breaking concatenation."""
    lut_a = LUT(path=tmp_path / "a.npz")
    lut_b = LUT(path=tmp_path / "missing.npz")
    lut_a.save(np.array([[1, 2], [10, 20]]))
    loaded = LUTS(luts=[lut_a, lut_b]).load()
    np.testing.assert_array_equal(loaded, [[1, 2], [10, 20]])


def test_luts_load_iterated(tmp_path):
    """Chained lookup tables: frag->mid->seg."""
    lut_a = LUT(path=tmp_path / "a.npz")
    lut_b = LUT(path=tmp_path / "b.npz")
    # a: 1->10, 2->20
    lut_a.save(np.array([[1, 2], [10, 20]]))
    # b: 10->100, 20->200
    lut_b.save(np.array([[10, 20], [100, 200]]))
    luts = LUTS(luts=[lut_a, lut_b])
    loaded = luts.load_iterated()
    # Should chain: 1->100, 2->200
    np.testing.assert_array_equal(loaded[0], [1, 2])
    np.testing.assert_array_equal(loaded[1], [100, 200])
