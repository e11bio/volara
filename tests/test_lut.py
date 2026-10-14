import fsspec
import numpy as np
import pytest

from volara.lut import LUT, LUTS


@pytest.fixture
def fake_s3(monkeypatch):
    """
    Back `LUT._s3fs` with an in-memory fsspec filesystem so that s3 code
    paths can be exercised without a bucket.
    """
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    fs.pseudo_dirs.clear()
    monkeypatch.setattr(LUT, "_s3fs", lambda self: fs)
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
    assert not lut.is_s3
    assert lut.uri == str(tmp_path / "local.npz")
    assert lut.name == "local"


def test_lut_s3_save_load(fake_s3):
    lut = LUT(path="s3://bucket/path/to/data.zarr/lut.npz")
    assert lut.is_s3
    assert lut.uri == "s3://bucket/path/to/data.zarr/lut.npz"
    assert lut.name == "lut"
    assert not lut.exists()
    assert lut.load() is None

    data = np.array([[1, 2, 3], [10, 20, 30]])
    lut.save(data)
    assert lut.exists()
    np.testing.assert_array_equal(lut.load(), data)


def test_lut_s3_save_load_with_edges(fake_s3):
    lut = LUT(path="s3://bucket/lut.npz")
    data = np.array([[1, 2], [10, 20]])
    edges = np.array([[1, 2, 0.5]])
    lut.save(data, edges=edges)
    np.testing.assert_array_equal(lut.load(), data)


def test_lut_s3_extension_appended(fake_s3):
    lut = LUT(path="s3://bucket/path/to/data.zarr/lut")
    assert lut.uri == "s3://bucket/path/to/data.zarr/lut.npz"
    lut.save(np.array([[1], [2]]))
    assert fake_s3.exists("s3://bucket/path/to/data.zarr/lut.npz")


def test_lut_s3_drop(fake_s3):
    lut = LUT(path="s3://bucket/lut.npz")
    lut.save(np.array([[1], [2]]))
    assert lut.exists()
    lut.drop()
    assert not lut.exists()
    # dropping a missing lut is a no-op
    lut.drop()


def test_lut_s3_file_raises(fake_s3):
    """s3 luts have no local path."""
    lut = LUT(path="s3://bucket/lut.npz")
    with pytest.raises(ValueError, match="no local path"):
        lut.file


def test_luts_mixed_local_and_s3(tmp_path, fake_s3):
    """Local and s3 luts can be combined."""
    local = LUT(path=tmp_path / "a.npz")
    remote = LUT(path="s3://bucket/b.npz")
    local.save(np.array([[1, 2], [10, 20]]))
    remote.save(np.array([[10, 20], [100, 200]]))

    luts = local + remote
    np.testing.assert_array_equal(luts.load(), [[1, 2, 10, 20], [10, 20, 100, 200]])
    iterated = luts.load_iterated()
    np.testing.assert_array_equal(iterated[0], [1, 2])
    np.testing.assert_array_equal(iterated[1], [100, 200])


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
