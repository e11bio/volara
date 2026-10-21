import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
import zarr
from funlib.geometry import Coordinate
from funlib.persistence import prepare_ds
from pydantic import ValidationError

from volara.datasets import LSD, Affs, Labels, MaskDataset, Raw


@pytest.fixture
def zarr_store(tmp_path):
    """Creates a temporary Zarr array for testing."""
    path = tmp_path / "test_data.zarr"
    store = zarr.open_array(
        str(path), mode="w", shape=(2, 10, 10), chunks=(1, 5, 5), dtype="float32"
    )
    store[:] = np.ones((2, 10, 10))  # Fill with 1s
    store.attrs["offset"] = (0, 0, 0)
    store.attrs["voxel_size"] = (1, 1, 1)
    return path


@pytest.fixture
def second_zarr_store(tmp_path):
    """Creates a second temporary Zarr array for stacking tests."""
    path = tmp_path / "stack_data.zarr"
    store = zarr.open_array(
        str(path), mode="w", shape=(2, 10, 10), chunks=(1, 5, 5), dtype="float32"
    )
    store[:] = np.full((2, 10, 10), 2.0)  # Fill with 2s
    return path


# ==========================================
# 2. Test Cases
# ==========================================

# --- Test Category 1: Serialization / Deserialization ---


@pytest.mark.parametrize(
    "stack",
    [
        "",
        """
    stack:
      dataset_type: raw
      store: {stack_store}
      scale_shift: [2.0, 0.0]
    """,
        """
    stack:
      dataset_type: labels
      store: {stack_store}
    """,
        """
    stack:
      dataset_type: mask
      store: {stack_store}
    """,
    ],
    ids=["no-stack", "stack-raw", "stack-labels", "stack-mask"],
)
def test_raw_serialization_roundtrip(tmp_path, stack):
    """
    Test that YAML -> Python A -> Python B ensures equality.
    Checks handling of tuple/list conversion in Pydantic.

    A worker rebuilds its task from the JSON dump in ``config.json``, so the
    nested ``stack`` dataset has to come back as its own class with its own
    fields: ``Raw.stack`` is typed with the discriminated union, not the
    abstract ``Dataset``, which would dump only the base fields and reload as
    nothing.
    """
    yaml_data = f"""
    dataset_type: raw
    store: {tmp_path / "test.zarr"}
    voxel_size: [10, 10, 10]
    scale_shift: [0.5, 10.0]
    channels: 0
    """ + stack.format(stack_store=tmp_path / "stack.zarr")

    # 1. Load YAML to Dict
    config_dict = yaml.safe_load(yaml_data)

    # 2. Create Python Object A
    raw_a = Raw(**config_dict)

    # 3. Dump A to Dict (simulating YAML dump)
    dumped_a = raw_a.model_dump()

    # 4. Create Python Object B from Dump
    raw_b = Raw(**dumped_a)

    # Assertions
    assert raw_a == raw_b
    assert raw_a.scale_shift == (0.5, 10.0)
    # Check that list in yaml became tuple/list correctly based on logic
    assert raw_a.voxel_size == (10, 10, 10) or raw_a.voxel_size == [10, 10, 10]

    # 5. The JSON round trip a spawned worker makes through config.json
    raw_c = Raw.model_validate_json(raw_a.model_dump_json())
    assert raw_c.model_dump(mode="json") == raw_a.model_dump(mode="json")

    if stack:
        expected = config_dict["stack"]
        for back in (raw_b, raw_c):
            assert back.stack is not None
            assert back.stack.dataset_type == expected["dataset_type"]
            assert type(back.stack) is type(raw_a.stack)
            if "scale_shift" in expected:
                assert back.stack.scale_shift == tuple(expected["scale_shift"])
    else:
        assert raw_b.stack is None and raw_c.stack is None


def test_affs_serialization(tmp_path):
    """Test Affs specific serialization with neighborhood."""
    yaml_data = f"""
    dataset_type: affs
    store: {tmp_path / "affs.zarr"}
    neighborhood:
      - [0, 1, 0]
      - [0, 0, 1]
    """
    config_dict = yaml.safe_load(yaml_data)
    affs_obj = Affs(**config_dict)

    # Verify neighborhood parsed correctly
    assert len(affs_obj.neighborhood) == 2
    assert tuple(affs_obj.neighborhood[0]) == (0, 1, 0)

    # Round trip
    affs_reloaded = Affs(**affs_obj.model_dump())
    assert affs_obj == affs_reloaded


# --- Test Category 2: Lazy Operations ---


def test_lazy_channel_slicing(zarr_store):
    """
    Test that 'channels' attribute applies slicing to the array.
    Original Shape: (2, 10, 10) (C, Y, X)
    """
    ds = Raw(store=zarr_store, channels=0)

    arr = ds.array()
    data = arr[:]

    assert data.shape == (10, 10)
    assert np.all(data == 1.0)  # Original data was 1s

    ds = Raw(store=zarr_store, channels=[0, [0, 2, 4, 6, 8]])

    arr = ds.array()
    data = arr[:]

    assert data.shape == (5, 10)
    assert np.all(data == 1.0)  # Original data was 1s

    ds = Raw(store=zarr_store, channels=[0, 0, [0, 2, 4, 6, 8]])

    arr = ds.array()
    data = arr[:]

    assert data.shape == (5,)
    assert np.all(data == 1.0)  # Original data was 1s


def test_lazy_scale_shift(zarr_store):
    """
    Test that scale_shift lazy op modifies values.
    Original: 1.0
    Scale: 2.0, Shift: 5.0
    Expected: 7.0
    """
    ds = Raw(store=zarr_store, scale_shift=(2.0, 5.0))
    arr = ds.array()
    data = arr[:]

    assert np.allclose(data, 7.0)


def test_lazy_stacking(zarr_store, second_zarr_store):
    """
    Test stacking two datasets.
    DS 1 (Base): values=1.0, shape=(2,10,10)
    DS 2 (Stack): values=2.0, shape=(2,10,10)
    Result: shape=(4,10,10)
    """
    stack_ds = Raw(store=second_zarr_store)
    base_ds = Raw(store=zarr_store, stack=stack_ds)

    arr = base_ds.array()
    data = arr[:]

    assert data.shape == (4, 10, 10)
    # First 2 channels should be 1.0 (base)
    assert np.all(data[0:2] == 1.0)
    # Next 2 channels should be 2.0 (stacked)
    assert np.all(data[2:4] == 2.0)


# --- Test Category 3: Attributes ---


def test_attrs_generation_raw(zarr_store):
    """Test that Raw generates correct attributes (bounds)."""
    # Create a dummy zarr for OME metadata
    ome_path = Path(zarr_store).parent / "ome.zarr"
    ome = zarr.open_group(str(ome_path), mode="w")
    # Mock minimal OME structure
    ome.attrs["omero"] = {
        "channels": [
            {"window": {"min": 0, "max": 255}},
            {"window": {"min": 10, "max": 100}},
        ]
    }

    ds = Raw(store=zarr_store, ome_norm=ome_path)

    attrs = ds.attrs
    assert "bounds" in attrs
    assert attrs["bounds"] == [(0, 255), (10, 100)]


def test_attrs_affs_validation(tmp_path):
    """
    Test Affs validation logic regarding neighborhood in attributes vs arguments.
    """
    path = tmp_path / "affs.zarr"

    # Case 1: New dataset, neighborhood provided in code -> OK
    nh = [Coordinate(0, 1), Coordinate(1, 0)]
    ds = Affs(store=path, neighborhood=nh)
    assert ds.attrs["neighborhood"] == nh

    # Simulate writing this to disk (mocking what prepare() would do)
    ds.prepare(
        (2, 10, 10),
        (1, 5, 5),
        (0, 0),
        (1, 1),
        ("nm", "nm"),
        ("affs", "y", "x"),
        ("affs", "space", "space"),
        np.float32,
    )

    # Case 2: Existing dataset on disk, no neighborhood in code -> OK (reads from disk)
    ds_load = Affs(store=path)
    # Note: The model_post_init logic in your code tries to open the array.
    # Since we created the zarr above, open_ds should find it.
    # We verify that it populated the pydantic model field from the disk attrs
    assert len(ds_load.neighborhood) == 2

    # Case 3: Mismatch -> Error
    with pytest.raises(ValidationError):
        # Disk has [[0,1], [1,0]], we provide [[5,5]]
        Affs(store=path, neighborhood=[Coordinate(5, 5)])


def test_attrs_labels_lsd():
    """Simple check for static attributes in subclasses."""
    l = Labels(store="dummy")
    assert l.attrs == {}

    lsd = LSD(store="dummy")
    assert lsd.attrs == {"lsds": True}


def test_list_channels_keep_the_array_writeable(tmp_path):
    """List-based `channels` must use a native slice op, not a callable.

    funlib.persistence marks an array unwriteable as soon as a custom callable lazy
    op is applied, so selecting channels with `lambda d: d[channels]` silently makes
    every OUTPUT dataset unwritable -- the write then fails at block-process time,
    not at config time.

    pytest tests/test_dataset.py::test_list_channels_keep_the_array_writeable
    """
    store = tmp_path / "d.zarr" / "labels"
    prepare_ds(
        store,
        shape=(5, 4, 8, 8),
        offset=Coordinate(0, 0, 0),
        voxel_size=Coordinate(1, 1, 1),
        axis_names=["c^", "z", "y", "x"],
        dtype=np.uint8,
        mode="w",
    )[:] = np.arange(5 * 4 * 8 * 8, dtype=np.uint8).reshape(5, 4, 8, 8)

    arr = Labels(store=store, channels=[[0, 2]]).array(mode="a")

    # Correct selection...
    expected = np.arange(5 * 4 * 8 * 8, dtype=np.uint8).reshape(5, 4, 8, 8)[[0, 2]]
    assert arr.shape == expected.shape
    np.testing.assert_array_equal(arr[:], expected)

    # ...and still writeable, which a callable lazy op would have prevented.
    assert arr.is_writeable
    arr[:] = np.zeros_like(expected)
    np.testing.assert_array_equal(arr[:], np.zeros_like(expected))


# --- Test Category 4: Drop ---


@pytest.mark.parametrize("as_str", [False, True], ids=["Path", "str"])
def test_drop_removes_a_local_store(zarr_store, as_str):
    """`store` is `Path | str`; a local path must be dropped whichever it arrives as."""
    store = str(zarr_store) if as_str else zarr_store
    Raw(store=store).drop()
    assert not zarr_store.exists()


@pytest.mark.parametrize("as_str", [False, True], ids=["Path", "str"])
def test_drop_of_a_missing_local_store_is_a_noop(tmp_path, as_str):
    path = tmp_path / "never-created.zarr"
    Raw(store=str(path) if as_str else path).drop()
    assert not path.exists()


@pytest.fixture
def memory_fs():
    import fsspec

    fs = fsspec.filesystem("memory")
    fs.store.clear()
    fs.pseudo_dirs.clear()
    yield fs
    fs.store.clear()
    fs.pseudo_dirs.clear()


def test_a_remote_store_is_prepared_written_read_and_dropped(memory_fs):
    """`store` may be a URL on any fsspec filesystem; every operation goes
    through the same `UPath`, so nothing is special-cased on `s3://`."""
    ds = Labels(store="memory://bucket/data.zarr/labels")
    assert not ds.is_local
    assert ds.name == "labels"
    ds.prepare(
        shape=(4, 4), chunk_shape=(2, 2), offset=(0, 0), voxel_size=(1, 1),
        units=("nm", "nm"), axis_names=("y", "x"), types=("space", "space"),
        dtype=np.uint8,
    )
    assert memory_fs.exists("/bucket/data.zarr/labels/zarr.json")
    arr = ds.array("r+")
    arr[:] = 7
    np.testing.assert_array_equal(ds.array("r")[:], np.full((4, 4), 7, dtype=np.uint8))
    ds.drop()
    assert not memory_fs.exists("/bucket/data.zarr/labels/zarr.json")
    ds.drop()  # a missing remote store is a no-op


def test_store_serialises_as_the_string_it_was_given(tmp_path):
    for given in ("s3://bucket/x.zarr", str(tmp_path / "x.zarr"), tmp_path / "x.zarr"):
        ds = Raw(store=given)
        assert ds.model_dump()["store"] == str(given)
        assert Raw.model_validate_json(ds.model_dump_json()) == ds


def test_s3_kwargs_is_accepted_as_the_former_name_of_storage_options():
    ds = Raw.model_validate({"store": "memory://bucket/x.zarr", "s3_kwargs": {"anon": True}})
    assert ds.storage_options == {"anon": True}
    assert ds.store.storage_options == {"anon": True}  # carried by the UPath itself
    assert "s3_kwargs" not in ds.model_dump()


# --- MaskDataset: a block-grid skip mask ---


def _write_mask(path, array):
    z = zarr.create_array(
        store=str(path), shape=array.shape, chunks=array.shape, dtype=array.dtype
    )
    z[:] = array
    return path


def test_mask_dataset_round_trips_and_reads_like_any_dataset(tmp_path):
    array = np.array([[0], [1], [0], [0]], dtype=np.uint8)
    dataset = MaskDataset(store=_write_mask(tmp_path / "mask.zarr", array))

    rebuilt = MaskDataset.model_validate_json(dataset.model_dump_json())

    assert rebuilt.dataset_type == "mask"
    assert rebuilt == dataset
    assert rebuilt.array("r")[:].tolist() == array.tolist()


# --- Storage layout: chunk_shape (the write unit), sharding, inner_chunk_shape ------------------

_FUNLIB_SHARDS = "sharding" in inspect.signature(prepare_ds).parameters
needs_sharding = pytest.mark.skipif(
    not _FUNLIB_SHARDS,
    reason="funlib.persistence here has no prepare_ds(sharding=...) (funkelab/funlib.persistence#40)",
)


def _prepare(ds, write_shape, shape=(2, 64, 64)):
    """`Dataset.prepare` as a task calls it: `chunk_shape` is the block it writes."""
    ds.prepare(
        shape=shape,
        chunk_shape=write_shape,
        offset=(0, 0),
        voxel_size=(1, 1),
        units=("nm", "nm"),
        axis_names=("c^", "y", "x"),
        types=("channel", "space", "space"),
        dtype=np.uint8,
    )


@pytest.mark.parametrize(
    "layout, problem",
    [
        ({"inner_chunk_shape": (2, 8, 8)}, "only applies with sharding=True"),
        ({"sharding": True, "chunk_shape": (2, 32, 32), "inner_chunk_shape": (8, 8)},
         "differ in dimensions"),
        ({"sharding": True, "chunk_shape": (2, 32, 32), "inner_chunk_shape": (2, 12, 12)},
         "whole number of inner_chunk_shape"),
        ({"chunk_shape": (0, 8, 8)}, "positive per axis"),
        ({"sharding": True, "chunk_shape": (2, 8, 8), "inner_chunk_shape": (2, 0, 8)},
         "positive per axis"),
    ],
    ids=["inner-without-sharding", "dims", "not-a-multiple", "zero-chunk", "zero-inner"],
)
def test_a_layout_is_validated_at_construction(tmp_path, layout, problem):
    with pytest.raises(ValidationError, match=problem):
        Labels(store=tmp_path / "x.zarr", **layout)


def test_stored_unit_is_the_declared_chunk_shape(tmp_path):
    store = tmp_path / "x.zarr"
    assert Labels(store=store).stored_unit is None
    assert Labels(store=store, chunk_shape=(2, 8, 8)).stored_unit == Coordinate(2, 8, 8)
    assert Labels(store=store, chunk_shape=(2, 32, 32), sharding=True).stored_unit == Coordinate(2, 32, 32)


def test_the_layout_rides_in_the_config_a_worker_rebuilds(tmp_path):
    ds = Labels(store=tmp_path / "x.zarr", chunk_shape=(2, 32, 32), sharding=True,
                inner_chunk_shape=(2, 8, 8))
    again = Labels.model_validate_json(ds.model_dump_json())
    assert (again.chunk_shape, again.sharding, again.inner_chunk_shape) == (
        Coordinate(2, 32, 32), True, Coordinate(2, 8, 8))
    assert again == ds


def test_prepare_stores_the_write_block_as_one_chunk_by_default(tmp_path):
    ds = Labels(store=tmp_path / "x.zarr")
    _prepare(ds, write_shape=(2, 16, 16))
    arr = ds.array("r")
    assert arr.chunk_shape == Coordinate(2, 16, 16)
    assert arr._source_data.shards is None


def test_prepare_refuses_a_write_block_that_is_not_whole_chunks(tmp_path):
    ds = Labels(store=tmp_path / "x.zarr", chunk_shape=(2, 16, 16))
    with pytest.raises(ValueError, match="not a whole number of chunks"):
        _prepare(ds, write_shape=(2, 24, 24))
    assert not (tmp_path / "x.zarr").exists(), "refused before anything was created"

    _prepare(ds, write_shape=(2, 32, 32))
    arr = ds.array("r")
    assert arr.chunk_shape == Coordinate(2, 16, 16), "the declared chunk, not the block"
    assert arr._source_data.shards is None


def test_prepare_refuses_a_write_block_with_the_wrong_dimensions(tmp_path):
    ds = Labels(store=tmp_path / "x.zarr")
    with pytest.raises(ValueError, match="one element per axis"):
        _prepare(ds, write_shape=(16, 16))
    ds = Labels(store=tmp_path / "y.zarr", chunk_shape=(16, 16))
    with pytest.raises(ValueError, match="one element per axis"):
        _prepare(ds, write_shape=(2, 16, 16))


@needs_sharding
def test_prepare_shards_at_the_chunk_shape(tmp_path):
    """Under `sharding` the declared chunk_shape is the shard: the write unit stays one knob."""
    ds = Labels(store=tmp_path / "x.zarr", chunk_shape=(2, 32, 32), sharding=True,
                inner_chunk_shape=(2, 8, 8))
    with pytest.raises(ValueError, match="not a whole number of shards"):
        _prepare(ds, write_shape=(2, 16, 16))
    assert not (tmp_path / "x.zarr").exists()

    _prepare(ds, write_shape=(2, 64, 64))
    arr = ds.array("r+")
    assert arr.chunk_shape == arr.shard_shape == Coordinate(2, 32, 32)
    assert arr.inner_chunk_shape == Coordinate(2, 8, 8)

    # a second prepare against the same layout reuses the array ...
    arr[:] = 3
    _prepare(ds, write_shape=(2, 32, 32))
    np.testing.assert_array_equal(ds.array("r")[:], 3)
    # ... a dataset saying nothing about sharding reuses it too (a creation option) ...
    _prepare(Labels(store=tmp_path / "x.zarr", chunk_shape=(2, 32, 32)), write_shape=(2, 32, 32))
    np.testing.assert_array_equal(ds.array("r")[:], 3)
    # ... and another write unit, or another inner chunk, is refused by name
    other = Labels(store=tmp_path / "x.zarr", chunk_shape=(2, 64, 64), sharding=True)
    with pytest.raises(ValueError, match="written in shards of \\(2, 32, 32\\).*shards of \\(2, 64, 64\\)"):
        _prepare(other, write_shape=(2, 64, 64))
    inner = Labels(store=tmp_path / "x.zarr", chunk_shape=(2, 32, 32), sharding=True,
                   inner_chunk_shape=(2, 16, 16))
    with pytest.raises(ValueError, match="inner chunks \\(2, 8, 8\\)"):
        _prepare(inner, write_shape=(2, 32, 32))
    np.testing.assert_array_equal(ds.array("r")[:], 3)


@needs_sharding
def test_prepare_shards_with_funlibs_default_inner_chunk(tmp_path):
    ds = Labels(store=tmp_path / "x.zarr", chunk_shape=(2, 128, 128), sharding=True)
    _prepare(ds, write_shape=(2, 128, 128), shape=(2, 256, 256))
    arr = ds.array("r")
    assert arr.chunk_shape == Coordinate(2, 128, 128)
    assert arr.inner_chunk_shape == Coordinate(2, 64, 64)


def test_prepare_names_a_funlib_that_cannot_shard(tmp_path, monkeypatch):
    import volara.datasets as datasets

    def prepare_ds_without_sharding(store, shape, chunk_shape=None, **kwargs):
        raise AssertionError("must not be reached")

    monkeypatch.setattr(datasets, "prepare_ds", prepare_ds_without_sharding)
    ds = Labels(store=tmp_path / "x.zarr", chunk_shape=(2, 32, 32), sharding=True)
    with pytest.raises(RuntimeError, match="no sharding parameter"):
        _prepare(ds, write_shape=(2, 32, 32))


def _zarr_sharded(store, shape=(2, 64, 64), chunks=(2, 8, 8), shards=(2, 32, 32), fill=0):
    """A zarr v3 SHARDED array with funlib's attrs, built by zarr itself so the tests run on a
    funlib that cannot create one (0.7.0)."""
    import zarr

    z = zarr.create_array(
        store=str(store), shape=shape, chunks=chunks, shards=shards, dtype="uint8", zarr_format=3
    )
    z.attrs.update(
        {"offset": [0, 0], "voxel_size": [1, 1], "axis_names": ["c^", "y", "x"],
         "units": ["nm", "nm"], "types": ["channel", "space", "space"]}
    )
    z[:] = fill
    return z


def test_stored_layout_reads_zarr_itself(tmp_path):
    """Right on a funlib without sharding support: the shard comes off the zarr array."""
    from funlib.persistence import open_ds

    from volara.datasets import stored_layout, stored_unit_of

    _zarr_sharded(tmp_path / "s.zarr")
    arr = open_ds(tmp_path / "s.zarr")
    assert stored_layout(arr) == ((2, 8, 8), (2, 32, 32))
    assert stored_unit_of(arr) == (2, 32, 32)
    prepare_ds(tmp_path / "p.zarr", (8, 8), chunk_shape=(4, 4), mode="w")
    assert stored_layout(open_ds(tmp_path / "p.zarr")) == ((4, 4), None)
    assert stored_unit_of(open_ds(tmp_path / "p.zarr")) == (4, 4)


def test_prepare_judges_an_existing_sharded_array_by_its_write_unit(tmp_path):
    """An existing sharded array met by a dataset declaring nothing: the task's write block is
    the unit. Blocks of the shard cover whole shards and are fine; blocks of the inner chunk
    would share shards and are refused by name, whatever an older funlib would compare."""
    _zarr_sharded(tmp_path / "x.zarr", fill=7)
    ds = Labels(store=tmp_path / "x.zarr")
    with pytest.raises(ValueError, match="written in shards of \\(2, 32, 32\\).*chunks of \\(2, 8, 8\\)"):
        _prepare(ds, write_shape=(2, 8, 8))
    with pytest.raises(ValueError, match="written in shards"):
        _prepare(Labels(store=tmp_path / "x.zarr", chunk_shape=(2, 8, 8)), write_shape=(2, 32, 32))
    np.testing.assert_array_equal(ds.array("r")[:], 7), "nothing touched"
    if _FUNLIB_SHARDS:
        # the whole shard as the block is exactly the safe write
        _prepare(ds, write_shape=(2, 32, 32))
        np.testing.assert_array_equal(ds.array("r")[:], 7)
    else:
        # an old funlib compares the INNER chunk and would refuse with a bare PermissionError;
        # the sharded array is named instead ...
        with pytest.raises(RuntimeError, match="sharded.*funlib.persistence#40"):
            _prepare(ds, write_shape=(2, 32, 32))
        # ... except where the inner chunk IS the shard, which that funlib handles correctly
        _zarr_sharded(tmp_path / "one.zarr", chunks=(2, 32, 32), shards=(2, 32, 32), fill=4)
        one = Labels(store=tmp_path / "one.zarr")
        _prepare(one, write_shape=(2, 32, 32))
        np.testing.assert_array_equal(one.array("r")[:], 4)


@needs_sharding
def test_a_declared_inner_chunk_a_plain_array_has_not_got_is_refused_by_name(tmp_path):
    """funlib compares a declared inner chunk to the array's chunks whether or not it is sharded;
    volara names it, and accepts a restated chunk (the layout-copy idiom)."""
    plain = Labels(store=tmp_path / "x.zarr", chunk_shape=(2, 32, 32))
    _prepare(plain, write_shape=(2, 32, 32))
    asks = Labels(store=tmp_path / "x.zarr", chunk_shape=(2, 32, 32), sharding=True,
                  inner_chunk_shape=(2, 8, 8))
    with pytest.raises(ValueError, match="chunks \\(it is not sharded\\) \\(2, 32, 32\\), not the inner chunks"):
        _prepare(asks, write_shape=(2, 32, 32))


@needs_sharding
def test_a_sharding_dataset_reuses_a_plain_array_with_the_same_unit(tmp_path):
    """`sharding` is a creation option, the other way round too."""
    _prepare(Labels(store=tmp_path / "x.zarr", chunk_shape=(2, 32, 32)), write_shape=(2, 32, 32))
    ds = Labels(store=tmp_path / "x.zarr", chunk_shape=(2, 32, 32), sharding=True)
    _prepare(ds, write_shape=(2, 32, 32))
    assert ds.array("r")._source_data.shards is None, "reused as it is"


def test_prepare_refuses_an_existing_array_at_another_origin(tmp_path):
    """funlib relabels an existing array to the offset it is given; a task whose write roi moved
    would then anchor its blocks off the array's own grid. Compared in voxels, so an array
    relabelled to another voxel size (slabreg's physical-nm canvas) still re-enters."""
    ds = Labels(store=tmp_path / "x.zarr", chunk_shape=(2, 16, 16))
    _prepare(ds, write_shape=(2, 32, 32))
    kw: dict[str, Any] = dict(units=("nm", "nm"), axis_names=("c^", "y", "x"),
                              types=("channel", "space", "space"), dtype=np.uint8)
    with pytest.raises(ValueError, match="starts at voxel \\(0, 0\\), this task would prepare it at \\(16, 0\\)"):
        ds.prepare(shape=(2, 64, 64), chunk_shape=(2, 32, 32), offset=(16, 0), voxel_size=(1, 1), **kw)
    # the same voxel origin under another voxel size is the same array -- at a NONZERO origin,
    # where world and voxel offsets differ (as slabreg's canvas, relabelled to nm)
    ds = Labels(store=tmp_path / "y.zarr", chunk_shape=(2, 16, 16))
    ds.prepare(shape=(2, 64, 64), chunk_shape=(2, 32, 32), offset=(16, 0), voxel_size=(1, 1), **kw)
    prepare_ds(tmp_path / "y.zarr", (2, 64, 64), offset=Coordinate(64, 0), voxel_size=Coordinate(4, 8),
               axis_names=["c^", "y", "x"], chunk_shape=(2, 16, 16), dtype=np.uint8, mode="a")
    ds.prepare(shape=(2, 64, 64), chunk_shape=(2, 32, 32), offset=(16, 0), voxel_size=(1, 1), **kw)
    # a sub-voxel move is a move: not rounded away
    prepare_ds(tmp_path / "y.zarr", (2, 64, 64), offset=Coordinate(66, 0), voxel_size=Coordinate(4, 8),
               axis_names=["c^", "y", "x"], chunk_shape=(2, 16, 16), dtype=np.uint8, mode="a")
    with pytest.raises(ValueError, match="axes \\[0\\] differ"):
        ds.prepare(shape=(2, 64, 64), chunk_shape=(2, 32, 32), offset=(16, 0), voxel_size=(1, 1), **kw)
    # different spatial axes are named as such, not as a moved origin
    with pytest.raises(ValueError, match="spatial axes.*the axes differ"):
        ds.prepare(shape=(2, 64, 64), chunk_shape=(2, 32, 32), offset=(0, 0, 0), voxel_size=(1, 1, 1),
                   units=("nm",) * 3, axis_names=("z", "y", "x"), types=("space",) * 3, dtype=np.uint8)


def test_voxels_leaves_a_world_offset_alone_under_a_zero_voxel_size():
    """The origin comparison in `prepare` works in voxels; an axis with voxel size 0 cannot be
    divided through, so its offset is compared as given. An exact division is an int, an inexact
    one a float, so (6, 3) and (7, 2) never compare equal to each other's voxel count."""
    from volara.datasets import _voxels

    assert _voxels(6, 0) == 6
    assert _voxels(6, 3) == 2 and isinstance(_voxels(6, 3), int)
    assert _voxels(7, 2) == 3.5
