from pathlib import Path

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


def test_drop_refuses_a_non_s3_url_rather_than_silently_skipping():
    """A non-s3 URL must not be coerced to a Path that never exists (a no-op that
    looks like success)."""
    with pytest.raises(ValueError, match="URL this method cannot delete"):
        Raw(store="gs://bucket/x.zarr").drop()


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
