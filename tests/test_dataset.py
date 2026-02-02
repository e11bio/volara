import pytest
import yaml
import numpy as np
import zarr
from pathlib import Path
from funlib.geometry import Coordinate

from volara.datasets import Raw, Affs, LSD, Labels
from pydantic import ValidationError


@pytest.fixture
def zarr_store(tmp_path):
    """Creates a temporary Zarr array for testing."""
    path = tmp_path / "test_data.zarr"
    store = zarr.open(
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
    store = zarr.open(
        str(path), mode="w", shape=(2, 10, 10), chunks=(1, 5, 5), dtype="float32"
    )
    store[:] = np.full((2, 10, 10), 2.0)  # Fill with 2s
    return path


# ==========================================
# 2. Test Cases
# ==========================================

# --- Test Category 1: Serialization / Deserialization ---


def test_raw_serialization_roundtrip(tmp_path):
    """
    Test that YAML -> Python A -> Python B ensures equality.
    Checks handling of tuple/list conversion in Pydantic.
    """
    yaml_data = f"""
    dataset_type: raw
    store: {tmp_path / "test.zarr"}
    voxel_size: [10, 10, 10]
    scale_shift: [0.5, 10.0]
    channels: 0
    """

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

    ds = Raw(store=zarr_store, channels=[0, [0,2,4,6,8]])

    arr = ds.array()
    data = arr[:]

    assert data.shape == (5, 10)
    assert np.all(data == 1.0)  # Original data was 1s

    ds = Raw(store=zarr_store, channels=[0, 0, [0,2,4,6,8]])

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
    ome = zarr.open(str(ome_path), mode="w")
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
        Affs(store=path, neighborhood=[[5, 5]])


def test_attrs_labels_lsd():
    """Simple check for static attributes in subclasses."""
    l = Labels(store="dummy")
    assert l.attrs == {}

    lsd = LSD(store="dummy")
    assert lsd.attrs == {"lsds": True}
