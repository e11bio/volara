import numpy as np
from funlib.geometry import Coordinate
from funlib.persistence.arrays import prepare_ds

from volara.blockwise import Argmax
from volara.datasets import Labels, Raw


def _make_probs(tmp_path, shape, values_per_channel=None):
    """Helper to create a multi-channel probability array."""
    if values_per_channel is not None:
        data = np.zeros(shape, dtype=np.float32)
        for i, v in enumerate(values_per_channel):
            data[i] = v
    else:
        data = np.arange(1, int(np.prod(shape)) + 1, dtype=np.float32).reshape(shape)
    path = tmp_path / "data.zarr" / "probs"
    prepare_ds(
        path,
        shape=data.shape,
        voxel_size=Coordinate(*(1,) * (len(shape) - 1)),
        dtype=data.dtype,
        mode="w",
    )[:] = data
    return path, data


def test_argmax_basic(tmp_path, block_2d):
    """Channel 1 always > channel 0 -> argmax is 1 everywhere."""
    probs_path, _ = _make_probs(tmp_path, (2, 10, 10))
    # Values 1..100 in ch0, 101..200 in ch1 -> ch1 always wins
    sem_path = tmp_path / "data.zarr" / "labels"
    prepare_ds(
        sem_path,
        shape=(10, 10),
        voxel_size=Coordinate(1, 1),
        dtype=np.uint32,
        mode="w",
    )

    task = Argmax(
        probs_data=Raw(store=probs_path),
        sem_data=Labels(store=sem_path),
        block_size=Coordinate(10, 10),
    )
    with task.process_block_func() as process_block:
        process_block(block_2d)

    result = Labels(store=sem_path).array("r")[:]
    assert np.all(result == 1)


def test_argmax_combine_classes(tmp_path, block_2d):
    """combine_classes sums channels before argmax."""
    # ch0=1, ch1=2, ch2=4 -> group[0,1]=3, group[2]=4 -> argmax=1
    probs_path, _ = _make_probs(
        tmp_path, (3, 10, 10), values_per_channel=[1.0, 2.0, 4.0]
    )
    sem_path = tmp_path / "data.zarr" / "labels"
    prepare_ds(
        sem_path,
        shape=(10, 10),
        voxel_size=Coordinate(1, 1),
        dtype=np.uint8,
        mode="w",
    )

    task = Argmax(
        probs_data=Raw(store=probs_path),
        sem_data=Labels(store=sem_path),
        block_size=Coordinate(10, 10),
        combine_classes=[[0, 1], [2]],
    )
    with task.process_block_func() as process_block:
        process_block(block_2d)

    result = Labels(store=sem_path).array("r")[:]
    # group 0: 1+2=3, group 1: 4 -> argmax picks 1
    assert np.all(result == 1)


def test_argmax_multiblock(blocks_20x10, tmp_path):
    """Two blocks tile a (2,20,10) array correctly."""
    probs_path, _ = _make_probs(tmp_path, (2, 20, 10), values_per_channel=[1.0, 2.0])
    sem_path = tmp_path / "data.zarr" / "labels"
    prepare_ds(
        sem_path,
        shape=(20, 10),
        voxel_size=Coordinate(1, 1),
        dtype=np.uint8,
        mode="w",
    )

    task = Argmax(
        probs_data=Raw(store=probs_path),
        sem_data=Labels(store=sem_path),
        block_size=Coordinate(10, 10),
    )

    with task.process_block_func() as process_block:
        for b in blocks_20x10:
            process_block(b)

    result = Labels(store=sem_path).array("r")[:]
    assert result.shape == (20, 10)
    assert np.all(result == 1)
