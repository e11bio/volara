import pytest
from funlib.geometry import Coordinate

from volara.blockwise import Threshold
from volara.datasets import Labels, Raw


@pytest.mark.parametrize("drop_outputs", [True, False])
def test_pipeline_drop_forwards_drop_outputs_to_every_task(
    zarr_2d, tmp_path, drop_outputs
):
    raw_path, _ = zarr_2d
    tasks = [
        Threshold(
            in_data=Raw(store=raw_path),
            mask=Labels(store=tmp_path / "test.zarr" / name),
            threshold=threshold,
            block_size=Coordinate(10, 10),
        )
        for name, threshold in [("mask1", 0.5), ("mask2", 0.3)]
    ]
    for task in tasks:
        task.init()
        task.meta_dir.mkdir(parents=True)

    (tasks[0] + tasks[1]).drop(drop_outputs=drop_outputs)

    kept = not drop_outputs
    assert [task.mask.store.exists() for task in tasks] == [kept, kept]
    assert [task.meta_dir.exists() for task in tasks] == [False, False]
