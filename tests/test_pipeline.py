from funlib.geometry import Coordinate

from volara.blockwise import Threshold
from volara.blockwise.pipeline import Pipeline
from volara.datasets import Labels, Raw


def test_pipeline_sequential(zarr_2d, tmp_path):
    """Threshold + Threshold in sequence creates a pipeline with an edge."""
    raw_path, _ = zarr_2d
    task1 = Threshold(
        in_data=Raw(store=raw_path),
        mask=Labels(store=tmp_path / "test.zarr" / "mask1"),
        threshold=0.5,
        block_size=Coordinate(10, 10),
    )
    task2 = Threshold(
        in_data=Raw(store=raw_path),
        mask=Labels(store=tmp_path / "test.zarr" / "mask2"),
        threshold=0.3,
        block_size=Coordinate(10, 10),
    )

    pipeline = task1 + task2
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.task_graph.nodes()) == 2
    assert len(pipeline.task_graph.edges()) == 1


def test_pipeline_parallel(zarr_2d, tmp_path):
    """Threshold | Threshold creates parallel graph (no edges)."""
    raw_path, _ = zarr_2d
    task1 = Threshold(
        in_data=Raw(store=raw_path),
        mask=Labels(store=tmp_path / "test.zarr" / "mask1"),
        threshold=0.5,
        block_size=Coordinate(10, 10),
    )
    task2 = Threshold(
        in_data=Raw(store=raw_path),
        mask=Labels(store=tmp_path / "test.zarr" / "mask2"),
        threshold=0.3,
        block_size=Coordinate(10, 10),
    )

    pipeline = task1 | task2
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.task_graph.nodes()) == 2
    assert len(pipeline.task_graph.edges()) == 0


def test_pipeline_drop(zarr_2d, tmp_path):
    """drop() propagates to all tasks in the pipeline."""
    raw_path, _ = zarr_2d
    mask1_path = tmp_path / "test.zarr" / "mask1"
    mask2_path = tmp_path / "test.zarr" / "mask2"

    task1 = Threshold(
        in_data=Raw(store=raw_path),
        mask=Labels(store=mask1_path),
        threshold=0.5,
        block_size=Coordinate(10, 10),
    )
    task2 = Threshold(
        in_data=Raw(store=raw_path),
        mask=Labels(store=mask2_path),
        threshold=0.3,
        block_size=Coordinate(10, 10),
    )

    task1.init()
    task2.init()
    assert mask1_path.exists()
    assert mask2_path.exists()

    pipeline = task1 + task2
    pipeline.drop()
    assert not mask1_path.exists()
    assert not mask2_path.exists()


def test_pipeline_drop_outputs_false(zarr_2d, tmp_path):
    """Pipeline.drop forwards drop_outputs to every task."""
    raw_path, _ = zarr_2d
    mask1_path = tmp_path / "test.zarr" / "mask1"
    mask2_path = tmp_path / "test.zarr" / "mask2"

    task1 = Threshold(
        in_data=Raw(store=raw_path),
        mask=Labels(store=mask1_path),
        threshold=0.5,
        block_size=Coordinate(10, 10),
    )
    task2 = Threshold(
        in_data=Raw(store=raw_path),
        mask=Labels(store=mask2_path),
        threshold=0.3,
        block_size=Coordinate(10, 10),
    )

    for task in (task1, task2):
        task.init()
        task.init_block_array()
    assert mask1_path.exists()
    assert mask2_path.exists()

    pipeline = task1 + task2
    pipeline.drop(drop_outputs=False)
    assert mask1_path.exists()
    assert mask2_path.exists()
    assert not task1.meta_dir.exists()
    assert not task2.meta_dir.exists()
