from contextlib import contextmanager
from typing import Literal

from funlib.geometry import Coordinate, Roi

from volara.blockwise.blockwise import BlockwiseTask
from volara.blockwise.pipeline import Pipeline


class DummyTask(BlockwiseTask):
    """Minimal concrete subclass for testing the ABC."""

    task_type: Literal["dummy"] = "dummy"
    label: str = "dummy-task"

    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False

    @property
    def task_name(self) -> str:
        return self.label

    @property
    def write_roi(self) -> Roi:
        return Roi((0, 0), (10, 10))

    @property
    def write_size(self) -> Coordinate:
        return Coordinate(10, 10)

    @property
    def context_size(self) -> Coordinate:
        return Coordinate(0, 0)

    def drop_artifacts(self):
        pass

    @contextmanager
    def process_block_func(self):
        def process_block(block):
            pass

        yield process_block


def test_dummy_task_properties():
    t = DummyTask()
    assert t.task_name == "dummy-task"
    assert t.write_roi == Roi((0, 0), (10, 10))
    assert t.write_size == Coordinate(10, 10)
    assert t.context_size == Coordinate(0, 0)


def test_meta_dir_and_config_file(tmp_path):
    from volara.logging import set_log_basedir

    set_log_basedir(tmp_path / "logs")
    t = DummyTask()
    assert t.meta_dir == tmp_path / "logs" / "dummy-task-meta"
    assert t.config_file == tmp_path / "logs" / "dummy-task-meta" / "config.json"


def test_drop_removes_meta_dir(tmp_path):
    from volara.logging import set_log_basedir

    set_log_basedir(tmp_path / "logs")
    t = DummyTask()
    t.meta_dir.mkdir(parents=True)
    assert t.meta_dir.exists()
    t.drop()
    assert not t.meta_dir.exists()


DROP_ARTIFACTS_CALLS: list[str] = []


class RecordingTask(DummyTask):
    """DummyTask that records every ``drop_artifacts`` call."""

    def drop_artifacts(self):
        DROP_ARTIFACTS_CALLS.append(self.task_name)


def test_drop_calls_drop_artifacts_by_default(tmp_path):
    """drop() is a full reset: meta dir removed and outputs dropped."""
    from volara.logging import set_log_basedir

    set_log_basedir(tmp_path / "logs")
    DROP_ARTIFACTS_CALLS.clear()
    t = RecordingTask()
    t.meta_dir.mkdir(parents=True)
    t.drop()
    assert not t.meta_dir.exists()
    assert DROP_ARTIFACTS_CALLS == [t.task_name]


def test_drop_outputs_false_skips_drop_artifacts(tmp_path):
    """drop(drop_outputs=False) clears the meta dir but keeps the outputs."""
    from volara.logging import set_log_basedir

    set_log_basedir(tmp_path / "logs")
    DROP_ARTIFACTS_CALLS.clear()
    t = RecordingTask()
    t.meta_dir.mkdir(parents=True)
    t.drop(drop_outputs=False)
    assert not t.meta_dir.exists()
    assert DROP_ARTIFACTS_CALLS == []


def test_init_block_array(tmp_path):
    from volara.logging import set_log_basedir

    set_log_basedir(tmp_path / "logs")
    t = DummyTask()
    t.init_block_array()
    assert t.block_ds.exists()


def test_pipeline_add_operator():
    t1 = DummyTask(label="task-a")
    t2 = DummyTask(label="task-b")
    p = t1 + t2
    assert isinstance(p, Pipeline)
    assert len(p.task_graph.nodes()) == 2
    assert len(p.task_graph.edges()) == 1


def test_pipeline_or_operator():
    t1 = DummyTask(label="task-a")
    t2 = DummyTask(label="task-b")
    p = t1 | t2
    assert isinstance(p, Pipeline)
    assert len(p.task_graph.nodes()) == 2
    assert len(p.task_graph.edges()) == 0


def test_process_block_func_context_manager():
    t = DummyTask()
    with t.process_block_func() as process_block:
        assert callable(process_block)


def test_serialization_roundtrip():
    t = DummyTask()
    json_str = t.model_dump_json()
    t2 = DummyTask.model_validate_json(json_str)
    assert t2.task_name == t.task_name


def test_check_block_func(tmp_path):
    """mark_block_done then check_block should detect the block as done."""
    import daisy

    from volara.logging import set_log_basedir

    set_log_basedir(tmp_path / "logs")
    t = DummyTask()
    t.init_block_array()

    block = daisy.Block(
        total_roi=Roi((0, 0), (10, 10)),
        read_roi=Roi((0, 0), (10, 10)),
        write_roi=Roi((0, 0), (10, 10)),
        block_id=0,
    )

    check_block = t.check_block_func()
    mark_done = t.mark_block_done_func()

    assert not check_block(block)
    mark_done(block)
    assert check_block(block)
