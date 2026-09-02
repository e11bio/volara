from contextlib import contextmanager
from typing import Literal

import daisy
import numpy as np
import pytest
import zarr
from funlib.geometry import Coordinate, Roi

from volara.blockwise.blockwise import _SEED_ATTR, BlockwiseTask
from volara.datasets import MaskDataset
from volara.logging import set_log_basedir


class MaskableTask(BlockwiseTask):
    """Minimal concrete task over a 4x1 block grid; the body touches one file
    per call under ``count_dir`` (daisy may run the body in worker processes,
    so an in-process list would count nothing)."""

    task_type: Literal["maskable"] = "maskable"
    label: str = "maskable-task"
    count_dir: str = ""

    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False

    @property
    def task_name(self) -> str:
        return self.label

    @property
    def write_roi(self) -> Roi:
        return Roi((0, 0), (40, 10))

    @property
    def write_size(self) -> Coordinate:
        return Coordinate(10, 10)

    @property
    def context_size(self) -> Coordinate:
        return Coordinate(0, 0)

    def drop_artifacts(self):
        pass

    def init(self):
        pass

    @contextmanager
    def process_block_func(self):
        count_dir = self.count_dir

        def process_block(block):
            from pathlib import Path as _P

            name = "_".join(str(int(o)) for o in block.write_roi.offset)
            (_P(count_dir) / name).touch()

        yield process_block


def _counted(cls, tmp_path, **kw):
    d = tmp_path / "calls"
    d.mkdir(exist_ok=True)
    return cls(count_dir=str(d), **kw)


def _mask(tmp_path, bits, name="mask.zarr"):
    arr = np.asarray(bits, dtype=np.uint8)
    z = zarr.open(
        str(tmp_path / name),
        mode="w",
        shape=arr.shape,
        chunks=arr.shape,
        dtype="uint8",
    )
    z[:] = arr
    return MaskDataset(store=tmp_path / name)


def _calls(task) -> list[str]:
    from pathlib import Path as _P

    return sorted(p.name for p in _P(task.count_dir).iterdir())


def _run(task) -> int:
    from pathlib import Path as _P

    d = _P(task.count_dir)
    for f in d.iterdir():
        f.unlink()
    with task.task(multiprocessing=False) as t:
        daisy.run_blockwise([t], progress=False)
    return len(_calls(task))


def test_fresh_seed_runs_exactly_the_unmasked_blocks(tmp_path):
    set_log_basedir(tmp_path / "logs")
    t = _counted(
        MaskableTask, tmp_path, block_done_mask=_mask(tmp_path, [[1], [0], [1], [0]])
    )
    assert _run(t) == 2
    assert _calls(t) == ["10_0", "30_0"]
    done = zarr.open(str(t.tracking_path / "done"), mode="r")
    assert np.asarray(done[:]).ravel().tolist() == [1, 1, 1, 1]


def test_a_narrower_mask_unskips_what_it_no_longer_covers(tmp_path):
    set_log_basedir(tmp_path / "logs")
    t = _counted(
        MaskableTask,
        tmp_path,
        block_done_mask=_mask(tmp_path, [[1], [1], [0], [0]], "a.zarr"),
    )
    assert _run(t) == 2  # cells 2, 3 computed for real
    t2 = _counted(
        MaskableTask,
        tmp_path,
        block_done_mask=_mask(tmp_path, [[1], [0], [0], [0]], "b.zarr"),
    )
    assert _run(t2) == 1, (
        "cell 1 was only ever mask-skipped; the narrower mask must re-run it"
    )
    assert _calls(t2) == ["10_0"]


def test_a_daisy_written_store_counts_as_real_completions(tmp_path):
    """A STRICT SUBSET must complete first: with all four done, `_run == 0` holds
    even under the bug where a mask erases real completions, so the assertion
    could not fail on what its name claims."""
    set_log_basedir(tmp_path / "logs")
    # Mask cells 0 and 1 so only 2 and 3 are ever really computed by daisy...
    t = _counted(
        MaskableTask,
        tmp_path,
        block_done_mask=_mask(tmp_path, [[1], [1], [0], [0]], "sub.zarr"),
    )
    assert _run(t) == 2 and _calls(t) == ["20_0", "30_0"]
    # ...then drop to a mask that covers neither: 0 and 1 were only mask-skipped
    # and must re-run, while 2 and 3 were earned and must not.
    t2 = _counted(
        MaskableTask,
        tmp_path,
        block_done_mask=_mask(tmp_path, [[0], [0], [0], [0]], "none.zarr"),
    )
    assert _run(t2) == 2, "the two mask-only cells re-run; the two real ones do not"
    assert _calls(t2) == ["0_0", "10_0"]


def test_a_wrong_shape_mask_is_refused_in_python_naming_the_grid(tmp_path):
    set_log_basedir(tmp_path / "logs")
    t = _counted(
        MaskableTask, tmp_path, block_done_mask=_mask(tmp_path, [[1], [0], [1]])
    )
    with pytest.raises(ValueError, match=r"block grid \(4, 1\)"):
        t._seed_block_done_mask()


def test_a_layout_change_refuses_to_reuse_the_seeded_store(tmp_path):
    set_log_basedir(tmp_path / "logs")
    t = _counted(
        MaskableTask, tmp_path, block_done_mask=_mask(tmp_path, [[1], [0], [1], [0]])
    )
    assert _run(t) == 2

    class ShiftedTask(MaskableTask):
        # same grid shape (4, 1), different world placement: the shape check
        # cannot catch this; only the recorded layout can
        @property
        def write_roi(self) -> Roi:
            return Roi((10, 0), (40, 10))

    t2 = _counted(
        ShiftedTask,
        tmp_path,
        block_done_mask=_mask(tmp_path, [[1], [0], [1], [0]], "c.zarr"),
    )
    with pytest.raises(RuntimeError, match="different task geometry"):
        t2._seed_block_done_mask()


def test_the_seed_record_is_written_alongside_the_done_array(tmp_path):
    set_log_basedir(tmp_path / "logs")
    t = _counted(
        MaskableTask, tmp_path, block_done_mask=_mask(tmp_path, [[0], [1], [0], [0]])
    )
    t._seed_block_done_mask()
    group = zarr.open_group(str(t.tracking_path), mode="r")
    assert np.asarray(group["seed"][:]).ravel().tolist() == [0, 1, 0, 0]
    record = group.attrs[_SEED_ATTR]
    assert record["layout"]["grid_shape"] == [4, 1]
    # `len(...) == 64` holds for ANY hexdigest; pin the digest of THIS mask, so a
    # record that describes a different mask fails.
    import hashlib

    expected = hashlib.sha256(
        np.asarray([[0], [1], [0], [0]], dtype=np.uint8).tobytes()
    )
    assert record["mask_sha256"] == expected.hexdigest()


def test_mask_dataset_survives_the_task_config_round_trip(tmp_path):
    t = MaskableTask(block_done_mask=MaskDataset(store=tmp_path / "m.zarr"))
    round_tripped = MaskableTask.model_validate_json(t.model_dump_json())
    assert round_tripped.block_done_mask is not None
    # Comparing str(...) to str(...) passes whatever the type became; compare the
    # value the way a consumer uses it, and pin the type separately.
    got = round_tripped.block_done_mask.store
    assert got == t.block_done_mask.store
    assert isinstance(got, type(t.block_done_mask.store))


def test_block_grid_slices_anchor_below_the_write_offset_when_context_exceeds_a_block():
    """⛔ The old version of this test used ``context=9 < block=10``, where the
    buggy and correct anchors coincide -- it passed on the bug and enshrined the
    wrong contract. Both regimes are pinned here, and
    ``test_the_grid_anchor_matches_daisys_own_done_array`` checks the answer
    against daisy instead of against our own formula."""

    class NoContext(MaskableTask):
        @property
        def write_roi(self) -> Roi:
            return Roi((100, 0), (40, 10))

        @property
        def context_size(self) -> Coordinate:
            return Coordinate(9, 0)

    t = NoContext()
    # grown total spans [91, 149) -> ceil(58 / 10) = 6 cells in dim 0, and
    # 9 // 10 == 0, so cell 0 still starts at the write offset.
    assert t.block_grid_shape == (6, 1)
    assert t.block_grid_cell_origin() == Coordinate(100, 0)
    assert t.block_grid_slices(Roi((100, 0), (10, 10))) == (slice(0, 1), slice(0, 1))
    assert t.block_grid_slices(Roi((130, 0), (10, 10))) == (slice(3, 4), slice(0, 1))
    assert t.block_grid_slices(Roi((91, 0), (9, 10))) == (slice(0, 0), slice(0, 1))

    class ContextExceedsBlock(MaskableTask):
        @property
        def write_roi(self) -> Roi:
            return Roi((100, 0), (40, 10))

        @property
        def context_size(self) -> Coordinate:
            return Coordinate(25, 0)

    t2 = ContextExceedsBlock()
    # grown total spans [75, 165) -> 9 cells; 25 // 10 == 2, so the block written
    # at 100 is cell 2 and the grid's cell 0 starts at 80.
    assert t2.block_grid_shape == (9, 1)
    assert t2.block_grid_cell_origin() == Coordinate(80, 0)
    assert t2.block_grid_slices(Roi((100, 0), (10, 10))) == (slice(2, 3), slice(0, 1))
    assert t2.block_grid_slices(Roi((130, 0), (10, 10))) == (slice(5, 6), slice(0, 1))


def test_the_grid_anchor_matches_daisys_own_done_array(tmp_path):
    """⛔ The property B2 was about, checked against daisy rather than restated.

    daisy indexes a block as ``(write_roi.offset - total_roi.offset) //
    write_shape`` and volara hands it the CONTEXT-GROWN write ROI, so the cell a
    block lands in shifts by ``context_low // block``. Run a real mask-less task
    and require that every cell daisy marks is one ``block_grid_slices`` predicts
    for that block -- a formula-vs-formula test cannot see this.
    """
    set_log_basedir(tmp_path / "logs")

    class ContextExceedsBlock(MaskableTask):
        @property
        def context_size(self) -> Coordinate:
            return Coordinate(25, 0)

    t = _counted(ContextExceedsBlock, tmp_path)
    assert _run(t) == 4
    done = np.asarray(zarr.open(str(t.tracking_path / "done"), mode="r")[:]).ravel()

    predicted = np.zeros_like(done)
    for name in _calls(t):
        off = Coordinate(int(x) for x in name.split("_"))
        sl = t.block_grid_slices(Roi(off, t.write_size))
        predicted[sl[0]] = 1
    assert predicted.tolist() == done.tolist(), (
        f"block_grid_slices predicts {predicted.tolist()} but daisy marked "
        f"{done.tolist()} -- the grid anchor disagrees with daisy"
    )
    # And the shift is real, not a no-op: 25 // 10 == 2 cells.
    assert done.tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 0]


def test_dropping_the_mask_unskips_what_it_had_seeded(tmp_path):
    """⛔ B1. Seeding only when a mask is present leaves the last run's skip bits
    standing, and daisy honours them: the narrowest mask of all -- removing the
    field -- would be the one case that never un-skips. Measured before the fix:
    the mask-less rerun logged "resumed -- 1/4 blocks skipped" and left the
    masked region unwritten with failed=0."""
    set_log_basedir(tmp_path / "logs")
    t = _counted(
        MaskableTask, tmp_path, block_done_mask=_mask(tmp_path, [[1], [1], [0], [0]])
    )
    assert _run(t) == 2 and _calls(t) == ["20_0", "30_0"]

    t2 = _counted(MaskableTask, tmp_path)  # same meta dir, NO mask
    assert _run(t2) == 2, "the two mask-only cells must run once the mask is gone"
    assert _calls(t2) == ["0_0", "10_0"]


def test_a_store_volara_never_seeded_is_left_alone(tmp_path):
    """The other half of B1's fix: an unmasked task on an unmasked store must not
    write anything, must not need a capable daisy, and must resume normally."""
    set_log_basedir(tmp_path / "logs")
    t = _counted(MaskableTask, tmp_path)
    assert _run(t) == 4
    group = zarr.open_group(str(t.tracking_path), mode="r")
    assert _SEED_ATTR not in group.attrs and "seed" not in group
    assert _run(t) == 0, "daisy's own markers still resume"


def test_a_layout_change_refuses_even_without_a_mask(tmp_path):
    """⛔ B3. A store volara seeded never gets daisy's own ``daisy_task_hash``
    (volara creates the group, and daisy only stamps groups it creates), and a
    hash-less store is accepted for ANY layout -- so volara's stamp is that
    store's only layout identity. Checking it only on masked runs leaves the
    mask-less path free to reuse another geometry's done bits."""
    set_log_basedir(tmp_path / "logs")
    t = _counted(
        MaskableTask, tmp_path, block_done_mask=_mask(tmp_path, [[1], [0], [1], [0]])
    )
    assert _run(t) == 2

    class ShiftedTask(MaskableTask):
        @property
        def write_roi(self) -> Roi:
            return Roi((10, 0), (40, 10))

    t2 = _counted(
        ShiftedTask, tmp_path
    )  # same grid shape, different placement, NO mask
    with pytest.raises(RuntimeError, match="different task geometry"):
        t2._seed_block_done_mask()


def test_widening_then_narrowing_keeps_the_real_completions(tmp_path):
    """⛔ N1. Recording the whole new mask as seeded overwrites the bits a worker
    earned, so a later narrowing re-runs completed blocks. Safe in direction,
    but it can cost a full pass."""
    set_log_basedir(tmp_path / "logs")
    t = _counted(
        MaskableTask,
        tmp_path,
        block_done_mask=_mask(tmp_path, [[1], [1], [0], [0]], "w1.zarr"),
    )
    assert _run(t) == 2 and _calls(t) == ["20_0", "30_0"]  # 2 and 3 are REAL

    t2 = _counted(
        MaskableTask,
        tmp_path,
        block_done_mask=_mask(tmp_path, [[1], [1], [1], [1]], "w2.zarr"),
    )
    assert _run(t2) == 0  # everything masked

    t3 = _counted(
        MaskableTask,
        tmp_path,
        block_done_mask=_mask(tmp_path, [[0], [0], [0], [0]], "w3.zarr"),
    )
    assert _run(t3) == 2, "cells 2 and 3 were earned; only the mask-only cells re-run"
    assert _calls(t3) == ["0_0", "10_0"]


def test_an_interrupted_narrowing_never_promotes_a_seed_to_a_completion(
    tmp_path, monkeypatch
):
    """⛔ The resume invariant is ``done & ~seed == real``, and the write ORDER is
    what preserves it under interruption. Narrowing ``seed`` before ``done``
    breaks it in exactly the direction this feature exists for: cell 0 is
    mask-skipped, the mask is then dropped, and a failure anywhere in the window
    leaves ``done=[1,1,1,1] seed=[0,0,0,0]`` -- cell 0 a permanent completion no
    worker ever ran, reported with ``failed=0``.

    Each write is interrupted in turn by making it raise, which is what an ENOSPC
    or a killed driver looks like from the store's side; the next run must still
    recover cell 0.

    ⚠️ This replaces an ``inspect.getsource`` index comparison that passed on the
    bug and on the fix alike -- it pinned the source text, not the behaviour.
    """
    from volara.blockwise.blockwise import BlockwiseTask

    # On the class a staticmethod is already the plain function.
    real_write = BlockwiseTask._write_grid_array

    for stop_after in range(3):
        base = tmp_path / f"crash{stop_after}"
        base.mkdir()
        set_log_basedir(base / "logs")
        t = _counted(
            MaskableTask, base, block_done_mask=_mask(base, [[1], [0], [0], [0]])
        )
        assert _run(t) == 3 and _calls(t) == ["10_0", "20_0", "30_0"]

        done_writes = {"n": 0}

        def flaky(group, name, values, _stop=stop_after, _n=done_writes):
            if _n["n"] >= _stop:
                raise OSError(28, "No space left on device")
            _n["n"] += 1
            real_write(group, name, values)

        t2 = _counted(MaskableTask, base)  # the mask is DROPPED -- the narrowest of all
        with monkeypatch.context() as mp:
            mp.setattr(BlockwiseTask, "_write_grid_array", staticmethod(flaky))
            with pytest.raises(OSError):
                t2._seed_block_done_mask()

        t3 = _counted(MaskableTask, base)
        _run(t3)
        assert "0_0" in _calls(t3), (
            f"interrupted after {stop_after} write(s): cell 0 was mask-skipped and "
            f"never computed, yet the store reports it done"
        )


def test_a_mask_arriving_on_a_daisy_owned_store_keeps_its_completions(tmp_path):
    """The ``prior is None`` branch -- a store daisy wrote, so every bit in it was
    earned. Ordering matters: the first run must be UNMASKED, or ``prior`` is set
    and this exercises ``cur & ~prev_seed`` instead. (An earlier version of this
    file named this branch in a test that never reached it.)"""
    set_log_basedir(tmp_path / "logs")
    t = _counted(MaskableTask, tmp_path)
    assert _run(t) == 4  # daisy writes its own store
    group = zarr.open_group(str(t.tracking_path), mode="r")
    assert _SEED_ATTR not in group.attrs, "precondition: daisy owns this store"

    t2 = _counted(
        MaskableTask, tmp_path, block_done_mask=_mask(tmp_path, [[1], [0], [0], [0]])
    )
    assert _run(t2) == 0, "a mask must not erase completions daisy recorded"


def test_a_daisy_owned_store_is_not_bricked_by_a_refused_masked_run(tmp_path):
    """⛔ Where daisy's own ``daisy_task_hash`` is present, daisy validates the
    geometry and is the authority; volara's stamp is for the hash-less stores
    daisy accepts for ANY layout. Honouring our stamp on a daisy-owned store
    means a masked run at the wrong geometry writes the stamp, daisy refuses and
    runs nothing, and every later run of the ORIGINAL task refuses on our stale
    stamp -- a full recompute owed to a mistake that computed nothing."""
    set_log_basedir(tmp_path / "logs")
    t = _counted(MaskableTask, tmp_path)
    assert _run(t) == 4

    class ShiftedTask(MaskableTask):
        @property
        def write_roi(self) -> Roi:
            return Roi((10, 0), (40, 10))

    t2 = _counted(
        ShiftedTask,
        tmp_path,
        block_done_mask=_mask(tmp_path, [[1], [0], [0], [0]], "s.zarr"),
    )
    t2._seed_block_done_mask()  # volara must not refuse; daisy owns the layout here

    t3 = _counted(MaskableTask, tmp_path)
    assert _run(t3) == 0, "the original task must still resume on its own completions"
