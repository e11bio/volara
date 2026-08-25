"""BASE-vs-HEAD behaviour parity probe for volara Dataset lazy ops.

Run as:  cd <worktree> && PYTHONPATH=. python probe.py <outfile.json>
"""

import hashlib
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import zarr
from funlib.persistence import prepare_ds

import volara

TREE = volara.__file__
assert TREE.startswith("/home/jeff/.claude/jobs/b9887beb/tmp/"), TREE

from volara.datasets import Affs, Labels, Raw  # noqa: E402

ROOT = Path(tempfile.mkdtemp(prefix="parity_"))


def mkstore(name, shape, dtype=np.uint16):
    p = ROOT / f"{name}.zarr"
    if p.exists():
        shutil.rmtree(p)
    arr = prepare_ds(
        p,
        shape=shape,
        dtype=dtype,
        voxel_size=(1,) * len(shape),
        offset=(0,) * len(shape),
        axis_names=list("tczyx"[-len(shape):]),
        units=["nm"] * len(shape),
        mode="w",
    )
    data = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    arr[:] = data
    return p, data


def mkome(name, n_channels, lo=10.0, hi=110.0):
    p = ROOT / f"{name}_meta.zarr"
    if p.exists():
        shutil.rmtree(p)
    g = zarr.open_group(str(p), mode="w")
    g.attrs["omero"] = {
        "channels": [
            {"window": {"min": lo + i, "max": hi + i * 2}} for i in range(n_channels)
        ]
    }
    return p


def _try(fn, default="<raised>"):
    try:
        v = fn()
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v]
        return str(v)
    except Exception as e:
        return f"{default}:{type(e).__name__}"


def describe(arr):
    rec = {}
    try:
        a = np.asarray(arr[:])
        rec["sha"] = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:24]
        rec["dtype"] = str(a.dtype)
        rec["shape"] = list(a.shape)
        rec["first8"] = [float(x) for x in a.ravel()[:8].tolist()]
        rec["sum"] = float(np.nansum(a.astype(np.float64)))
    except Exception as e:
        rec["materialize_error"] = f"{type(e).__name__}: {e}"
    rec["arr_shape"] = _try(lambda: list(arr.shape))
    rec["offset"] = _try(lambda: list(arr.offset))
    rec["voxel_size"] = _try(lambda: list(arr.voxel_size))
    rec["axis_names"] = _try(lambda: list(arr.axis_names))
    rec["units"] = _try(lambda: list(arr.units))
    rec["is_writeable"] = _try(lambda: arr.is_writeable)
    rec["roi"] = _try(lambda: arr.roi)
    rec["n_lazy_ops"] = _try(lambda: len(arr.lazy_ops))
    rec["data_dtype"] = _try(lambda: arr.data.dtype)
    return rec


RESULTS = {}


def probe(name):
    def deco(fn):
        try:
            out = fn()
            RESULTS[name] = {"ok": True, "value": out}
        except Exception as e:
            RESULTS[name] = {
                "ok": False,
                "exc_type": type(e).__name__,
                "exc_msg": str(e).replace(str(ROOT), "<TMP>"),
                "where": traceback.format_exc().strip().splitlines()[-3:-1],
            }
        return fn

    return deco


def arr_of(**kw):
    mode = kw.pop("_mode", "r")
    return describe(Raw(**kw).array(mode))


# ------------------------------------------------------------------ plain / channels
@probe("01_nothing_set")
def _():
    p, _d = mkstore("s01", (5, 4, 3))
    return arr_of(store=p, voxel_size=(1, 1, 1), offset=(0, 0, 0))


@probe("02_channels_bare_int")
def _():
    p, _d = mkstore("s02", (3, 5, 4, 3))
    return arr_of(store=p, channels=0, voxel_size=(1,) * 4, offset=(0,) * 4)


@probe("03_channels_list_of_ints")
def _():
    p, _d = mkstore("s03", (2, 3, 5, 4, 3))
    return arr_of(store=p, channels=[1, 2], voxel_size=(1,) * 5, offset=(0,) * 5)


@probe("04_channels_nested_single")
def _():
    p, _d = mkstore("s04", (4, 5, 4, 3))
    return arr_of(store=p, channels=[[0, 2]], voxel_size=(1,) * 4, offset=(0,) * 4)


@probe("05_channels_nested_repeated")
def _():
    p, _d = mkstore("s05", (4, 5, 4, 3))
    return arr_of(store=p, channels=[[0, 1], [1]], voxel_size=(1,) * 4, offset=(0,) * 4)


@probe("06_channels_mixed_nested_and_int")
def _():
    p, _d = mkstore("s06", (2, 4, 5, 4, 3))
    return arr_of(store=p, channels=[1, [0, 2]], voxel_size=(1,) * 5, offset=(0,) * 5)


@probe("07_channels_out_of_range")
def _():
    p, _d = mkstore("s07", (3, 5, 4, 3))
    return arr_of(store=p, channels=99, voxel_size=(1,) * 4, offset=(0,) * 4)


# ------------------------------------------------------------------ flip
@probe("10_flip_single_axis")
def _():
    p, _d = mkstore("s10", (5, 4, 3))
    return arr_of(store=p, flip=[0], voxel_size=(1, 1, 1), offset=(0, 0, 0))


@probe("11_flip_two_axes")
def _():
    p, _d = mkstore("s11", (5, 4, 3))
    return arr_of(store=p, flip=[0, 2], voxel_size=(1, 1, 1), offset=(0, 0, 0))


@probe("12_flip_empty_list")
def _():
    p, _d = mkstore("s12", (5, 4, 3))
    return arr_of(store=p, flip=[], voxel_size=(1, 1, 1), offset=(0, 0, 0))


@probe("13_flip_plus_channels")
def _():
    p, _d = mkstore("s13", (2, 3, 5, 4, 3))
    return arr_of(store=p, channels=[1, 2], flip=[0], voxel_size=(1,) * 5, offset=(0,) * 5)


@probe("14_flip_out_of_range_mode_r")
def _():
    p, _d = mkstore("s14", (5, 4, 3))
    return arr_of(store=p, flip=[7], voxel_size=(1, 1, 1), offset=(0, 0, 0))


@probe("15_flip_out_of_range_after_channels_mode_r")
def _():
    p, _d = mkstore("s15", (3, 5, 4, 3))
    return arr_of(store=p, channels=0, flip=[3], voxel_size=(1,) * 4, offset=(0,) * 4)


@probe("16_flip_negative_axis_mode_r")
def _():
    p, _d = mkstore("s16", (5, 4, 3))
    return arr_of(store=p, flip=[-1], voxel_size=(1, 1, 1), offset=(0, 0, 0))


@probe("17_flip_mode_a")
def _():
    p, _d = mkstore("s17", (5, 4, 3))
    return arr_of(store=p, flip=[0], voxel_size=(1, 1, 1), offset=(0, 0, 0), _mode="a")


@probe("18_flip_mode_w")
def _():
    p, _d = mkstore("s18", (5, 4, 3))
    return arr_of(store=p, flip=[0], voxel_size=(1, 1, 1), offset=(0, 0, 0), _mode="w")


@probe("19_flip_mode_rplus")
def _():
    p, _d = mkstore("s19", (5, 4, 3))
    return arr_of(store=p, flip=[0], voxel_size=(1, 1, 1), offset=(0, 0, 0), _mode="r+")


@probe("20_ADVISOR5_flip_plus_bad_channels_mode_a")
def _():
    """Exception-ORDER probe: bad channels + flip + write mode."""
    p, _d = mkstore("s20", (3, 5, 4, 3))
    return arr_of(store=p, channels=99, flip=[0], voxel_size=(1,) * 4, offset=(0,) * 4, _mode="a")


@probe("21_ADVISOR5_flip_plus_broken_omenorm_mode_a")
def _():
    """Exception-ORDER probe: ome_norm that cannot load + flip + write mode."""
    p, _d = mkstore("s21", (3, 5, 4, 3))
    return arr_of(
        store=p,
        ome_norm=ROOT / "does_not_exist_meta.zarr",
        flip=[0],
        voxel_size=(1,) * 4,
        offset=(0,) * 4,
        _mode="a",
    )


@probe("22_ADVISOR5_flip_out_of_range_plus_mode_a")
def _():
    p, _d = mkstore("s22", (5, 4, 3))
    return arr_of(store=p, flip=[7], voxel_size=(1, 1, 1), offset=(0, 0, 0), _mode="a")


# ------------------------------------------------------------------ scale_shift
@probe("30_scale_shift_positive")
def _():
    p, _d = mkstore("s30", (5, 4, 3))
    return arr_of(store=p, scale_shift=(0.5, 3.0), voxel_size=(1, 1, 1), offset=(0, 0, 0))


@probe("31_scale_shift_negative")
def _():
    p, _d = mkstore("s31", (5, 4, 3))
    return arr_of(store=p, scale_shift=(-2.0, -7.5), voxel_size=(1, 1, 1), offset=(0, 0, 0))


@probe("32_scale_shift_zero_scale")
def _():
    p, _d = mkstore("s32", (5, 4, 3))
    return arr_of(store=p, scale_shift=(0.0, 0.0), voxel_size=(1, 1, 1), offset=(0, 0, 0))


@probe("33_scale_shift_plus_channels")
def _():
    p, _d = mkstore("s33", (3, 5, 4, 3))
    return arr_of(
        store=p, scale_shift=(0.25, 1.0), channels=1, voxel_size=(1,) * 4, offset=(0,) * 4
    )


@probe("34_scale_shift_plus_channels_plus_flip")
def _():
    p, _d = mkstore("s34", (3, 5, 4, 3))
    return arr_of(
        store=p,
        scale_shift=(0.25, 1.0),
        channels=1,
        flip=[0],
        voxel_size=(1,) * 4,
        offset=(0,) * 4,
    )


# ------------------------------------------------------------------ ome_norm
@probe("40_ome_norm_exact_channel_count")
def _():
    p, _d = mkstore("s40", (3, 5, 4, 3))
    m = mkome("s40", 3)
    return arr_of(store=p, ome_norm=m, voxel_size=(1,) * 4, offset=(0,) * 4)


@probe("41_ome_norm_metadata_has_MORE_channels")
def _():
    p, _d = mkstore("s41", (2, 5, 4, 3))
    m = mkome("s41", 5)
    return arr_of(store=p, ome_norm=m, voxel_size=(1,) * 4, offset=(0,) * 4)


@probe("42_ADVISOR6_ome_norm_metadata_has_FEWER_channels")
def _():
    p, _d = mkstore("s42", (4, 5, 4, 3))
    m = mkome("s42", 2)
    return arr_of(store=p, ome_norm=m, voxel_size=(1,) * 4, offset=(0,) * 4)


@probe("43_ome_norm_plus_channels")
def _():
    p, _d = mkstore("s43", (3, 5, 4, 3))
    m = mkome("s43", 3)
    return arr_of(store=p, ome_norm=m, channels=1, voxel_size=(1,) * 4, offset=(0,) * 4)


@probe("44_ome_norm_plus_scale_shift")
def _():
    p, _d = mkstore("s44", (3, 5, 4, 3))
    m = mkome("s44", 3)
    return arr_of(
        store=p, ome_norm=m, scale_shift=(2.0, 1.0), voxel_size=(1,) * 4, offset=(0,) * 4
    )


@probe("45_ome_norm_str_path")
def _():
    p, _d = mkstore("s45", (3, 5, 4, 3))
    m = mkome("s45", 3)
    return arr_of(store=p, ome_norm=str(m), voxel_size=(1,) * 4, offset=(0,) * 4)


@probe("46_ome_norm_attrs_property")
def _():
    p, _d = mkstore("s46", (2, 5, 4, 3))
    m = mkome("s46", 5)
    return str(Raw(store=p, ome_norm=m, voxel_size=(1,) * 4, offset=(0,) * 4).attrs)


# ------------------------------------------------------------------ stack
@probe("50_stack_only")
def _():
    p, _d = mkstore("s50a", (2, 5, 4, 3))
    q, _e = mkstore("s50b", (3, 5, 4, 3))
    other = Raw(store=q, voxel_size=(1,) * 4, offset=(0,) * 4)
    return arr_of(store=p, stack=other, voxel_size=(1,) * 4, offset=(0,) * 4)


@probe("51_stack_plus_channels")
def _():
    p, _d = mkstore("s51a", (2, 5, 4, 3))
    q, _e = mkstore("s51b", (3, 5, 4, 3))
    other = Raw(store=q, voxel_size=(1,) * 4, offset=(0,) * 4)
    return arr_of(store=p, stack=other, channels=[[0, 3]], voxel_size=(1,) * 4, offset=(0,) * 4)


@probe("52_stack_plus_scale_shift")
def _():
    p, _d = mkstore("s52a", (2, 5, 4, 3))
    q, _e = mkstore("s52b", (3, 5, 4, 3))
    other = Raw(store=q, voxel_size=(1,) * 4, offset=(0,) * 4)
    return arr_of(
        store=p, stack=other, scale_shift=(2.0, 0.0), voxel_size=(1,) * 4, offset=(0,) * 4
    )


@probe("53_stack_with_channels_on_the_other")
def _():
    p, _d = mkstore("s53a", (2, 5, 4, 3))
    q, _e = mkstore("s53b", (2, 3, 5, 4, 3))
    other = Raw(store=q, channels=1, voxel_size=(1,) * 5, offset=(0,) * 5)
    return arr_of(store=p, stack=other, voxel_size=(1,) * 4, offset=(0,) * 4)


@probe("54_stack_shape_mismatch")
def _():
    p, _d = mkstore("s54a", (2, 5, 4, 3))
    q, _e = mkstore("s54b", (3, 6, 4, 3))
    other = Raw(store=q, voxel_size=(1,) * 4, offset=(0,) * 4)
    return arr_of(store=p, stack=other, voxel_size=(1,) * 4, offset=(0,) * 4)


# ------------------------------------------------------------------ modes / writability
@probe("60_writable_false_mode_w")
def _():
    p, _d = mkstore("s60", (5, 4, 3))
    return arr_of(store=p, writable=False, voxel_size=(1, 1, 1), offset=(0, 0, 0), _mode="w")


@probe("61_mode_a_plain")
def _():
    p, _d = mkstore("s61", (5, 4, 3))
    return arr_of(store=p, voxel_size=(1, 1, 1), offset=(0, 0, 0), _mode="a")


@probe("62_mode_a_with_channels")
def _():
    p, _d = mkstore("s62", (3, 5, 4, 3))
    return arr_of(store=p, channels=1, voxel_size=(1,) * 4, offset=(0,) * 4, _mode="a")


@probe("63_mode_w_with_channels")
def _():
    p, _d = mkstore("s63", (3, 5, 4, 3))
    return arr_of(store=p, channels=1, voxel_size=(1,) * 4, offset=(0,) * 4, _mode="w")


@probe("64_store_missing_mode_r")
def _():
    return arr_of(store=ROOT / "nope.zarr", voxel_size=(1, 1, 1), offset=(0, 0, 0))


@probe("65_store_missing_mode_a")
def _():
    return arr_of(store=ROOT / "nope2.zarr", voxel_size=(1, 1, 1), offset=(0, 0, 0), _mode="a")


@probe("66_store_missing_with_flip_mode_a")
def _():
    return arr_of(
        store=ROOT / "nope3.zarr", flip=[0], voxel_size=(1, 1, 1), offset=(0, 0, 0), _mode="a"
    )


# ------------------------------------------------------------------ other Dataset subclasses
@probe("70_labels_flip_mode_w")
def _():
    p, _d = mkstore("s70", (5, 4, 3))
    return describe(Labels(store=p, flip=[0], voxel_size=(1, 1, 1), offset=(0, 0, 0)).array("w"))


@probe("71_labels_channels")
def _():
    p, _d = mkstore("s71", (3, 5, 4, 3))
    return describe(Labels(store=p, channels=1, voxel_size=(1,) * 4, offset=(0,) * 4).array("r"))


@probe("72_affs_with_flip")
def _():
    p, _d = mkstore("s72", (6, 5, 4, 3))
    a = Affs(
        store=p,
        neighborhood=[[1, 0, 0], [0, 1, 0], [0, 0, 1], [3, 0, 0], [0, 3, 0], [0, 0, 3]],
        flip=[1],
        voxel_size=(1,) * 4,
        offset=(0,) * 4,
    )
    return describe(a.array("r"))


# ------------------------------------------------------------------ serialisation / spoof
@probe("80_model_dump_json_plain")
def _():
    p, _d = mkstore("s80", (5, 4, 3))
    return Raw(store=p, voxel_size=(1, 1, 1), offset=(0, 0, 0)).model_dump_json().replace(
        str(ROOT), "<TMP>"
    )


@probe("81_model_dump_python_keys")
def _():
    p, _d = mkstore("s81", (5, 4, 3))
    return sorted(Raw(store=p, flip=[0], voxel_size=(1, 1, 1), offset=(0, 0, 0)).model_dump())


@probe("82_roundtrip_json")
def _():
    p, _d = mkstore("s82", (3, 5, 4, 3))
    r = Raw(store=p, channels=1, flip=[0], voxel_size=(1,) * 4, offset=(0,) * 4)
    rebuilt = Raw.model_validate_json(r.model_dump_json())
    return describe(rebuilt.array("r"))


@probe("83_spoof")
def _():
    p, _d = mkstore("s83", (5, 4, 3))
    d = ROOT / "spoofdir"
    r = Raw(store=p, flip=[0], writable=False, voxel_size=(1, 1, 1), offset=(0, 0, 0))
    s = r.spoof(d)
    return {"cls": type(s).__name__, "json": s.model_dump_json().replace(str(ROOT), "<TMP>")}


@probe("84_json_schema_keys")
def _():
    return sorted(Raw.model_json_schema()["properties"])


# ------------------------------------------------------------------ subclass override (ADVISOR 7)
@probe("90_ADVISOR7_subclass_lazy_ops_calls_super")
def _():
    class ClipRaw(Raw):
        def lazy_ops(self, arr):
            super().lazy_ops(arr)
            arr.lazy_op(lambda d: np.clip(d, 0, 10))

    p, _d = mkstore("s90", (5, 4, 3))
    return describe(
        ClipRaw(store=p, scale_shift=(0.5, 0.0), voxel_size=(1, 1, 1), offset=(0, 0, 0)).array("r")
    )


@probe("91_ADVISOR7_subclass_lazy_ops_no_super")
def _():
    class ClipRaw2(Raw):
        def lazy_ops(self, arr):
            arr.lazy_op(lambda d: np.clip(d, 0, 10))

    p, _d = mkstore("s91", (5, 4, 3))
    return describe(
        ClipRaw2(store=p, scale_shift=(0.5, 0.0), voxel_size=(1, 1, 1), offset=(0, 0, 0)).array("r")
    )


@probe("92_ADVISOR7_subclass_super_only_ome_norm")
def _():
    class ClipRaw3(Raw):
        def lazy_ops(self, arr):
            super().lazy_ops(arr)
            arr.lazy_op(lambda d: np.clip(d, 0, 10))

    p, _d = mkstore("s92", (3, 5, 4, 3))
    m = mkome("s92", 3)
    return describe(
        ClipRaw3(store=p, ome_norm=m, voxel_size=(1,) * 4, offset=(0,) * 4).array("r")
    )


@probe("93_raw_has_lazy_ops_override")
def _():
    return {
        "Raw_defines_lazy_ops": "lazy_ops" in Raw.__dict__,
        "Dataset_defines_lazy_ops": "lazy_ops" in Raw.__mro__[1].__dict__,
    }


with open(sys.argv[1], "w") as f:
    json.dump({"tree": TREE, "results": RESULTS}, f, indent=1, sort_keys=True, default=str)
print("wrote", sys.argv[1], "tree", TREE)
