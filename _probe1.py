"""Serialization / worker round-trip probes for volara Dataset ops."""
import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import volara
from funlib.persistence import prepare_ds
from pydantic import TypeAdapter

print("VOLARA:", volara.__file__)
assert Path(volara.__file__).is_relative_to(Path.cwd()), "WRONG TREE"

from volara.datasets import Raw, Dataset, PydanticDataset  # noqa: E402
from volara.utils import StrictBaseModel  # noqa: E402

HAS_OPS = "ops" in Raw.model_fields
print("HAS_OPS field:", HAS_OPS)

tmp = Path(tempfile.mkdtemp())


def store(shape, name="t.zarr"):
    p = tmp / name
    arr = prepare_ds(
        p, shape=shape, dtype=np.uint16, voxel_size=(1,) * len(shape),
        offset=(0,) * len(shape), axis_names=list("tczyx"[-len(shape):]),
        units=["nm"] * len(shape), mode="w",
    )
    data = np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)
    arr[:] = data
    return p, data


def section(name):
    print("\n===== " + name + " =====")


p3, d3 = store((5, 4, 3))


def mk(**kw):
    return Raw(store=p3, voxel_size=(1, 1, 1), offset=(0, 0, 0), **kw)


# ---------------------------------------------------------------- 1. python-mode dump
section("1. model_dump() python-mode with ops")
if HAS_OPS:
    r = mk(ops=[lambda d: d * 2])
    dumped = r.model_dump()
    print("type of dumped['ops'][0]:", type(dumped["ops"][0]).__name__)
    print("keys:", sorted(dumped))
else:
    print("skip (no ops field)")

# ---------------------------------------------------------------- 2. spoof
section("2. spoof() with ops")
try:
    r = mk(ops=[lambda d: d * 2]) if HAS_OPS else mk(flip=[0])
    sp = r.spoof(tmp / "spoofdir")
    print("spoof OK ->", type(sp).__name__, "ops:", getattr(sp, "ops", "n/a"))
except Exception as e:
    print("spoof RAISED:", type(e).__name__, e)

# ---------------------------------------------------------------- 3. worker hop exactly as cli.py
section("3. worker hop: model_dump_json -> json.loads -> validate_python")
if HAS_OPS:
    r = mk(ops=[lambda d: d * 2])
    js = r.model_dump_json()
    cfg = json.loads(js)
    print("json ops entry type:", type(cfg["ops"][0]).__name__, repr(cfg["ops"][0])[:40])
    rebuilt = TypeAdapter(Raw).validate_python(cfg)
    got = np.asarray(rebuilt.array("r")[:])
    print("array equal:", np.array_equal(got, d3 * 2))
    # re-dump
    js2 = rebuilt.model_dump_json()
    print("re-dump ok, identical bytes:", js == js2)
    rebuilt2 = Raw.model_validate_json(js2)
    print("2nd rebuild array equal:", np.array_equal(np.asarray(rebuilt2.array("r")[:]), d3 * 2))
else:
    print("skip")

# ---------------------------------------------------------------- 4. equality
section("4. equality / hashing after round trip")
r = mk(flip=[0])
r2 = Raw.model_validate_json(r.model_dump_json())
print("keyword-only Raw: rebuilt == original ->", r2 == r)
if HAS_OPS:
    ro = mk(ops=[lambda d: d * 2])
    ro2 = Raw.model_validate_json(ro.model_dump_json())
    print("ops Raw: rebuilt == original ->", ro2 == ro)
    ro3 = Raw.model_validate_json(ro.model_dump_json())
    print("two rebuilds of same json equal ->", ro2 == ro3)
try:
    print("hash(Raw):", hash(r))
except Exception as e:
    print("hash RAISED:", type(e).__name__, e)

# ---------------------------------------------------------------- 5. nested in a task-like model
section("5. Raw nested as a typed field in another StrictBaseModel")


class _Task(StrictBaseModel):
    in_data: Raw
    other: PydanticDataset | None = None


if HAS_OPS:
    t = _Task(in_data=mk(ops=[lambda d: d + 1]), other=mk(ops=[lambda d: d + 3]))
else:
    t = _Task(in_data=mk(flip=[0]), other=mk(flip=[0]))
tj = t.model_dump_json()
t2 = TypeAdapter(_Task).validate_python(json.loads(tj))
print("nested rebuilt ok:", type(t2.in_data).__name__, type(t2.other).__name__)
if HAS_OPS:
    print("nested ops array:", np.array_equal(np.asarray(t2.in_data.array("r")[:]), d3 + 1))
    print("nested union ops array:", np.array_equal(np.asarray(t2.other.array("r")[:]), d3 + 3))

# ---------------------------------------------------------------- 6. stack nesting
section("6. Raw with stack=<Raw>")
p4, d4 = store((1, 5, 4, 3), name="s.zarr")
p4b, d4b = store((1, 5, 4, 3), name="s2.zarr")
inner_kw = dict(ops=[lambda d: d + 7]) if HAS_OPS else {}
inner = Raw(store=p4b, voxel_size=(1,) * 4, offset=(0,) * 4, **inner_kw)
outer = Raw(store=p4, voxel_size=(1,) * 4, offset=(0,) * 4, stack=inner)
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    sj = outer.model_dump_json()
    for wi in w:
        print("WARN:", str(wi.message)[:300])
print("stack json:", json.dumps(json.loads(sj)["stack"])[:300])
try:
    back = Raw.model_validate_json(sj)
    print("stack rebuild OK, stack type:", type(back.stack).__name__)
    print("stack array works:", np.asarray(back.array("r")[:]).shape)
except Exception as e:
    print("stack rebuild RAISED:", type(e).__name__, str(e)[:400])

# ---------------------------------------------------------------- 7. json schema
section("7. model_json_schema()")
for cls in (Raw,):
    try:
        s = cls.model_json_schema()
        print(cls.__name__, "schema OK; ops prop:", json.dumps(s["properties"].get("ops"))[:200])
    except Exception as e:
        print(cls.__name__, "schema RAISED:", type(e).__name__, str(e)[:300])
try:
    from volara.blockwise import get_blockwise_tasks_type
    ta = get_blockwise_tasks_type()
    ta.json_schema()
    print("blockwise TypeAdapter json_schema OK")
except Exception as e:
    print("blockwise TypeAdapter json_schema RAISED:", type(e).__name__, str(e)[:300])

# ---------------------------------------------------------------- 8. deepcopy / model_copy
section("8. deepcopy / model_copy(deep=True)")
try:
    rr = mk(ops=[lambda d: d * 2]) if HAS_OPS else mk(flip=[0])
    c1 = copy.deepcopy(rr)
    c2 = rr.model_copy(deep=True)
    print("deepcopy ok:", np.array_equal(np.asarray(c1.array("r")[:]), np.asarray(rr.array("r")[:])))
    print("model_copy ok:", np.array_equal(np.asarray(c2.array("r")[:]), np.asarray(rr.array("r")[:])))
except Exception as e:
    print("copy RAISED:", type(e).__name__, str(e)[:300])

print("\nDONE")
