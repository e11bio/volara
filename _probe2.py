"""Callable-kind round trips: what actually survives the worker hop."""
import functools
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

from volara.datasets import Raw  # noqa: E402

tmp = Path(tempfile.mkdtemp())
p = tmp / "t.zarr"
arr = prepare_ds(p, shape=(5, 4, 3), dtype=np.uint16, voxel_size=(1, 1, 1),
                 offset=(0, 0, 0), axis_names=["z", "y", "x"], units=["nm"] * 3, mode="w")
d3 = np.arange(60, dtype=np.uint16).reshape(5, 4, 3)
arr[:] = d3


def mk(**kw):
    return Raw(store=p, voxel_size=(1, 1, 1), offset=(0, 0, 0), **kw)


def worker_hop(r):
    """exactly what volara/cli.py blockwise_worker does"""
    return TypeAdapter(Raw).validate_python(json.loads(r.model_dump_json()))


def double(d):
    return d * 2


def scale(d, k):
    return d * k


class Holder:
    def __init__(self, k):
        self.k = k

    def __call__(self, d):
        return d * self.k

    def method(self, d):
        return d * self.k


print("\n===== A. lambda defined in this script (__main__) =====")
try:
    r = worker_hop(mk(ops=[lambda d: d * 2]))
    print("OK", np.array_equal(np.asarray(r.array("r")[:]), d3 * 2))
except Exception as e:
    print("RAISED:", type(e).__name__, str(e)[:200])

print("\n===== B. module-level function in THIS module (__main__) =====")
try:
    r = worker_hop(mk(ops=[double]))
    print("OK", np.array_equal(np.asarray(r.array("r")[:]), d3 * 2))
except Exception as e:
    print("RAISED:", type(e).__name__, str(e)[:200])

print("\n===== C. functools.partial =====")
try:
    r = worker_hop(mk(ops=[functools.partial(scale, k=3)]))
    print("OK", np.array_equal(np.asarray(r.array("r")[:]), d3 * 3))
except Exception as e:
    print("RAISED:", type(e).__name__, str(e)[:200])

print("\n===== D. callable object / bound method =====")
for label, fn in (("__call__ obj", Holder(4)), ("bound method", Holder(5).method)):
    try:
        r = worker_hop(mk(ops=[fn]))
        print(label, "OK", np.asarray(r.array("r")[:])[0, 0, 1])
    except Exception as e:
        print(label, "RAISED:", type(e).__name__, str(e)[:200])

print("\n===== E. function from a SEPARATE importable module (by-reference pickle) =====")
modsrc = "def triple(d):\n    return d * 3\n"
moddir = tmp / "extmod"
moddir.mkdir()
(moddir / "myops.py").write_text(modsrc)
sys.path.insert(0, str(moddir))
import myops  # noqa: E402

r = mk(ops=[myops.triple])
js = r.model_dump_json()
b64 = json.loads(js)["ops"][0]
import base64
blob = base64.b64decode(b64)
print("pickled BY REFERENCE (module name appears in blob):", b"myops" in blob, "len", len(blob))
# simulate the worker: same code, module no longer importable
del sys.modules["myops"]
sys.path.remove(str(moddir))
try:
    r2 = TypeAdapter(Raw).validate_python(json.loads(js))
    print("worker rebuild OK ->", np.array_equal(np.asarray(r2.array("r")[:]), d3 * 3))
except Exception as e:
    print("worker rebuild RAISED:", type(e).__name__, str(e)[:200])

print("\n===== F. unpicklable callable (closes over a file handle) =====")
fh = open(tmp / "x.txt", "w")
try:
    r = mk(ops=[lambda d, _f=fh: d])
    js = r.model_dump_json()
    print("dump OK (unexpected)", len(js))
except Exception as e:
    print("dump RAISED:", type(e).__name__, str(e)[:200])

print("\n===== G. ops on a write-mode open =====")
outp = tmp / "out.zarr"
prepare_ds(outp, shape=(5, 4, 3), dtype=np.uint16, voxel_size=(1, 1, 1), offset=(0, 0, 0),
           axis_names=["z", "y", "x"], units=["nm"] * 3, mode="w")
from volara.datasets import Labels  # noqa: E402
lab = Labels(store=outp, voxel_size=(1, 1, 1), offset=(0, 0, 0), ops=[lambda d: d + 1])
try:
    a = lab.array("r+")
    print("write-mode array() returned; is_writeable:", a.is_writeable)
    try:
        a[a.roi] = np.ones((5, 4, 3), dtype=np.uint16)
        print("write went through; on-disk value now:",
              np.asarray(Labels(store=outp, voxel_size=(1, 1, 1), offset=(0, 0, 0)).array("r")[:])[0, 0, 0])
    except Exception as e:
        print("__setitem__ RAISED:", type(e).__name__, str(e)[:200])
except Exception as e:
    print("array('r+') RAISED:", type(e).__name__, str(e)[:200])

print("\n===== H. channels in write mode (base behaviour, for comparison) =====")
lab2 = Labels(store=outp, voxel_size=(1, 1, 1), offset=(0, 0, 0), channels=0)
try:
    a2 = lab2.array("r+")
    print("channels write-mode array ok; is_writeable:", a2.is_writeable)
except Exception as e:
    print("RAISED:", type(e).__name__, str(e)[:200])

print("\nDONE")
