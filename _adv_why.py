"""Why ReverseAxes needs a volara-level guard but a callable does not: funlib guards callables
(is_writeable False -> explicit RuntimeError) but NOT reversed slices (is_writeable True -> the
write silently lands in the wrong place)."""
import tempfile, pathlib, numpy as np
from funlib.persistence import prepare_ds, open_ds
tmp = pathlib.Path(tempfile.mkdtemp())
p = tmp / "r.zarr"
a = prepare_ds(p, shape=(4,), dtype=np.uint16, voxel_size=(1,), offset=(0,),
               axis_names=["x"], units=["nm"], mode="w")
a[:] = np.array([0,1,2,3], np.uint16)

arr = open_ds(p, mode="a")
arr.lazy_op((slice(None, None, -1),))          # exactly what ReverseAxes emits
print("reversed slice: is_writeable =", arr.is_writeable)
print("view before write:", np.asarray(arr[:]))
try:
    arr[np.s_[0:1]] = 99                        # write index 0 of the REVERSED view
    print("write: NO ERROR (funlib does not guard this)")
except Exception as e:
    print("write:", type(e).__name__, " ".join(str(e).split())[:90])
print("on-disk after write:", np.asarray(open_ds(p, mode="r")[:]))
print("-> intended reversed[0] == source[3]; observed source =", np.asarray(open_ds(p, mode="r")[:]))
