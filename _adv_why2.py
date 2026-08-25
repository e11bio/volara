import tempfile, pathlib, numpy as np, traceback
from funlib.persistence import prepare_ds, open_ds
tmp = pathlib.Path(tempfile.mkdtemp()); p = tmp / "r.zarr"
a = prepare_ds(p, shape=(4,), dtype=np.uint16, voxel_size=(1,), offset=(0,),
               axis_names=["x"], units=["nm"], mode="w")
a[:] = np.array([0,1,2,3], np.uint16)
arr = open_ds(p, mode="a"); arr.lazy_op((slice(None, None, -1),))
try:
    arr[np.s_[0:1]] = 99
except Exception:
    tb = traceback.format_exc().strip().splitlines()
    print("LAST 4 TB LINES:"); print("\n".join(tb[-4:]))
