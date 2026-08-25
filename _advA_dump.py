import json, sys
import volara
from volara.datasets import Raw
print("VOLARA_FROM:", volara.__file__)
r = Raw(store="/tmp/x.zarr", voxel_size=(1,1,1), offset=(0,0,0))
j = r.model_dump_json()
print("JSON:", j)
print("KEYS:", sorted(r.model_dump().keys()))
print("HAS_ops_key:", "ops" in r.model_dump())
print("SCHEMA_PROPS:", sorted(Raw.model_json_schema()["properties"].keys()))
open("/home/jeff/.claude/jobs/b9887beb/tmp/_advA_head.json","w").write(j)
