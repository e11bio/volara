import sys, volara
from volara.datasets import Raw
T = "/home/jeff/.claude/jobs/b9887beb/tmp"
print("READER TREE:", volara.__file__)
for label, path in [("main-written ", f"{T}/_advC_main.json"),
                    ("BASE-written ", f"{T}/_advB_base.json"),
                    ("HEAD-written ", f"{T}/_advA_head.json")]:
    j = open(path).read()
    try:
        Raw.model_validate_json(j); print(f"  {label} -> OK")
    except Exception as e:
        ls = str(e).splitlines()
        print(f"  {label} -> ValidationError: {ls[1].strip()}  {ls[2].strip()}")
