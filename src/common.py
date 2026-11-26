"""train.py / predict.py 공통: config 로드, run_id 생성."""
import copy
import re
from datetime import datetime

import yaml

DEFAULTS = {
    "split": {"mode": "auto", "ratio": 0.2, "min_items": 3, "seed": 42},
    "eval": {"k": 20, "test_users_only": True},
    "params": {},
}


def load_config(path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    out = copy.deepcopy(DEFAULTS)
    for key, val in cfg.items():
        if isinstance(val, dict) and key in out:
            out[key].update(val)
        else:
            out[key] = val
    if "model" not in out:
        raise KeyError(f"{path}에 model이 없음")
    return out


def make_run_id(model: str, name: str = None) -> str:
    rid = f"{datetime.now():%m%d-%H%M%S}_{model}"
    if name:
        rid += "_" + re.sub(r"[^\w\-]+", "-", name)
    return rid
