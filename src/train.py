"""로컬 validation 실험.

  python -m src.train --config configs/ease.yaml [--name reg1000] [--note "메모"]

train.csv를 split -> split train으로 학습 -> valid Recall@20 -> outputs/runs/<run_id>/, experiments.csv 기록.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import yaml

from . import data, experiments, metrics
from .common import load_config, make_run_id
from .models import build_model
from .split import get_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", default=None, help="run_id 뒤에 붙일 이름")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    k = cfg["eval"]["k"]
    t0 = time.time()

    df = data.load_train()
    test_users = data.load_test_users()
    enc = data.Encoder(df, test_users)
    tr, va, split_used = get_split(df, **cfg["split"])
    X_tr = enc.matrix(tr)

    gt = pd.DataFrame({"u": enc.user_index(va["user_id"]), "i": enc.item_index(va["item_id"])})
    if cfg["eval"]["test_users_only"]:
        gt = gt[gt["u"].isin(enc.user_index(test_users))]
    gt = gt.groupby("u")["i"].apply(set)
    eval_users, gt_sets = gt.index.to_numpy(), gt.tolist()
    print(f"users={enc.n_users} items={enc.n_items} train_nnz={X_tr.nnz} "
          f"valid_users={len(eval_users)} split={split_used}")

    def eval_fn(model):
        return metrics.recall_at_k(model.recommend(eval_users, X_tr, k), gt_sets, k)

    model = build_model(cfg["model"], cfg["params"])
    model.fit(X_tr, eval_fn=eval_fn, ctx={"df": tr, "enc": enc, "stage": "valid"})
    score = eval_fn(model)
    sec = time.time() - t0

    run_id = make_run_id(cfg["model"], args.name)
    run_dir = data.OUTPUTS / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    result = {"recall@20": score, "n_eval_users": int(len(eval_users)), "split": split_used,
              "best_epoch": model.best_epoch, "seconds": round(sec, 1)}
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    experiments.append(run_id, "valid", cfg["model"], cfg["params"], split_used, score, args.note)

    print(f"\n[{run_id}] valid Recall@{k} = {score:.5f}  ({sec:.1f}s)")


if __name__ == "__main__":
    main()
