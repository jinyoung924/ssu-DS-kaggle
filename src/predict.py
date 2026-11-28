"""train.csv 전체로 재학습 -> test user top-20 -> submission csv.

  python -m src.predict --config configs/ease.yaml
  python -m src.predict --from-run <run_id>   # 그 run의 config 사용, LightGCN은 best_epoch로 학습
"""
import argparse
import json

import numpy as np
import pandas as pd
import yaml

from . import data, experiments
from .common import load_config, make_run_id
from .models import build_model
from .validate_sub import check


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--config")
    g.add_argument("--from-run")
    ap.add_argument("--name", default=None)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    if args.from_run:
        run_dir = data.OUTPUTS / "runs" / args.from_run
        cfg = load_config(run_dir / "config.yaml")
        best_epoch = json.loads((run_dir / "metrics.json").read_text()).get("best_epoch")
        if best_epoch:
            cfg["params"]["epochs"] = best_epoch
            print(f"epochs={best_epoch} (from {args.from_run})")
        note = args.note or f"from {args.from_run}"
    else:
        cfg = load_config(args.config)
        note = args.note
    k = cfg["eval"]["k"]

    df = data.load_train()
    test_users = data.load_test_users()
    enc = data.Encoder(df, test_users)
    X = enc.matrix(df)

    model = build_model(cfg["model"], cfg["params"])
    model.fit(X, ctx={"df": df, "enc": enc, "stage": "full"})  # eval_fn 없음 -> early stopping 없이 epochs만큼 학습
    recs = model.recommend(enc.user_index(test_users), X, k)
    item_ids = enc.items.to_numpy()[recs]

    run_id = make_run_id(cfg["model"], args.name)
    out = data.OUTPUTS / "submissions" / f"{run_id}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "user_id": test_users,
        "item_id": [" ".join(map(str, row)) for row in item_ids],
    }).to_csv(out, index=False)

    run_dir = data.OUTPUTS / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    experiments.append(run_id, "submit", cfg["model"], cfg["params"], "full", None, note)

    errors = check(out, k=k)
    print(f"\nsubmission: {out}")
    print("format OK" if not errors else "format ERROR:\n  " + "\n  ".join(errors))
    print("제출 전: MMDD_NN_<strategy>.csv로 이름을 바꾸고 outputs/submissions/README.md 표에 한 줄 추가")
    print(f"제출 후: python -m src.experiments lb {run_id} <public score>")


if __name__ == "__main__":
    main()
