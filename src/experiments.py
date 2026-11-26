"""실험 기록 (experiments.csv).

  python -m src.experiments show              # 기록 보기
  python -m src.experiments lb <run_id> 0.0812  # 제출한 run의 public LB 점수 기록
"""
import argparse
import json
from datetime import datetime

import pandas as pd

from . import data

PATH = data.ROOT / "experiments.csv"
COLUMNS = ["time", "run_id", "stage", "model", "params", "split", "recall@20", "public_lb", "note"]


def append(run_id, stage, model, params, split="", recall=None, note=""):
    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "run_id": run_id,
        "stage": stage,
        "model": model,
        "params": json.dumps(params, ensure_ascii=False),
        "split": split,
        "recall@20": None if recall is None else round(recall, 5),
        "public_lb": None,
        "note": note,
    }
    df = pd.DataFrame([row], columns=COLUMNS)
    df.to_csv(PATH, mode="a", header=not PATH.exists(), index=False)


def set_lb(run_id, score):
    df = pd.read_csv(PATH)
    mask = (df["run_id"] == run_id) & (df["stage"] == "submit")
    if not mask.any():
        raise KeyError(f"stage=submit인 {run_id} 기록이 없음")
    df.loc[mask, "public_lb"] = score
    df.to_csv(PATH, index=False)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show")
    p = sub.add_parser("lb")
    p.add_argument("run_id")
    p.add_argument("score", type=float)
    args = ap.parse_args()
    if args.cmd == "show":
        with pd.option_context("display.max_colwidth", 60, "display.width", 200):
            print(pd.read_csv(PATH).to_string(index=False))
    else:
        set_lb(args.run_id, args.score)
        print(f"{args.run_id}: public_lb={args.score}")


if __name__ == "__main__":
    main()
