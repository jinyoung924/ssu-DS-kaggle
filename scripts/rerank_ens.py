"""재정렬 앙상블: 여러 scripts/rerank2.py 실행의 예측을 유저 안 순위 평균으로 합친다.

  # 검증 (각 run의 out-of-fold 예측 data/processed/rerank/pred_<run_id>_eval.npy)
  python -m scripts.rerank_ens --config configs/rerank_v2.yaml --runs <run_id> <run_id> ...
  # 제출 (각 run을 --submit으로 돌려 pred_<run_id>_test.npy가 있어야 함)
  python -m scripts.rerank_ens --config configs/rerank_v2.yaml --runs <run_id> <run_id> ... --submit

같은 config/feature 버전의 run끼리만 합칠 수 있다 (후보 목록이 같아야 함).
binary와 lambdarank처럼 점수 척도가 다른 모델도 합칠 수 있게 점수 대신 순위를 평균한다.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.rerank2 import CACHE, FEAT_V
from src import data, experiments
from src.common import make_run_id
from src.validate_sub import check


def rank_avg(preds, N, weights):
    R = 0.0
    for w, p in zip(weights, preds):
        P = p.reshape(-1, N)
        R = R + w * np.argsort(np.argsort(-P, axis=1, kind="stable"), axis=1)  # 0이 1위
    return -R


def recall(score, cand_y, gt_n, k=20):
    order = np.argsort(-score, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(cand_y, order, axis=1).sum() / np.minimum(gt_n, k).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--weights", type=float, nargs="*", default=None)
    ap.add_argument("--eval-seed", type=int, default=42)
    ap.add_argument("--cands", type=int, default=100)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    tag, N, k = Path(args.config).stem, args.cands, 20
    w = args.weights or [1.0] * len(args.runs)

    z = np.load(CACHE / f"{tag}_N{N}_f{FEAT_V}_s{args.eval_seed}.npz", allow_pickle=True)
    y, gt_n = z["y"], z["gt_n"]
    preds = [np.load(CACHE / f"pred_{r}_eval.npy") for r in args.runs]
    for r, p in zip(args.runs, preds):
        print(f"  {r}: {recall(p.reshape(-1, N), y, gt_n, k):.5f}")
    r_ens = recall(rank_avg(preds, N, w), y, gt_n, k)
    print(f"ensemble ({len(preds)} runs, weights {w}): {r_ens:.5f}  (blend {y[:, :k].sum() / np.minimum(gt_n, k).sum():.5f})")
    run_id = make_run_id("rerank_ens", f"{len(preds)}runs")
    experiments.append(run_id, "valid", "rerank_ens", {"config": args.config, "runs": args.runs, "weights": w},
                       "random", float(r_ens), args.note)
    if not args.submit:
        return

    zf = np.load(CACHE / f"{tag}_N{N}_f{FEAT_V}_full.npz", allow_pickle=True)
    cand = zf["cand"]
    preds = [np.load(CACHE / f"pred_{r}_test.npy").reshape(-1, N) for r in args.runs]
    score = rank_avg(preds, N, w)
    rec = np.take_along_axis(cand, np.argsort(-score, axis=1, kind="stable")[:, :k], axis=1)
    df = data.load_train()
    test_users = data.load_test_users()
    enc = data.Encoder(df, test_users)
    item_ids = enc.items.to_numpy()[rec]
    out = data.OUTPUTS / "submissions" / f"{run_id}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"user_id": test_users, "item_id": [" ".join(map(str, row)) for row in item_ids]}).to_csv(out, index=False)
    experiments.append(run_id, "submit", "rerank_ens", {"config": args.config, "runs": args.runs, "weights": w},
                       "full", None, args.note or "rerank ensemble")
    errors = check(out, k=k)
    print(f"\nsubmission: {out}\n" + ("format OK" if not errors else "format ERROR:\n  " + "\n  ".join(errors)))
    print(f"제출 후: python -m src.experiments lb {run_id} <public score>")


if __name__ == "__main__":
    main()
