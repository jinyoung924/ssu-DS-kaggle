"""blend 가중치 탐색.

component를 한 번씩만 학습한 뒤, valid 유저 배치마다 점수를 계산해 여러 가중치 조합을 한 번에 평가한다.
(src.train으로 조합마다 돌리면 EASE를 매번 다시 학습해서 느림)

  python -m scripts.tune_blend configs/tune/step2_artifact.yaml

spec (yaml):
  split: {seed: 42}                  # 선택. 기본값은 src/common.py DEFAULTS
  components: {name: {model, params}}
  evaluate: {label: "ease=1,fnb20=0.3"}   # 고정 조합 (ablation 등)
  init: "ease=1,fnb20=0.3"           # coordinate descent 시작점
  search: {fnb20: [0.2, 0.4]}        # component별 후보 weight
  rounds: 2
  save_best: configs/blend_xxx.yaml  # 최종 가중치를 blend config로 저장
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import data, metrics
from src.common import DEFAULTS
from src.models import build_model
from src.models.base import topk
from src.models.blend import rownorm

K = 20


def parse_combo(c) -> dict:
    if isinstance(c, str):
        return {k.strip(): float(v) for k, v in (p.split("=") for p in c.split(","))}
    return {k: float(v) for k, v in c.items()}


class Evaluator:
    def __init__(self, spec):
        from src.split import get_split

        split = {**DEFAULTS["split"], **(spec.get("split") or {})}
        df = data.load_train()
        test_users = data.load_test_users()
        self.enc = enc = data.Encoder(df, test_users)
        tr, va, _ = get_split(df, **split)
        self.X = enc.matrix(tr)
        gt = pd.DataFrame({"u": enc.user_index(va["user_id"]), "i": enc.item_index(va["item_id"])})
        gt = gt[gt["u"].isin(enc.user_index(test_users))].groupby("u")["i"].apply(set)
        self.users, self.gts = gt.index.to_numpy(), gt.tolist()
        pop = np.asarray(self.X.sum(axis=0)).ravel().astype(np.float32)
        self.tie = 1e-6 * pop / pop.max()
        ctx = {"df": tr, "enc": enc, "stage": "valid"}
        print(f"split={split} valid_users={len(self.users)}", flush=True)
        self.models = {}
        for name, c in spec["components"].items():
            t = time.time()
            self.models[name] = build_model(c["model"], c.get("params") or {}).fit(self.X, ctx=ctx)
            print(f"  fit {name:14s} {time.time() - t:6.1f}s", flush=True)

    def run(self, combos: dict, batch=1024) -> dict:
        need = sorted({k for w in combos.values() for k, v in w.items() if v})
        recs = {n: np.empty((len(self.users), K), np.int64) for n in combos}
        for s in range(0, len(self.users), batch):
            u = self.users[s:s + batch]
            C = {k: rownorm(np.asarray(self.models[k].score(u, self.X), dtype=np.float32)) for k in need}
            r, c = self.X[u].nonzero()
            for n, w in combos.items():
                S = self.tie + sum(v * C[k] for k, v in w.items() if v)
                S[r, c] = -np.inf
                recs[n][s:s + len(u)] = topk(S, K)
        return {n: metrics.recall_at_k(recs[n], self.gts, K) for n in combos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    args = ap.parse_args()
    with open(args.spec, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    t0 = time.time()
    ev = Evaluator(spec)
    log = []

    def record(stage, combos, res):
        for n in combos:
            log.append({"stage": stage, "combo": n, "weights": json.dumps(combos[n]), "recall@20": round(res[n], 5)})
            print(f"  [{stage}] {n:34s} {res[n]:.5f}", flush=True)

    combos = {n: parse_combo(c) for n, c in (spec.get("evaluate") or {}).items()}
    cur = parse_combo(spec["init"]) if spec.get("init") else None
    if cur:
        combos["init"] = cur
    best = None
    if combos:
        res = ev.run(combos)
        record("evaluate", combos, res)
        best = res.get("init")
    for rnd in range(spec.get("rounds", 1) if spec.get("search") and cur else 0):
        for comp, cands in spec["search"].items():
            cands = [float(v) for v in cands if float(v) != cur.get(comp, 0.0)]
            if not cands:
                continue
            combos = {f"{comp}={v:g}": {**cur, comp: v} for v in cands}
            res = ev.run(combos)
            record(f"r{rnd + 1}", combos, res)
            n = max(res, key=res.get)
            if res[n] > best:
                best, cur = res[n], combos[n]
            print(f"  -> r{rnd + 1} {comp}={cur.get(comp, 0.0):g}  best {best:.5f}  ({time.time() - t0:.0f}s)", flush=True)

    out = data.OUTPUTS / "tune"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{datetime.now():%m%d-%H%M%S}_{Path(args.spec).stem}.csv"
    pd.DataFrame(log).to_csv(path, index=False)
    print(f"\nlog: {path}")
    if cur:
        print(f"best {best:.5f}: " + ", ".join(f"{k}={v:g}" for k, v in cur.items() if v))
        if spec.get("save_best"):
            cfg = {"model": "blend", "params": {"components": [
                {"model": spec["components"][k]["model"], "weight": v,
                 "params": spec["components"][k].get("params") or {}}
                for k, v in cur.items() if v
            ]}}
            if spec.get("split"):
                cfg["split"] = spec["split"]
            with open(spec["save_best"], "w", encoding="utf-8") as f:
                f.write(f"# scripts/tune_blend.py {Path(args.spec).name} 결과: valid Recall@20 {best:.5f}\n")
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            print(f"saved: {spec['save_best']}")


if __name__ == "__main__":
    main()
