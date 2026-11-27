"""Step 7 (H7): blend 후보 top-N을 LightGBM LambdaRank로 재정렬.

  python -m scripts.rerank --config configs/blend_best.yaml            # 검증: 유저 2-fold out-of-fold Recall@20
  python -m scripts.rerank --config configs/blend_best.yaml --submit   # + train 전체로 제출 파일 생성

feature: blend 점수/순위, component별 정규화 점수, 인기도, 유저가 산 같은 브랜드/카테고리 수,
         가격 차이, 유저-아이템 시기 차이, 유저 활동량. label: valid 정답 여부.
"""
import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src import data, experiments, features, metrics
from src.common import load_config, make_run_id
from src.features import DAY
from src.models import build_model
from src.models.base import topk
from src.models.blend import rownorm
from src.split import get_split
from src.validate_sub import check

PARAMS = dict(
    objective="lambdarank", metric="ndcg", ndcg_eval_at=[20], learning_rate=0.05, num_leaves=63,
    min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambdarank_truncation_level=30, verbose=-1,
)
YEAR = 365 * DAY


class FeatureBuilder:
    """blend를 학습하고, 유저 배치마다 후보 top-N과 feature 행렬을 만든다."""

    def __init__(self, cfg, X, df, enc, stage):
        self.X = X
        comps = [c for c in cfg["params"]["components"] if c.get("weight")]
        self.blend = build_model("blend", cfg["params"]).fit(X, ctx={"df": df, "enc": enc, "stage": stage})
        self.comp_names = [f"s_{c['model']}{i}" for i, c in enumerate(comps)]
        n_u, n_i = X.shape
        self.logpop = np.log1p(self.blend.popularity)
        self.user_n = np.asarray(X.sum(axis=1)).ravel()
        self.attr = {}
        for col in ["store", "leaf", "path"]:
            codes, uniq = pd.factorize(pd.Series(features.item_attr(enc, col)).replace("", np.nan))
            ok = codes >= 0
            L = sp.csr_matrix((np.ones(ok.sum(), np.float32), (np.flatnonzero(ok), codes[ok])), shape=(n_i, len(uniq)))
            self.attr[col] = (codes, L)
        self.logprice = np.log1p(features.item_attr(enc, "price").astype(np.float64))
        u, i = enc.user_index(df["user_id"]), enc.item_index(df["item_id"])
        t = df["timestamp"].to_numpy() / YEAR
        self.item_tmed = pd.Series(t).groupby(i).median().reindex(range(n_i)).to_numpy()
        self.item_tfirst = pd.Series(t).groupby(i).min().reindex(range(n_i)).to_numpy()
        self.user_tmed = pd.Series(t).groupby(u).median().reindex(range(n_u)).to_numpy()
        self.user_tlast = pd.Series(t).groupby(u).max().reindex(range(n_u)).to_numpy()
        self.user_price = pd.Series(self.logprice[i]).groupby(u).median().reindex(range(n_u)).to_numpy()

    def batch(self, users, n_cand):
        X = self.X
        C = [rownorm(np.asarray(m.score(users, X), dtype=np.float32)) for _, m in self.blend.models]
        S = self.blend.tie + sum(w * c for (w, _), c in zip(self.blend.models, C))
        r, c = X[users].nonzero()
        S[r, c] = -np.inf
        cand = topk(S, n_cand)
        rows = np.arange(len(users))[:, None]
        f = {"blend": S[rows, cand], "blend_rank": np.broadcast_to(np.arange(n_cand), cand.shape)}
        for name, Ck in zip(self.comp_names, C):
            f[name] = Ck[rows, cand]
        f["logpop"] = self.logpop[cand]
        Xu = X[users]
        for col, (codes, L) in self.attr.items():
            P = (Xu @ L).toarray()  # 유저가 산 아이템 중 속성값별 개수
            cc = codes[cand]
            f[f"n_same_{col}"] = np.where(cc >= 0, P[rows, np.maximum(cc, 0)], np.nan)
        f["price"] = self.logprice[cand]
        f["price_diff"] = np.abs(self.logprice[cand] - self.user_price[users][:, None])
        f["dt_median"] = self.user_tmed[users][:, None] - self.item_tmed[cand]
        f["item_first_minus_user_last"] = self.item_tfirst[cand] - self.user_tlast[users][:, None]
        f["user_n"] = np.broadcast_to(self.user_n[users][:, None], cand.shape)
        F = np.stack([np.asarray(v, dtype=np.float32).ravel() for v in f.values()], axis=1)
        return cand, F, list(f)


def build(fb, users, n_cand, batch=1024):
    cands, Fs = [], []
    for s in range(0, len(users), batch):
        c, F, names = fb.batch(users[s:s + batch], n_cand)
        cands.append(c)
        Fs.append(F)
    return np.vstack(cands), np.vstack(Fs), names


def rows_of(user_pos, n):
    return (user_pos[:, None] * n + np.arange(n)).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cands", type=int, default=100)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    N, k = args.cands, cfg["eval"]["k"]
    t0 = time.time()

    df = data.load_train()
    test_users = data.load_test_users()
    enc = data.Encoder(df, test_users)
    tr, va, split_used = get_split(df, **cfg["split"])
    X_tr = enc.matrix(tr)
    gt = pd.DataFrame({"u": enc.user_index(va["user_id"]), "i": enc.item_index(va["item_id"])})
    gt = gt[gt["u"].isin(enc.user_index(test_users))].groupby("u")["i"].apply(set)
    users, gts = gt.index.to_numpy(), gt.tolist()

    fb = FeatureBuilder(cfg, X_tr, tr, enc, "valid")
    cand, F, names = build(fb, users, N)
    y = np.array([[i in g for i in row] for row, g in zip(cand, gts)], dtype=np.float32).ravel()
    print(f"features {F.shape}, positives {int(y.sum())}, candidate Recall@{N} "
          f"{metrics.recall_at_k(cand, gts, N):.5f}  ({time.time() - t0:.0f}s)", flush=True)

    rng = np.random.default_rng(0)
    fold = rng.integers(0, 2, len(users))
    oof = np.zeros(len(y), np.float32)
    best_iters, imp = [], np.zeros(len(names))
    for f in (0, 1):
        trp, tep = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        es = rng.random(len(trp)) < 0.1  # early stopping용 (평가 fold와 분리)
        dtr = lgb.Dataset(F[rows_of(trp[~es], N)], y[rows_of(trp[~es], N)], group=[N] * int((~es).sum()), feature_name=names)
        dva = lgb.Dataset(F[rows_of(trp[es], N)], y[rows_of(trp[es], N)], group=[N] * int(es.sum()), feature_name=names)
        model = lgb.train(PARAMS, dtr, num_boost_round=1000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[rows_of(tep, N)] = model.predict(F[rows_of(tep, N)], num_iteration=model.best_iteration)
        best_iters.append(model.best_iteration)
        imp += model.feature_importance("gain")
        print(f"  fold {f}: best_iter {model.best_iteration}  ({time.time() - t0:.0f}s)", flush=True)

    order = np.argsort(-oof.reshape(-1, N), axis=1, kind="stable")[:, :k]
    rec = np.take_along_axis(cand, order, axis=1)
    r_rerank, r_blend = metrics.recall_at_k(rec, gts, k), metrics.recall_at_k(cand[:, :k], gts, k)
    print(f"\nvalid Recall@{k}: blend {r_blend:.5f} -> rerank {r_rerank:.5f} (out-of-fold, 2-fold by user)")
    top = np.argsort(-imp)[:10]
    print("top features (gain): " + ", ".join(f"{names[j]}" for j in top))
    run_id = make_run_id("rerank", f"N{N}")
    experiments.append(run_id, "valid", "rerank", {"config": args.config, "cands": N, "blend_valid": round(r_blend, 5)},
                       split_used, r_rerank, args.note)

    if not args.submit:
        return
    n_iter = int(np.mean(best_iters))
    model = lgb.train(PARAMS, lgb.Dataset(F, y, group=[N] * len(users), feature_name=names), num_boost_round=n_iter)
    del F, oof
    X = enc.matrix(df)
    fb_full = FeatureBuilder(cfg, X, df, enc, "full")
    tu = enc.user_index(test_users)
    cand, F, _ = build(fb_full, tu, N)
    pred = model.predict(F).reshape(-1, N)
    rec = np.take_along_axis(cand, np.argsort(-pred, axis=1, kind="stable")[:, :k], axis=1)
    item_ids = enc.items.to_numpy()[rec]
    out = data.OUTPUTS / "submissions" / f"{run_id}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"user_id": test_users, "item_id": [" ".join(map(str, row)) for row in item_ids]}).to_csv(out, index=False)
    experiments.append(run_id, "submit", "rerank", {"config": args.config, "cands": N, "n_iter": n_iter}, "full", None,
                       args.note or f"rerank of {args.config}")
    errors = check(out, k=k)
    print(f"\nsubmission: {out}\n" + ("format OK" if not errors else "format ERROR:\n  " + "\n  ".join(errors)))
    print(f"제출 후: python -m src.experiments lb {run_id} <public score>  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
