"""2단계 추천 v2: blend 후보 top-N을 LightGBM으로 재정렬. 여러 split으로 학습 데이터를 늘린다 (개선안_01.md 축 B).

  # 검증: seed 1~4 split으로 학습, seed 42 split(기존 로컬 검증)으로 평가
  python -m scripts.rerank2 --config configs/blend_best.yaml --train-seeds 1 2 3 4 --eval-seed 42
  # 제출: seed 42까지 포함해 학습하고 train 전체로 만든 후보를 재정렬
  python -m scripts.rerank2 --config configs/blend_best.yaml --train-seeds 1 2 3 4 --eval-seed 42 --submit

v1(scripts/rerank.py)이 실패한 이유: 정답이 split 하나의 6,781개뿐이라 GBDT가 blend보다 못 배웠다.
v2: split마다 component를 다시 학습해 후보와 feature를 만들고(data/processed/rerank/에 캐시), 정답을 split 수만큼 늘린다.
각 split의 valid는 그 split의 train에 없는 아이템이므로, test 상황(정답은 train에 없음)과 같다.
"""
import argparse
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src import data, experiments, features, metrics
from src.common import DEFAULTS, load_config, make_run_id
from src.features import DAY
from src.models import build_model
from src.models.base import topk
from src.models.blend import rownorm
from src.split import get_split
from src.validate_sub import check

YEAR = 365 * DAY
CACHE = data.PROCESSED / "rerank"
FEAT_V = int(os.environ.get("RERANK_FEAT_V", 2))  # feature를 바꾸면 올린다 (캐시 파일 이름에 들어감)
PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=300, feature_fraction=0.7,
    bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, metric="auc", verbose=-1, num_threads=16, seed=0,
)


def _agg(values, groups, how, n):
    return pd.Series(values).groupby(groups).agg(how).reindex(range(n)).to_numpy()


class FeatureBuilder:
    """component를 학습하고, 유저 배치마다 blend 후보 top-N과 feature 행렬을 만든다."""

    def __init__(self, cfg, X, df, enc, stage):
        self.X = X = X.tocsr()
        ctx = {"df": df, "enc": enc, "stage": stage}
        self.comps = []  # (이름, blend 가중치, 모델). feature: true인 component는 가중치 0으로 feature만 만든다
        for j, c in enumerate(cfg["params"]["components"]):
            if c.get("weight") or c.get("feature"):
                t = time.time()
                m = build_model(c["model"], c.get("params") or {}).fit(X, ctx=ctx)
                self.comps.append((f"{c['model']}{j}", float(c.get("weight") or 0.0), m))
                print(f"    fit {c['model']}{j} {time.time() - t:.0f}s", flush=True)
        n_u, n_i = X.shape
        self.n_u = n_u
        self.pop = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)
        self.tie = 1e-6 * self.pop / self.pop.max()
        self.user_n = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
        # 유저가 평소 사는 아이템의 인기도 (log) 평균
        self.user_logpop = np.asarray(X @ np.log1p(self.pop)).ravel() / np.maximum(self.user_n, 1)
        # EASE 계열 component의 B: 후보와 가장 강하게 연결된 '하나의' 구매 아이템 (합이 아니라 최대)
        self.ease_B = next((m.B for _, _, m in self.comps
                            if getattr(m, "B", None) is not None and m.B.shape == (n_i, n_i)), None)
        self.attr = {}
        for col in ["store", "leaf", "path", "dept"]:
            codes, uniq = pd.factorize(pd.Series(features.item_attr(enc, col)).replace("", np.nan))
            ok = codes >= 0
            L = sp.csr_matrix((np.ones(ok.sum(), np.float32), (np.flatnonzero(ok), codes[ok])), shape=(n_i, len(uniq)))
            self.attr[col] = (codes, L)
        self.logprice = np.log1p(features.item_attr(enc, "price").astype(np.float64))
        u, i = enc.user_index(df["user_id"]), enc.item_index(df["item_id"])
        t = df["timestamp"].to_numpy() / YEAR
        self.item_t = {h: _agg(t, i, h, n_i) for h in ("min", "median", "max")}
        self.user_t = {h: _agg(t, u, h, n_u) for h in ("min", "median", "max")}
        self.user_price = _agg(self.logprice[i], u, "median", n_u)
        r = df["rating"].to_numpy().astype(np.float32)
        self.item_rating, self.user_rating = _agg(r, i, "mean", n_i), _agg(r, u, "mean", n_u)
        # 아이템 월별 누적 판매 수 -> 유저 활동 기간(첫~마지막 구매 월) 안의 판매 수
        m = (df["timestamp"].to_numpy() // (30 * DAY)).astype(np.int64)
        m -= m.min()
        cnt = np.zeros((n_i, int(m.max()) + 2), np.float32)
        np.add.at(cnt, (i, m + 1), 1)
        self.cum = cnt.cumsum(axis=1)
        self.user_m = (_agg(m, u, "min", n_u).astype(np.int64), _agg(m, u, "max", n_u).astype(np.int64))
        # Ordering Artifact: 파일 위치, (유저, 아이템) 구매 여부 조회용 정렬 key
        self.order = features.user_file_order(enc)
        self.pos = np.empty(n_u, np.int64)
        self.pos[self.order] = np.arange(n_u)
        coo = X.tocoo()
        self.n_i = n_i
        self.keys = np.sort(coo.row.astype(np.int64) * n_i + coo.col)
        # 유저가 파일 이웃(+-1, +-3)과 겹치는 아이템 수: 같은 상품 페이지에서 발견된 무리 안에 있는지
        ov = {}
        for d in range(1, 4):
            a, b = self.order[:-d], self.order[d:]
            ov[d] = np.asarray(X[a].multiply(X[b]).sum(axis=1)).ravel().astype(np.float32)
        self.user_ov = {}
        for W in (1, 3):
            s = np.zeros(n_u, np.float32)
            for d in range(1, W + 1):
                np.add.at(s, self.order[:-d], ov[d])
                np.add.at(s, self.order[d:], ov[d])
            self.user_ov[W] = s
        # 아이템 구매자들이 파일에서 몰려 있는 정도: 구매자 중 다른 구매자가 +-W칸 안에 있는 비율
        p = self.pos[coo.row]
        o = np.lexsort((p, coo.col))
        it, ps = coo.col[o], p[o]
        big = np.int64(10 ** 9)
        gap_prev = np.where(np.r_[False, it[1:] == it[:-1]], np.r_[big, np.diff(ps)], big)
        gap_next = np.where(np.r_[it[:-1] == it[1:], False], np.r_[np.diff(ps), big], big)
        g = np.minimum(gap_prev, gap_next)
        n_b = np.maximum(np.bincount(it, minlength=n_i), 1)
        self.item_clu = {W: np.bincount(it, weights=(g <= W), minlength=n_i) / n_b for W in (3, 10)}
        # 거의 같은 상품: TF-IDF 유사도가 thr 이상인 아이템을 유저가 몇 개 샀나
        self.text_thr = []
        ts = [m for name, _, m in self.comps if name.startswith("textsim") and getattr(m, "field", "") == "text"]
        if ts:
            for thr in (0.3, 0.5, 0.7):
                St = ts[0].S.copy()
                St.data = (St.data >= thr).astype(np.float32)
                St.eliminate_zeros()
                self.text_thr.append((thr, St.tocsr()))

    def batch(self, users, n_cand):
        X = self.X
        raw, C = {}, {}
        S = np.tile(self.tie, (len(users), 1))
        for name, w, m in self.comps:
            s = np.asarray(m.score(users, X), dtype=np.float32)
            s = np.where(np.isfinite(s), s, 0.0).astype(np.float32)
            raw[name], C[name] = s, rownorm(s)
            if w:
                S += w * C[name]
        Xu = X[users]
        r, c = Xu.nonzero()
        S[r, c] = -np.inf
        cand = topk(S, n_cand)
        rows = np.arange(len(users))[:, None]
        shape = cand.shape
        f = {}
        f["blend"] = S[rows, cand]
        f["blend_rel"] = f["blend"] / np.maximum(f["blend"][:, :1], 1e-9)
        f["blend_rank"] = np.broadcast_to(np.arange(n_cand, dtype=np.float32), shape)
        for name in raw:
            v = C[name][rows, cand]
            f[f"n_{name}"] = v
            f[f"raw_{name}"] = raw[name][rows, cand]
            f[f"k_{name}"] = np.argsort(np.argsort(-v, axis=1, kind="stable"), axis=1).astype(np.float32)
        f["logpop"] = np.log1p(self.pop[cand])
        f["pop_diff"] = f["logpop"] - self.user_logpop[users][:, None]
        un = self.user_n[users][:, None]
        f["user_n"] = np.broadcast_to(un, shape)
        if self.ease_B is not None:
            lens = np.diff(Xu.indptr)
            L = int(lens.max())
            mask = np.arange(L)[None, :] < lens[:, None]
            pad = np.zeros((len(users), L), np.int64)
            pad[mask] = Xu.indices
            V = np.where(mask[:, :, None], self.ease_B[pad[:, :, None], cand[:, None, :]], -np.inf)
            f["ease_max"] = V.max(axis=1)
            f["ease_cnt"] = (V > 0.003).sum(axis=1).astype(np.float32)
        for col, (codes, L) in self.attr.items():
            P = (Xu @ L).toarray()
            cc = codes[cand]
            v = np.where(cc >= 0, P[rows, np.maximum(cc, 0)], np.nan)
            f[f"same_{col}"] = v
            f[f"same_{col}_frac"] = v / un
        f["price"] = self.logprice[cand]
        f["price_diff"] = self.logprice[cand] - self.user_price[users][:, None]
        f["item_rating"] = self.item_rating[cand]
        f["user_rating"] = np.broadcast_to(self.user_rating[users][:, None], shape)
        f["dt_med"] = self.user_t["median"][users][:, None] - self.item_t["median"][cand]
        f["item_first_minus_user_last"] = self.item_t["min"][cand] - self.user_t["max"][users][:, None]
        f["user_first_minus_item_last"] = self.user_t["min"][users][:, None] - self.item_t["max"][cand]
        f["item_span"] = self.item_t["max"][cand] - self.item_t["min"][cand]
        f["user_span"] = np.broadcast_to((self.user_t["max"] - self.user_t["min"])[users][:, None], shape)
        lo, hi = self.user_m[0][users][:, None], self.user_m[1][users][:, None]
        inwin = self.cum[cand, hi + 1] - self.cum[cand, lo]
        f["inwin"] = np.log1p(inwin)
        f["inwin_frac"] = inwin / np.maximum(self.pop[cand], 1)
        # 파일 이웃: +-W칸 안에서 이 아이템을 산 이웃 수, 가장 가까운 이웃까지의 거리
        pos = self.pos[users]
        cnt = np.zeros(shape, np.float32)
        near = np.full(shape, 99.0, np.float32)
        for d in range(1, 11):
            for sgn in (-1, 1):
                q = pos + sgn * d
                ok = (q >= 0) & (q < self.n_u)
                v = self.order[np.clip(q, 0, self.n_u - 1)]
                hit = self.has(v[:, None], cand) & ok[:, None]
                cnt += hit
                near = np.where(hit & (near > d), d, near)
            if d in (1, 3, 10):
                f[f"nb{d}"] = cnt.copy()
        f["nb_near"] = near
        f["fpos"] = np.broadcast_to((pos / self.n_u)[:, None], shape).astype(np.float32)
        for W, s in self.user_ov.items():
            f[f"u_ov{W}"] = np.broadcast_to(s[users][:, None], shape)
        for W, s in self.item_clu.items():
            f[f"i_clu{W}"] = s[cand]
        for thr, St in self.text_thr:
            f[f"txt{thr}"] = (Xu @ St).toarray()[rows, cand]
        F = np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in f.values()], axis=1)
        return cand, F, list(f)

    def has(self, users, items):
        """users, items: 같은 shape의 index 배열 -> train에서 그 유저가 그 아이템을 샀는지 (bool)."""
        q = users.astype(np.int64) * self.n_i + items
        idx = np.minimum(np.searchsorted(self.keys, q), len(self.keys) - 1)
        return self.keys[idx] == q


def build(fb, users, n_cand, batch=1024):
    cands, Fs = [], []
    names = None
    for s in range(0, len(users), batch):
        c, F, names = fb.batch(users[s:s + batch], n_cand)
        cands.append(c)
        Fs.append(F)
    return np.vstack(cands), np.vstack(Fs), names


def split_features(cfg, cfg_tag, seed, n_cand, df, enc, test_users, nested_of=None):
    """seed split의 train으로 component를 학습하고 valid 유저의 후보/feature/label을 만든다 (캐시).

    nested_of=s: seed s split의 train(전체의 90%)을 다시 나눈다 (train 81% / valid 9%).
    feature를 '가진 데이터의 90%'로 만들어 학습하고 100%로 만든 feature에 적용하는 제출 상황을 로컬에서 재현할 때 쓴다.
    """
    nest = f"_nest{nested_of}" if nested_of is not None else ""
    path = CACHE / f"{cfg_tag}_N{n_cand}_f{FEAT_V}{nest}_s{seed}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return z["cand"], z["F"], z["y"], z["users"], list(z["names"]), z["gt_n"]
    t0 = time.time()
    split = {**DEFAULTS["split"], "seed": seed}
    if nested_of is not None:
        df = get_split(df, **{**DEFAULTS["split"], "seed": nested_of})[0]
    tr, va, _ = get_split(df, **split)
    X = enc.matrix(tr)
    gt = pd.DataFrame({"u": enc.user_index(va["user_id"]), "i": enc.item_index(va["item_id"])})
    gt = gt[gt["u"].isin(enc.user_index(test_users))].groupby("u")["i"].apply(set)
    users = gt.index.to_numpy()
    print(f"  [seed {seed}] build features ({len(users)} users)", flush=True)
    fb = FeatureBuilder(cfg, X, tr, enc, "valid")
    cand, F, names = build(fb, users, n_cand)
    y = np.array([[i in g for i in row] for row, g in zip(cand, gt.tolist())], dtype=np.int8)
    gt_n = np.array([len(g) for g in gt.tolist()], np.int32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, cand=cand.astype(np.int32), F=F, y=y, users=users, names=np.array(names), gt_n=gt_n)
    print(f"  [seed {seed}] {F.shape}, positives {int(y.sum())}, cand recall@{n_cand} "
          f"{y.sum() / np.minimum(gt_n, 20).sum():.4f} ({time.time() - t0:.0f}s)", flush=True)
    return cand, F, y, users, names, gt_n


def gather(parts, seeds, user_mask, N, keep, objective, neg_rate, rng):
    """여러 split에서 user_mask 유저의 행만 모은다. binary는 오답을 neg_rate만큼 샘플링
    (유저 안 점수 순위는 그대로), lambdarank는 정답이 있는 유저(그룹)만 남긴다 (정답 없는 그룹은 gradient 0)."""
    Fs, ys, group = [], [], []
    for s in seeds:
        F, y = parts[s][1], parts[s][2].ravel()
        if objective == "lambdarank":
            sel = user_mask & (parts[s][2].sum(axis=1) > 0)
            r = np.repeat(sel, N)
            group += [N] * int(sel.sum())
        else:
            r = np.repeat(user_mask, N)
            if neg_rate < 1.0:
                r &= (y > 0) | (rng.random(len(y)) < neg_rate)
        idx = np.flatnonzero(r)
        Fs.append(F[idx][:, keep])
        ys.append(y[idx])
    return np.vstack(Fs), np.concatenate(ys), (group or None)


def recall_from_scores(pred, cand, y, gt_n, k=20):
    order = np.argsort(-pred.reshape(cand.shape), axis=1, kind="stable")[:, :k]
    hits = np.take_along_axis(y, order, axis=1).sum()
    return hits / np.minimum(gt_n, k).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--eval-seed", type=int, default=42)
    ap.add_argument("--cands", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--neg-rate", type=float, default=1.0,
                    help="학습 시 오답 후보를 이 비율만 사용 (binary). 유저 안 점수 순위는 그대로라 속도만 빨라짐")
    ap.add_argument("--objective", default="binary", choices=["binary", "lambdarank"])
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--min-leaf", type=int, default=300)
    ap.add_argument("--folds", type=int, default=2,
                    help="유저를 K묶음으로 나눠, 묶음 k는 그 유저를 학습에 쓰지 않은 모델로만 평가/제출 (외우기 누출 방지)")
    ap.add_argument("--drop", default="", help="뺄 feature 이름 (쉼표 구분)")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--build-only", type=int, nargs="*", default=None,
                    help="이 seed들의 feature 캐시만 만들고 끝낸다 (여러 프로세스로 나눠 만들 때)")
    ap.add_argument("--nested-of", type=int, default=None,
                    help="학습 split을 이 seed split의 train 안에서 다시 나눈다 (feature 분포 차이 시뮬레이션)")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tag = __import__("pathlib").Path(args.config).stem
    N, k, K = args.cands, 20, args.folds
    t0 = time.time()

    df = data.load_train()
    test_users = data.load_test_users()
    enc = data.Encoder(df, test_users)
    if args.build_only is not None:
        for s in args.build_only:
            split_features(cfg, tag, s, N, df, enc, test_users, nested_of=args.nested_of)
        return
    parts = {s: split_features(cfg, tag, s, N, df, enc, test_users, nested_of=args.nested_of)
             for s in args.train_seeds}
    parts[args.eval_seed] = split_features(cfg, tag, args.eval_seed, N, df, enc, test_users)
    if args.nested_of is not None and args.submit:
        raise SystemExit("--nested-of는 검증 전용 (--submit과 같이 쓰지 않음)")
    names = parts[args.eval_seed][4]
    drop = [d for d in args.drop.split(",") if d]
    keep = [j for j, n in enumerate(names) if n not in drop]
    names_k = [names[j] for j in keep]
    users = parts[args.eval_seed][3]
    assert all((parts[s][3] == users).all() for s in parts), "split마다 유저 순서가 같아야 함"
    n_users = len(users)

    params = {**PARAMS, "learning_rate": args.lr, "num_leaves": args.leaves, "min_data_in_leaf": args.min_leaf}
    if args.objective == "lambdarank":
        params.update(objective="lambdarank", metric="ndcg", ndcg_eval_at=[20], lambdarank_truncation_level=40)
    # 같은 유저의 (유저, 아이템) 정답 쌍이 여러 split에 나오므로, 평가 유저는 학습에서 완전히 뺀다.
    # early stopping 유저도 모든 split에서 학습에서 뺀다.
    rng = np.random.default_rng(0)
    fold = rng.integers(0, K, n_users)
    es_flag = rng.random(n_users) < 0.1
    cand_e, F_e, y_e, _, _, gt_e = parts[args.eval_seed]
    pred = np.zeros(len(F_e), np.float64)
    best_iters, imp = [], np.zeros(len(names_k))
    for f in range(K):
        Ftr, ytr, gtr = gather(parts, args.train_seeds, (fold != f) & ~es_flag, N, keep, args.objective,
                               args.neg_rate, rng)
        Fes, yes, ges = gather(parts, args.train_seeds, (fold != f) & es_flag, N, keep, args.objective, 1.0, rng)
        dtr = lgb.Dataset(Ftr, ytr, group=gtr, feature_name=names_k, free_raw_data=True)
        des = lgb.Dataset(Fes, yes, group=ges, reference=dtr, feature_name=names_k, free_raw_data=True)
        model = lgb.train(params, dtr, num_boost_round=args.rounds, valid_sets=[des],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        rows = np.repeat(fold == f, N)
        pred[rows] = model.predict(F_e[rows][:, keep], num_iteration=model.best_iteration)
        best_iters.append(model.best_iteration)
        imp += model.feature_importance("gain")
        print(f"  fold {f}: train rows {len(ytr)} (pos {int(ytr.sum())}), best_iter {model.best_iteration} "
              f"({time.time() - t0:.0f}s)", flush=True)
        del Ftr, ytr, Fes, yes, dtr, des
    r_rerank = recall_from_scores(pred, cand_e, y_e, gt_e, k)
    r_blend = y_e[:, :k].sum() / np.minimum(gt_e, k).sum()
    imp = pd.Series(imp, index=names_k).sort_values(ascending=False)
    print(f"\n[eval seed {args.eval_seed}, {K}-fold by user] Recall@{k}: blend {r_blend:.5f} -> rerank {r_rerank:.5f} "
          f"(best_iter {best_iters}, {time.time() - t0:.0f}s)")
    print("top features (gain): " + ", ".join(f"{n} {v / imp.sum():.3f}" for n, v in imp.head(15).items()))
    run_id = make_run_id("rerank2", f"N{N}")
    experiments.append(run_id, "valid", "rerank2",
                       {"config": args.config, "cands": N, "train_seeds": args.train_seeds, "folds": K,
                        "best_iter": best_iters, "objective": args.objective, "neg_rate": args.neg_rate, "lr": args.lr,
                        "leaves": args.leaves, "min_leaf": args.min_leaf, "drop": drop, "feat_v": FEAT_V,
                        "blend_valid": round(float(r_blend), 5)}, "random", float(r_rerank), args.note)
    (data.OUTPUTS / "tune").mkdir(parents=True, exist_ok=True)
    imp.to_csv(data.OUTPUTS / "tune" / f"{run_id}_importance.csv", header=["gain"])
    np.save(CACHE / f"pred_{run_id}_eval.npy", pred.astype(np.float32))  # 앙상블 평가용 (scripts/rerank_ens.py)

    if not args.submit:
        return
    # 제출: eval seed까지 모든 split을 쓰되, 유저 묶음 k의 test 추천은 그 유저를 학습에 쓰지 않은 모델로 만든다
    # (같은 유저로 학습하면 test 정답이 학습 split에서 '오답'으로 외워져 불리해짐).
    seeds = [*args.train_seeds, args.eval_seed]
    n_iter = int(np.mean(best_iters) * 1.1)
    models = []
    for f in range(K):  # 최종 모델을 먼저 학습하고 split feature를 지운 뒤 test feature를 만든다 (메모리)
        Ftr, ytr, gtr = gather(parts, seeds, fold != f, N, keep, args.objective, args.neg_rate, rng)
        models.append(lgb.train(params, lgb.Dataset(Ftr, ytr, group=gtr, feature_name=names_k),
                                num_boost_round=n_iter))
        print(f"  final fold {f}: {len(seeds)} splits, {n_iter} rounds ({time.time() - t0:.0f}s)", flush=True)
        del Ftr, ytr
    del parts, F_e, cand_e, y_e
    tu = enc.user_index(test_users)
    full_path = CACHE / f"{tag}_N{N}_f{FEAT_V}_full.npz"
    if full_path.exists():
        z = np.load(full_path, allow_pickle=True)
        cand, F = z["cand"], z["F"]
    else:
        fb = FeatureBuilder(cfg, enc.matrix(df), df, enc, "full")
        cand, F, _ = build(fb, tu, N)
        del fb
        np.savez(full_path, cand=cand.astype(np.int32), F=F)
    pred = np.zeros(cand.shape, np.float64)
    tu_fold = fold[np.searchsorted(users, tu)]
    for f, model in enumerate(models):
        sel = tu_fold == f
        pred[sel] = model.predict(F[np.repeat(sel, N)][:, keep]).reshape(-1, N)
    np.save(CACHE / f"pred_{run_id}_test.npy", pred.astype(np.float32))
    rec = np.take_along_axis(cand, np.argsort(-pred, axis=1, kind="stable")[:, :k], axis=1)
    item_ids = enc.items.to_numpy()[rec]
    out = data.OUTPUTS / "submissions" / f"{run_id}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"user_id": test_users, "item_id": [" ".join(map(str, row)) for row in item_ids]}).to_csv(out, index=False)
    experiments.append(run_id, "submit", "rerank2", {"config": args.config, "cands": N, "n_iter": n_iter,
                                                     "train_seeds": seeds}, "full", None,
                       args.note or f"rerank2 of {args.config}")
    errors = check(out, k=k)
    print(f"\nsubmission: {out}\n" + ("format OK" if not errors else "format ERROR:\n  " + "\n  ".join(errors)))
    print(f"제출 후: python -m src.experiments lb {run_id} <public score>  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
