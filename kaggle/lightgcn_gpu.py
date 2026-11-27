"""Kaggle GPU용 LightGCN 학습 (11-27_exp.md Step 5-3). 이 파일 하나로 동작한다 (repo 코드 불필요).

Kaggle Notebook: 대회 데이터 추가 -> Accelerator: GPU -> 이 파일 내용을 셀에 붙여넣고 실행.
결과 (/kaggle/working):
  lightgcn_valid_user.npy, lightgcn_valid_item.npy  로컬과 같은 split(random 10%, seed 42)의 train으로 학습 -> 로컬 blend 튜닝용
  lightgcn_full_user.npy,  lightgcn_full_item.npy   train 전체로 best epoch만큼 학습 -> 제출용
  lightgcn_log.json                                 config별 valid Recall@20, best epoch
로컬: npy 4개를 data/external/에 넣고 precomputed component로 blend (kaggle/README.md 참고).

로컬 동작 확인: QUICK=1 DATA_DIR=data/raw OUT_DIR=<폴더> python kaggle/lightgcn_gpu.py
"""
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F

# GRID의 config를 차례로 학습해 valid Recall@20이 가장 좋은 config로 전체 재학습한다.
BASE = dict(dim=64, layers=3, lr=2e-3, reg=1e-4, batch_size=4096, epochs=300, eval_every=5, patience=5, seed=42)
GRID = [
    dict(),                       # dim 64, 3 layers, lr 2e-3
    dict(dim=128),
    dict(layers=2, lr=5e-3),
]
SPLIT = dict(ratio=0.1, min_items=3, seed=42)          # src/common.py DEFAULTS와 같아야 함
EXPECTED = dict(n_valid=46752, checksum=16491664987607)  # 로컬 split (sum(user*n_items + item))
QUICK = os.environ.get("QUICK") == "1"                 # 로컬 동작 확인용: config 1개, epoch 2, 배치 3개
OUT = os.environ.get("OUT_DIR", "/kaggle/working")
K = 20


def find_data():
    cands = [os.environ.get("DATA_DIR"), "/kaggle/input/ssu-data-science-amazon-top-k-recommendation",
             *glob.glob("/kaggle/input/*"), *glob.glob("/kaggle/input/*/*")]
    for d in cands:
        if d and os.path.exists(os.path.join(d, "train.csv")):
            return d
    raise FileNotFoundError("train.csv를 찾을 수 없음: 노트북에 대회 데이터를 추가했는지 확인")


def random_split(df, ratio, min_items, seed):
    """src/split.py의 mode=random과 같은 로직 (user별 ratio만큼 랜덤으로 valid)."""
    cols = list(df.columns)
    d = df.drop_duplicates(["user_id", "item_id"]).copy()
    d["_key"] = np.random.default_rng(seed).random(len(d))
    d = d.sort_values(["user_id", "_key"], kind="mergesort")
    rank = d.groupby("user_id").cumcount().to_numpy()
    n = d.groupby("user_id")["item_id"].transform("size").to_numpy()
    k = np.where(n >= min_items, np.maximum(1, np.floor(n * ratio)).astype(int), 0)
    is_valid = rank >= n - k
    return d.loc[~is_valid, cols], d.loc[is_valid, cols]


def interaction_matrix(df, nu, ni):
    X = sp.csr_matrix((np.ones(len(df), np.float32), (df["user_id"].to_numpy(), df["item_id"].to_numpy())), shape=(nu, ni))
    X.data[:] = 1.0
    return X


class LightGCN(torch.nn.Module):
    """LightGCN (He et al., 2020): A_hat = D^-1/2 A D^-1/2, layer 평균 임베딩, BPR loss."""

    def __init__(self, X, dim, layers, dev):
        super().__init__()
        nu, ni = X.shape
        self.nu, self.layers = nu, layers
        R = X.tocoo()
        row = np.concatenate([R.row, R.col + nu])
        col = np.concatenate([R.col + nu, R.row])
        deg = np.bincount(row, minlength=nu + ni).astype(np.float32)
        dinv = np.zeros_like(deg)
        dinv[deg > 0] = deg[deg > 0] ** -0.5
        idx = torch.as_tensor(np.stack([row, col]), dtype=torch.long)
        val = torch.as_tensor(dinv[row] * dinv[col], dtype=torch.float32)
        self.A = torch.sparse_coo_tensor(idx, val, (nu + ni, nu + ni)).coalesce().to(dev)
        self.emb = torch.nn.Embedding(nu + ni, dim).to(dev)
        torch.nn.init.normal_(self.emb.weight, std=0.1)

    def propagate(self):
        h = self.emb.weight
        out = [h]
        for _ in range(self.layers):
            h = torch.sparse.mm(self.A, h)
            out.append(h)
        final = torch.stack(out).mean(0)
        return final[: self.nu], final[self.nu:]


@torch.no_grad()
def evaluate(model, X, users, gts, dev, batch=4096):
    U, I = model.propagate()
    hits = denom = 0
    for s in range(0, len(users), batch):
        u = users[s:s + batch]
        S = U[torch.tensor(u, device=dev)] @ I.T
        r, c = X[u].nonzero()
        S[torch.as_tensor(r, device=dev), torch.as_tensor(c, device=dev)] = -float("inf")
        top = torch.topk(S, K, dim=1).indices.cpu().numpy()
        for row, g in zip(top, gts[s:s + batch]):
            hits += len(g.intersection(row.tolist()))
            denom += min(len(g), K)
    return hits / denom


def train(X, cfg, dev, eval_data=None, epochs=None):
    """eval_data=(users, gts)가 있으면 early stopping, 없으면 epochs만큼 학습."""
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    model = LightGCN(X, cfg["dim"], cfg["layers"], dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    coo = X.tocoo()
    users, items = coo.row, coo.col
    nu, ni = X.shape
    best, best_ep, best_state, bad = -1.0, 0, None, 0
    n_ep = 2 if QUICK else (epochs or cfg["epochs"])
    for ep in range(1, n_ep + 1):
        t = time.time()
        perm = rng.permutation(len(users))
        total = 0.0
        batches = range(0, len(perm), cfg["batch_size"])
        for bi, s in enumerate(batches):
            if QUICK and bi >= 3:
                break
            b = perm[s:s + cfg["batch_size"]]
            u = torch.as_tensor(users[b], dtype=torch.long, device=dev)
            p = torch.as_tensor(items[b], dtype=torch.long, device=dev)
            n = torch.randint(0, ni, (len(b),), device=dev)
            ue, ie = model.propagate()
            eu = ue[u]
            bpr = -F.logsigmoid((eu * ie[p]).sum(1) - (eu * ie[n]).sum(1)).mean()
            e0 = model.emb.weight
            l2 = (e0[u].pow(2).sum() + e0[nu + p].pow(2).sum() + e0[nu + n].pow(2).sum()) / (2 * len(b))
            loss = bpr + cfg["reg"] * l2
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        msg = f"  epoch {ep:3d} loss {total / len(users):.4f} ({time.time() - t:.1f}s)"
        if eval_data is not None and (QUICK or ep % cfg["eval_every"] == 0):
            score = evaluate(model, X, *eval_data, dev)
            msg += f"  valid Recall@20 {score:.5f}"
            if score > best:
                best, best_ep, bad = score, ep, 0
                best_state = model.emb.weight.detach().clone()
            else:
                bad += 1
        print(msg, flush=True)
        if eval_data is not None and bad >= cfg["patience"]:
            print(f"  early stop (best epoch {best_ep})", flush=True)
            break
    if best_state is not None:
        with torch.no_grad():
            model.emb.weight.copy_(best_state)
    with torch.no_grad():
        U, I = model.propagate()
    return U.cpu().numpy(), I.cpu().numpy(), best, best_ep


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = find_data()
    os.makedirs(OUT, exist_ok=True)
    print(f"device={dev} data={data_dir} out={OUT} quick={QUICK}", flush=True)
    df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    nu, ni = int(df["user_id"].max()) + 1, int(df["item_id"].max()) + 1

    tr, va = random_split(df, **SPLIT)
    checksum = int((va["user_id"].astype(np.int64) * ni + va["item_id"]).sum())
    print(f"split: train {len(tr)} valid {len(va)} checksum {checksum}", flush=True)
    if (len(va), checksum) != (EXPECTED["n_valid"], EXPECTED["checksum"]):
        print("WARNING: 로컬 split과 다름 -> valid 임베딩으로 로컬 튜닝한 결과를 믿을 수 없음", flush=True)

    X_tr = interaction_matrix(tr, nu, ni)
    gt = va.groupby("user_id")["item_id"].apply(set)
    eval_data = (gt.index.to_numpy(), gt.tolist())

    log, best = [], None
    for g in (GRID[:1] if QUICK else GRID):
        cfg = {**BASE, **g}
        print(f"\n[config] {cfg}", flush=True)
        t = time.time()
        U, I, score, ep = train(X_tr, cfg, dev, eval_data=eval_data)
        log.append({"config": cfg, "valid_recall@20": score, "best_epoch": ep, "minutes": round((time.time() - t) / 60, 1)})
        print(f"  -> valid Recall@20 {score:.5f} (best epoch {ep})", flush=True)
        if best is None or score > best[0]:
            best = (score, cfg, ep, U, I)

    score, cfg, ep, U, I = best
    np.save(os.path.join(OUT, "lightgcn_valid_user.npy"), U.astype(np.float32))
    np.save(os.path.join(OUT, "lightgcn_valid_item.npy"), I.astype(np.float32))
    print(f"\n[full] best config {cfg}, {ep} epochs on full train", flush=True)
    U, I, _, _ = train(interaction_matrix(df.drop_duplicates(["user_id", "item_id"]), nu, ni), cfg, dev, epochs=max(ep, 1))
    np.save(os.path.join(OUT, "lightgcn_full_user.npy"), U.astype(np.float32))
    np.save(os.path.join(OUT, "lightgcn_full_item.npy"), I.astype(np.float32))
    with open(os.path.join(OUT, "lightgcn_log.json"), "w") as f:
        json.dump({"runs": log, "best": {"config": cfg, "valid_recall@20": score, "best_epoch": ep}}, f, indent=2)
    print(f"\ndone. best valid Recall@20 {score:.5f} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
