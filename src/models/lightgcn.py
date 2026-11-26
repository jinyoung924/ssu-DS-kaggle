"""LightGCN (He et al., 2020) + BPR loss."""
import numpy as np
import torch
import torch.nn.functional as F

from .base import Recommender


def pick_device(pref="auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LightGCN(Recommender):
    def __init__(self, dim=64, layers=3, lr=1e-3, reg=1e-4, epochs=100, batch_size=4096,
                 eval_every=5, patience=3, device="auto", seed=42):
        self.dim, self.layers, self.lr, self.reg = dim, layers, lr, reg
        self.epochs, self.batch_size = epochs, batch_size
        self.eval_every, self.patience = eval_every, patience
        self.device, self.seed = device, seed

    def _build_graph(self, X):
        """A_hat = D^-1/2 A D^-1/2,  A = [[0, R], [R^T, 0]]."""
        R = X.tocoo()
        nu = X.shape[0]
        row = np.concatenate([R.row, R.col + nu])
        col = np.concatenate([R.col + nu, R.row])
        n = sum(X.shape)
        deg = np.bincount(row, minlength=n).astype(np.float32)
        dinv = np.zeros_like(deg)
        dinv[deg > 0] = deg[deg > 0] ** -0.5
        val = dinv[row] * dinv[col]
        self.g_row = torch.as_tensor(row, dtype=torch.long, device=self.dev)
        self.g_col = torch.as_tensor(col, dtype=torch.long, device=self.dev)
        self.g_val = torch.as_tensor(val, dtype=torch.float32, device=self.dev)
        self.A = None
        if self.dev.type != "mps":  # MPS는 sparse mm 지원이 불안정 -> index_add로 대체
            self.A = torch.sparse_coo_tensor(
                torch.stack([self.g_row, self.g_col]), self.g_val, (n, n)
            ).coalesce()

    def _propagate(self):
        h = self.emb.weight
        out = [h]
        for _ in range(self.layers):
            if self.A is not None:
                h = torch.sparse.mm(self.A, h)
            else:
                h = torch.zeros_like(h).index_add_(0, self.g_row, h[self.g_col] * self.g_val[:, None])
            out.append(h)
        final = torch.stack(out).mean(0)
        return final[: self.n_users], final[self.n_users:]

    def _cache(self):
        with torch.no_grad():
            ue, ie = self._propagate()
        self.U, self.I = ue.cpu().numpy(), ie.cpu().numpy()

    def _fit(self, X, eval_fn=None):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.dev = pick_device(self.device)
        nu, ni = X.shape
        print(f"[LightGCN] device={self.dev}, users={nu}, items={ni}, nnz={X.nnz}")
        self._build_graph(X)
        self.emb = torch.nn.Embedding(nu + ni, self.dim).to(self.dev)
        torch.nn.init.normal_(self.emb.weight, std=0.1)
        opt = torch.optim.Adam(self.emb.parameters(), lr=self.lr)

        coo = X.tocoo()
        users, items = coo.row, coo.col
        best, best_state, bad = -1.0, None, 0
        for ep in range(1, self.epochs + 1):
            perm = rng.permutation(len(users))
            total = 0.0
            for s in range(0, len(perm), self.batch_size):
                b = perm[s:s + self.batch_size]
                u = torch.as_tensor(users[b], dtype=torch.long, device=self.dev)
                p = torch.as_tensor(items[b], dtype=torch.long, device=self.dev)
                n = torch.randint(0, ni, (len(b),), device=self.dev)
                ue, ie = self._propagate()
                eu = ue[u]
                bpr = -F.logsigmoid((eu * ie[p]).sum(1) - (eu * ie[n]).sum(1)).mean()
                e0 = self.emb.weight
                l2 = (e0[u].pow(2).sum() + e0[nu + p].pow(2).sum() + e0[nu + n].pow(2).sum()) / (2 * len(b))
                loss = bpr + self.reg * l2
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            msg = f"[LightGCN] epoch {ep:3d} loss {total / len(users):.4f}"

            if eval_fn is not None and self.eval_every and ep % self.eval_every == 0:
                self._cache()
                score = eval_fn(self)
                msg += f"  valid Recall@20 {score:.5f}"
                if score > best:
                    best, bad, self.best_epoch = score, 0, ep
                    best_state = self.emb.weight.detach().clone()
                else:
                    bad += 1
            print(msg, flush=True)
            if eval_fn is not None and bad >= self.patience:
                print(f"[LightGCN] early stop (best epoch {self.best_epoch})")
                break

        if best_state is not None:
            with torch.no_grad():
                self.emb.weight.copy_(best_state)
        self._cache()

    def score(self, users, X):
        return self.U[users] @ self.I.T
