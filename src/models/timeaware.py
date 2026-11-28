"""시간 정보를 쓰는 모델 (11-27_exp.md Step 6)."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..features import DAY
from .base import Recommender
from .ease import EASE


class AliveWindow(Recommender):
    """유저 활동 기간 ±months 안에 train 구매가 있는 아이템이면 1, 없으면 0.

    blend에서 가산점으로 쓴다 (17년치 데이터라 단종/신상품 효과를 반영하려는 것).
    """

    def __init__(self, months=12):
        self.months = months

    def _fit(self, X, eval_fn=None):
        df, enc = self.ctx["df"], self.ctx["enc"]
        m = (df["timestamp"].to_numpy() // (30 * DAY)).astype(np.int64)
        m -= m.min()
        nm = int(m.max()) + 1
        u, i = enc.user_index(df["user_id"]), enc.item_index(df["item_id"])
        cnt = np.zeros((X.shape[1], nm + 1), np.int32)
        np.add.at(cnt, (i, m + 1), 1)
        self.cum = cnt.cumsum(axis=1)  # cum[i, t] = month < t인 구매 수
        lo = pd.Series(m).groupby(u).min()
        hi = pd.Series(m).groupby(u).max()
        self.lo = np.zeros(X.shape[0], np.int64)
        self.hi = np.full(X.shape[0], nm - 1, np.int64)
        self.lo[lo.index] = np.maximum(lo.to_numpy() - self.months, 0)
        self.hi[hi.index] = np.minimum(hi.to_numpy() + self.months, nm - 1)

    def score(self, users, X):
        c = self.cum[:, self.hi[users] + 1] - self.cum[:, self.lo[users]]  # (items, batch)
        return (c.T > 0).astype(np.float32)


class EASETime(EASE):
    """EASE의 Gram을 시간 감쇠로 가중: G_ij = sum_u n_u^(-user_pow) * exp(-|t_ui - t_uj| / tau).

    user_pow > 0이면 구매가 많은 유저의 쌍을 덜 센다 (헤비 유저가 Gram을 지배하지 않게).
    """

    def __init__(self, reg=500.0, tau_days=365.0, reg_pop=0.0, user_pow=0.0):
        super().__init__(reg=reg, max_items=None, reg_pop=reg_pop)
        self.tau_days = tau_days
        self.user_pow = user_pow

    def _gram(self, n_items):
        df = self.ctx["df"].drop_duplicates(["user_id", "item_id"])
        enc = self.ctx["enc"]
        d = pd.DataFrame({
            "u": enc.user_index(df["user_id"]),
            "i": enc.item_index(df["item_id"]),
            "t": df["timestamp"].to_numpy() / DAY,
        })
        d["w"] = d.groupby("u")["i"].transform("size").to_numpy().astype(np.float64) ** (-self.user_pow)
        p = d.merge(d[["u", "i", "t"]], on="u")  # 유저 내 모든 (i, j) 쌍
        w = np.exp(-np.abs(p["t_x"].to_numpy() - p["t_y"].to_numpy()) / self.tau_days)
        w = (w * p["w"].to_numpy()).astype(np.float32)
        G = sp.coo_matrix((w, (p["i_x"].to_numpy(), p["i_y"].to_numpy())), shape=(n_items, n_items)).toarray()
        del p, w
        return G

    def _fit(self, X, eval_fn=None):
        n = X.shape[1]
        self.idx = np.arange(n)
        self.B = self._solve(self._gram(n))
