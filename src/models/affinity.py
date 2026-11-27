import numpy as np
import pandas as pd
import scipy.sparse as sp

from .. import features
from .base import Recommender


class Affinity(Recommender):
    """유저의 속성 분포 x 아이템 인기도 (content 기반 선호).

    score(u, i) = p_u(attr(i)) * pop(i)^pop_power
    p_u(a) = 유저가 산 아이템 중 속성이 a인 비율. attr: store / leaf / path / dept
    """

    def __init__(self, attr="leaf", pop_power=1.0):
        self.attr, self.pop_power = attr, pop_power

    def _fit(self, X, eval_fn=None):
        a = pd.Series(features.item_attr(self.ctx["enc"], self.attr)).replace("", np.nan)
        codes, uniq = pd.factorize(a)
        ok = codes >= 0
        L = sp.csr_matrix(
            (np.ones(ok.sum(), np.float32), (np.flatnonzero(ok), codes[ok])), shape=(X.shape[1], len(uniq))
        )
        self.L, self.LT = L, L.T.tocsr()
        self.w = (self.popularity / self.popularity.max()) ** self.pop_power

    def score(self, users, X):
        P = (X[users] @ self.L).tocsr()
        P = sp.diags(1.0 / np.maximum(np.asarray(P.sum(axis=1)).ravel(), 1.0)) @ P
        return (P @ self.LT).toarray() * self.w
