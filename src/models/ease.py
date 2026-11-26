import numpy as np

from .base import Recommender


class EASE(Recommender):
    """EASE (Steck, 2019): item-item weight B를 closed form으로 구함.

    B = I - P * diag(1/diag(P)),  P = (X^T X + reg*I)^-1,  diag(B)=0
    item 수가 max_items보다 많으면 인기 상위 max_items개만 후보로 사용.
    """

    def __init__(self, reg=500.0, max_items=10000):
        self.reg = reg
        self.max_items = max_items

    def _fit(self, X, eval_fn=None):
        n = X.shape[1]
        if self.max_items and n > self.max_items:
            idx = np.sort(np.argsort(-self.popularity, kind="stable")[: self.max_items])
        else:
            idx = np.arange(n)
        self.idx = idx
        Xs = X[:, idx].astype(np.float32)
        G = (Xs.T @ Xs).toarray()
        G[np.diag_indices_from(G)] += self.reg
        P = np.linalg.inv(G)
        del G
        B = P / (-np.diag(P))
        B[np.diag_indices_from(B)] = 0.0
        self.B = B.astype(np.float32)

    def score(self, users, X):
        s = np.full((len(users), X.shape[1]), -np.inf, dtype=np.float32)
        s[:, self.idx] = X[users][:, self.idx] @ self.B
        return s
