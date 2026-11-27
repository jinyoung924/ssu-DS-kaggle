import numpy as np
import scipy.sparse as sp

from .base import Recommender
from .itemknn import keep_topk_per_row


class RP3beta(Recommender):
    """RP3beta (Paudel et al., 2017): item -> user -> item random walk 확률, 도착 아이템 인기도^beta로 감점.

    W_ij = (sum_u P(u|i)^alpha * P(j|u)^alpha) / pop_j^beta, 출발 아이템 i마다 top-k만 사용.
    """

    def __init__(self, alpha=1.0, beta=0.5, k=100):
        self.alpha, self.beta, self.k = alpha, beta, k

    def _fit(self, X, eval_fn=None):
        X = X.astype(np.float32).tocsr()
        du = np.maximum(np.asarray(X.sum(axis=1)).ravel(), 1.0)
        di = np.maximum(np.asarray(X.sum(axis=0)).ravel(), 1.0)
        Pui = (sp.diags(1.0 / du) @ X).tocsr()          # user -> item
        Piu = (sp.diags(1.0 / di) @ X.T).tocsr()        # item -> user
        if self.alpha != 1.0:
            Pui, Piu = Pui.power(self.alpha), Piu.power(self.alpha)
        W = (Piu @ Pui @ sp.diags(di ** -self.beta)).tocoo()
        off = W.row != W.col
        W = sp.csr_matrix((W.data[off], (W.row[off], W.col[off])), shape=W.shape, dtype=np.float32)
        self.W = keep_topk_per_row(W, self.k)

    def score(self, users, X):
        return (X[users] @ self.W).toarray()
