import numpy as np
import scipy.sparse as sp

from .base import Recommender


def keep_topk_per_row(M: sp.csr_matrix, k: int) -> sp.csr_matrix:
    M = M.tocsr()
    rows, cols, vals = [], [], []
    for i in range(M.shape[0]):
        s, e = M.indptr[i], M.indptr[i + 1]
        if s == e:
            continue
        d, c = M.data[s:e], M.indices[s:e]
        if e - s > k:
            sel = np.argpartition(-d, k - 1)[:k]
            d, c = d[sel], c[sel]
        rows.append(np.full(len(c), i))
        cols.append(c)
        vals.append(d)
    if not rows:
        return sp.csr_matrix(M.shape, dtype=np.float32)
    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=M.shape, dtype=np.float32,
    )


class ItemKNN(Recommender):
    """item-item cosine similarity 기반 CF.

    score(u, j) = sum_i x_ui * sim(i, j), j마다 가장 비슷한 k개 item만 사용.
    """

    def __init__(self, k=100, shrink=10.0):
        self.k = k
        self.shrink = shrink

    def _fit(self, X, eval_fn=None):
        X = X.astype(np.float32)
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=0)).ravel())
        co = (X.T @ X).tocoo()
        off = co.row != co.col
        row, col, val = co.row[off], co.col[off], co.data[off]
        val = val / (norms[row] * norms[col] + self.shrink)
        sim = sp.csr_matrix((val, (row, col)), shape=co.shape, dtype=np.float32)
        # sim은 대칭 -> row i에서 top-k를 남기고 transpose하면 column j에 j의 neighbor가 남음
        self.W = keep_topk_per_row(sim, self.k).T.tocsr()

    def score(self, users, X):
        return (X[users] @ self.W).toarray()
