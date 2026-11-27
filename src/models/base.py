"""모든 모델의 공통 interface: fit(X) -> score(users, X) -> recommend(users, X, k)."""
import numpy as np
import scipy.sparse as sp


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    part = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-part, axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1)


class Recommender:
    best_epoch = None  # early stopping을 쓰는 모델만 채움

    def fit(self, X: sp.csr_matrix, eval_fn=None, ctx=None):
        """ctx: X 외의 정보가 필요한 모델용. {"df": 학습 interaction df, "enc": Encoder, "stage": "valid"|"full"}"""
        self.n_users, self.n_items = X.shape
        self.popularity = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)
        self.ctx = ctx or {}
        self._fit(X, eval_fn)
        return self

    def _fit(self, X, eval_fn):
        raise NotImplementedError

    def score(self, users: np.ndarray, X: sp.csr_matrix) -> np.ndarray:
        """(len(users), n_items) dense score."""
        raise NotImplementedError

    def recommend(self, users, X, k=20, batch_size=512) -> np.ndarray:
        """user별 top-k item index. 이미 interaction한 item은 제외, interaction이 없는 user는 인기순."""
        users = np.asarray(users)
        out = np.empty((len(users), k), dtype=np.int64)
        for start in range(0, len(users), batch_size):
            u = users[start:start + batch_size]
            s = np.array(self.score(u, X), dtype=np.float32)
            Xu = X[u]
            cold = np.diff(Xu.indptr) == 0
            if cold.any():
                s[cold] = self.popularity
            r, c = Xu.nonzero()
            s[r, c] = -np.inf
            out[start:start + len(u)] = topk(s, k)
        return out
