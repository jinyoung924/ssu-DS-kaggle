import numpy as np
import scipy.sparse as sp

from .. import features
from .base import Recommender


class FileOrderNeighbor(Recommender):
    """Ordering Artifact (공지사항): train_reviews.jsonl에서 가까이 등장한 유저들의 구매 빈도.

    score(u, i) = sum_{v: 파일 위치 차이 d <= window} k(d) * x_vi
    kernel: uniform(1) / gauss(exp(-d^2/2sigma^2)) / tri(1-(d-1)/window) / inv(1/d)
    side: both / prev(파일에서 앞쪽 유저만) / next(뒤쪽 유저만)
    """

    def __init__(self, window=20, kernel="uniform", side="both", sigma=None):
        self.window, self.kernel, self.side = window, kernel, side
        self.sigma = sigma or window / 2

    def _weight(self, d):
        if self.kernel == "uniform":
            return 1.0
        if self.kernel == "gauss":
            return float(np.exp(-0.5 * (d / self.sigma) ** 2))
        if self.kernel == "tri":
            return 1.0 - (d - 1) / self.window
        if self.kernel == "inv":
            return 1.0 / d
        raise ValueError(f"unknown kernel: {self.kernel}")

    def _fit(self, X, eval_fn=None):
        order = features.user_file_order(self.ctx["enc"])
        rows, cols, vals = [], [], []
        for d in range(1, self.window + 1):
            a, b = order[:-d], order[d:]  # b는 파일에서 a보다 d칸 뒤
            w = np.full(len(a), self._weight(d), np.float32)
            if self.side in ("both", "next"):
                rows.append(a), cols.append(b), vals.append(w)
            if self.side in ("both", "prev"):
                rows.append(b), cols.append(a), vals.append(w)
        n = X.shape[0]
        self.A = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
        )

    def score(self, users, X):
        return (self.A[users] @ X).toarray()
