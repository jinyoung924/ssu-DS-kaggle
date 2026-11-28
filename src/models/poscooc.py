import numpy as np
import scipy.sparse as sp

from .. import features
from .base import Recommender
from .ease import EASE
from .itemknn import topk_sim


class PositionCooc(Recommender):
    """Ordering Artifact 심화: 리뷰 파일 위치 구간을 '가짜 세션'으로 보고 만든 아이템-아이템 유사도.

    Z[구간, i] = 그 구간(파일 위치 bin명) 유저 중 i 구매자 수. overlap이면 bin/2만큼 밀린 구간도 추가.
    base=itemknn: Z 기준 cosine top-k, base=ease: Z의 Gram으로 EASE.  score = X_u @ W
    """

    def __init__(self, bin=25, overlap=True, base="itemknn", k=100, shrink=10.0, reg=500.0):
        self.bin, self.overlap, self.base = bin, overlap, base
        self.k, self.shrink, self.reg = k, shrink, reg

    def _fit(self, X, eval_fn=None):
        n_i = X.shape[1]
        G = position_gram(X, self.ctx["enc"], self.bin, self.overlap)
        if self.base == "ease":
            self.W = EASE(reg=self.reg)._solve(G)
        else:
            d = np.sqrt(np.diag(G)).astype(np.float32)
            self.W = topk_sim(lambda s, e: G[s:e] / (d[s:e, None] * d[None, :] + self.shrink), n_i, self.k)

    def score(self, users, X):
        s = X[users] @ self.W
        return s if isinstance(s, np.ndarray) else s.toarray()


def position_gram(X, enc, bin=25, overlap=True) -> np.ndarray:
    """리뷰 파일 위치 bin명 구간을 가짜 세션으로 본 item x item Gram (dense). Z[구간, i] = 그 구간의 i 구매자 수."""
    order = features.user_file_order(enc)
    n_u = X.shape[0]
    pos = np.full(n_u, -1, np.int64)
    pos[order] = np.arange(len(order))
    ok = np.flatnonzero(pos >= 0)
    Z = []
    for off in ([0, bin // 2] if overlap else [0]):
        b = (pos[ok] + off) // bin
        M = sp.csr_matrix((np.ones(len(ok), np.float32), (b, ok)), shape=(b.max() + 1, n_u))
        Z.append((M @ X).toarray())
    Z = np.vstack(Z).astype(np.float32)
    return Z.T @ Z  # 같은 파일 구간에 구매자가 함께 등장한 정도
