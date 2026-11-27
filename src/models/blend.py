import numpy as np

from .base import Recommender


def rownorm(s: np.ndarray) -> np.ndarray:
    """유저(행)별 최댓값으로 나눠 component 간 scale을 맞춘다. -inf 등은 0으로."""
    s = np.where(np.isfinite(s), s, 0.0).astype(np.float32, copy=False)
    m = s.max(axis=1, keepdims=True)
    m[m <= 0] = 1.0
    return s / m


class Blend(Recommender):
    """여러 모델 점수를 유저별로 정규화(rownorm)한 뒤 가중합. 동점은 인기도로.

    components: [{model: ease, weight: 1.0, params: {...}}, ...]  (weight 0인 component는 학습하지 않음)
    """

    def __init__(self, components):
        self.components = components

    def _fit(self, X, eval_fn=None):
        from . import build_model  # 순환 import 방지

        self.models = []
        for c in self.components:
            if not c.get("weight"):
                continue
            m = build_model(c["model"], c.get("params") or {})
            m.fit(X, ctx=self.ctx)
            self.models.append((float(c["weight"]), m))
        self.tie = 1e-6 * self.popularity / max(self.popularity.max(), 1.0)

    def score(self, users, X):
        s = np.tile(self.tie, (len(users), 1))
        for w, m in self.models:
            s += w * rownorm(np.asarray(m.score(users, X), dtype=np.float32))
        return s
