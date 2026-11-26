import numpy as np

from .base import Recommender


class Popular(Recommender):
    """train interaction 수 기준 인기순 (하한선 baseline)."""

    def _fit(self, X, eval_fn=None):
        pass

    def score(self, users, X):
        return np.tile(self.popularity, (len(users), 1))
