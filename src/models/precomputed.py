import numpy as np

from .. import data
from .base import Recommender


class PrecomputedEmb(Recommender):
    """kaggle/lightgcn_gpu.py가 GPU로 학습한 user/item 임베딩. score = U_u . I_i

    ctx["stage"]에 맞는 파일을 쓴다: valid(split train으로 학습) / full(train 전체로 학습)
      {dir}/{name}_{stage}_user.npy, {dir}/{name}_{stage}_item.npy  (raw id 순서)
    """

    def __init__(self, name="lightgcn", dir="data/external"):
        self.name, self.dir = name, dir

    def _fit(self, X, eval_fn=None):
        stage, enc = self.ctx.get("stage", "valid"), self.ctx["enc"]
        base = data.ROOT / self.dir
        U = np.load(base / f"{self.name}_{stage}_user.npy")
        I = np.load(base / f"{self.name}_{stage}_item.npy")
        self.U = U[enc.users.to_numpy()].astype(np.float32)
        self.I = I[enc.items.to_numpy()].astype(np.float32)

    def score(self, users, X):
        return self.U[users] @ self.I.T
