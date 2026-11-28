"""콘텐츠 결합 EASE (CEASE, Jeunen et al. 2020의 아이디어를 시간 감쇠 EASE에 적용).

아이템 메타데이터를 "가상의 유저"로 보고 Gram에 더한다:
  G = G_time + alpha * S_text + beta * S_brand
  S_text : TF-IDF 코사인 top-k를 대칭화한 행렬 ((S + S^T) / 2)
  S_brand: 같은 브랜드면 1 / (브랜드 아이템 수)
따로 학습한 점수를 선형으로 섞는 blend와 달리, EASE가 구매 동시발생과 콘텐츠 유사도를 함께 보고 가중치를 푼다.
구매 기록이 적은 아이템일수록 콘텐츠 쪽 비중이 자연스럽게 커진다.
"""
import numpy as np
import pandas as pd

from .. import features
from .poscooc import position_gram
from .textsim import TextSim
from .timeaware import EASETime


class CEASE(EASETime):
    """pos_gamma > 0이면 Ordering Artifact도 Gram에 더한다: G += pos_gamma * (파일 위치 구간 동시등장 Gram)."""

    def __init__(self, reg=500.0, tau_days=365.0, reg_pop=0.0, user_pow=0.0, alpha=5.0, text_k=50, field="text",
                 beta=0.0, pos_gamma=0.0, pos_bin=25):
        super().__init__(reg=reg, tau_days=tau_days, reg_pop=reg_pop, user_pow=user_pow)
        self.alpha, self.text_k, self.field, self.beta = alpha, text_k, field, beta
        self.pos_gamma, self.pos_bin = pos_gamma, pos_bin

    def _fit(self, X, eval_fn=None):
        n = X.shape[1]
        G = self._gram(n)
        if self.pos_gamma:
            G += self.pos_gamma * position_gram(X, self.ctx["enc"], self.pos_bin)
        if self.alpha:
            S = TextSim(source="tfidf", field=self.field, k=self.text_k).fit(X, ctx=self.ctx).S
            S = ((S + S.T) * 0.5).tocoo()
            G[S.row, S.col] += self.alpha * S.data
        if self.beta:
            codes, _ = pd.factorize(pd.Series(features.item_attr(self.ctx["enc"], "store")).replace("", np.nan))
            for g in pd.Series(np.arange(n))[codes >= 0].groupby(codes[codes >= 0]):
                idx = g[1].to_numpy()
                if len(idx) > 1:
                    G[np.ix_(idx, idx)] += self.beta / len(idx)
        self.idx = np.arange(n)
        self.B = self._solve(G)
