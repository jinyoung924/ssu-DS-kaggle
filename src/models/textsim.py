import re
from collections import Counter

import numpy as np
import scipy.sparse as sp

from .. import data, features
from .base import Recommender
from .itemknn import topk_sim


def _dense(x):
    return x.toarray() if sp.issparse(x) else x


class TextSim(Recommender):
    """상품 텍스트 유사도 (content 기반). score = X_u @ S, S는 아이템마다 코사인 top-k.

    source=tfidf: 메타데이터 field(text: title + store + categories + features, title: 상품명만)로 로컬 TF-IDF
    source=emb: kaggle/text_embed_gpu.py가 GPU로 만든 임베딩 (.npy, item_id 순서)
    """

    def __init__(self, source="tfidf", field="text", k=50, emb_path="data/external/text_emb.npy", min_df=2, max_df=0.2):
        self.source, self.field, self.k, self.emb_path = source, field, k, emb_path
        self.min_df, self.max_df = min_df, max_df

    def _fit(self, X, eval_fn=None):
        enc = self.ctx["enc"]
        E = self._tfidf(enc) if self.source == "tfidf" else self._emb(enc)
        self.S = topk_sim(lambda s, e: _dense(E[s:e] @ E.T), X.shape[1], self.k)

    def _tfidf(self, enc):
        texts = features.item_attr(enc, self.field)
        tok = re.compile(r"[a-z0-9]+")
        vocab, rows, cols, cnts = {}, [], [], []
        for r, t in enumerate(texts):
            for w, c in Counter(tok.findall(str(t).lower())).items():
                rows.append(r)
                cols.append(vocab.setdefault(w, len(vocab)))
                cnts.append(c)
        n = len(texts)
        T = sp.csr_matrix((np.log1p(np.asarray(cnts, np.float32)), (rows, cols)), shape=(n, len(vocab)))
        df = np.bincount(cols, minlength=len(vocab))
        keep = np.flatnonzero((df >= self.min_df) & (df <= self.max_df * n))
        T = (T[:, keep] @ sp.diags(np.log(n / df[keep]).astype(np.float32))).tocsr()
        norm = np.sqrt(np.asarray(T.multiply(T).sum(axis=1)).ravel())
        return (sp.diags(1.0 / np.maximum(norm, 1e-9)) @ T).tocsr()

    def _emb(self, enc):
        E = np.load(data.ROOT / self.emb_path).astype(np.float32)[enc.items.to_numpy()]
        return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)

    def score(self, users, X):
        return (X[users] @ self.S).toarray()
