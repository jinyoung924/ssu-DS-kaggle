"""모델이 쓰는 부가 정보: 아이템 메타데이터, 리뷰 파일의 유저 등장 순서 (Ordering Artifact)."""
import json
import re

import numpy as np
import pandas as pd

from . import data

DAY = 86_400_000  # timestamp는 ms 단위
META_VERSION = 2  # load_meta의 column을 바꾸면 올려서 캐시를 새로 만든다


def _price(v):
    try:
        return float(str(v).replace("$", "").replace(",", ""))
    except ValueError:
        return np.nan


def load_meta() -> pd.DataFrame:
    """item_id index의 메타데이터.

    title, store, leaf(마지막 카테고리), path(상위 4단계 카테고리), dept(Department, 소문자),
    price, text(title + store + categories + features, TF-IDF/임베딩용)
    """
    path = data.PROCESSED / f"meta_v{META_VERSION}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    rows = []
    with open(data.RAW / "item_meta.jsonl", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            cats = [str(c) for c in (m.get("categories") or [])]
            det = m.get("details") if isinstance(m.get("details"), dict) else {}
            feats = [str(x) for x in (m.get("features") or [])]
            store = (m.get("store") or "").strip()
            rows.append({
                "item_id": m["item_id"],
                "title": m.get("title") or "",
                "store": store,
                "leaf": cats[-1] if cats else "",
                "path": " > ".join(cats[:4]),
                "dept": str(det.get("Department", "")).strip().lower(),
                "price": _price(m.get("price")),
                "text": " ".join([m.get("title") or "", store, " ".join(cats), " ".join(feats)]),
            })
    meta = pd.DataFrame(rows).set_index("item_id").sort_index()
    data.PROCESSED.mkdir(parents=True, exist_ok=True)
    meta.to_pickle(path)
    return meta


def item_attr(enc, col) -> np.ndarray:
    """Encoder item index 순서로 정렬한 메타데이터 column."""
    return load_meta()[col].reindex(enc.items).to_numpy()


def load_file_rank() -> pd.Series:
    """train_reviews.jsonl에서 각 user_id가 처음 등장한 순서 (user_id -> 0부터의 위치).

    공지사항의 Ordering Artifact: 이 파일에서 가까이 등장한 유저들은 같은 아이템을 샀을 가능성이 높다.
    대회 문서와 달리 train.csv와 행 순서가 다르므로(train.csv는 user_id 정렬) 반드시 이 파일에서 구한다.
    """
    path = data.PROCESSED / "user_file_rank.pkl"
    if path.exists():
        return pd.read_pickle(path)
    pat = re.compile(r'"user_id":\s*(\d+)')
    users = []
    with open(data.RAW / "train_reviews.jsonl", encoding="utf-8") as f:
        for line in f:
            users.append(int(pat.search(line).group(1)))
    order = pd.unique(np.asarray(users))
    rank = pd.Series(np.arange(len(order)), index=order, name="file_rank")
    data.PROCESSED.mkdir(parents=True, exist_ok=True)
    rank.to_pickle(path)
    return rank


def user_file_order(enc) -> np.ndarray:
    """리뷰 파일 등장 순서대로 나열한 user index (Encoder 기준)."""
    rank = load_file_rank()
    rank = rank[rank.index.isin(enc.users)]
    return enc.user_index(rank.sort_values().index.to_numpy())
