"""로컬 validation split.

test 조건(정답 item은 그 user의 train에 없음)을 따라 user별로 interaction 일부를 valid로 뗀다.
- time column이 있으면 user별 마지막 interaction들을, 없으면 랜덤으로 뗀다.
- interaction이 min_items 미만인 user는 전부 train에 남긴다.
"""
import numpy as np
import pandas as pd

from . import data


def get_split(df: pd.DataFrame, mode="auto", ratio=0.2, min_items=3, seed=42, cache=True):
    time_col = data.find_time_col(df) if mode in ("auto", "time") else None
    if mode == "time" and time_col is None:
        raise ValueError("mode=time인데 time column이 없음")
    used = "time" if time_col else "random"

    path = data.PROCESSED / f"split_{used}_r{ratio}_m{min_items}_s{seed}_n{len(df)}.pkl"
    if cache and path.exists():
        d = pd.read_pickle(path)
        return d["train"], d["valid"], used

    cols = list(df.columns)
    d = df.copy()
    if time_col:
        key = d[time_col]
        if not pd.api.types.is_numeric_dtype(key):
            key = pd.to_datetime(key).astype("int64")
        d["_key"] = key
        d = d.sort_values(["user_id", "_key"], kind="mergesort")
        d = d.drop_duplicates(["user_id", "item_id"], keep="first")
    else:
        d = d.drop_duplicates(["user_id", "item_id"])
        d["_key"] = np.random.default_rng(seed).random(len(d))
        d = d.sort_values(["user_id", "_key"], kind="mergesort")

    rank = d.groupby("user_id").cumcount().to_numpy()
    n = d.groupby("user_id")["item_id"].transform("size").to_numpy()
    k = np.where(n >= min_items, np.maximum(1, np.floor(n * ratio)).astype(int), 0)
    is_valid = rank >= n - k

    train, valid = d.loc[~is_valid, cols], d.loc[is_valid, cols]
    if cache:
        data.PROCESSED.mkdir(parents=True, exist_ok=True)
        pd.to_pickle({"train": train, "valid": valid}, path)
    return train, valid, used
