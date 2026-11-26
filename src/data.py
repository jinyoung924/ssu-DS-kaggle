"""데이터 로드와 raw id <-> index 매핑."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
# DS_ROOT로 data/, outputs/, experiments.csv 위치를 바꿀 수 있음 (smoke test용)
ROOT = Path(os.environ.get("DS_ROOT", REPO))
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
CONFIGS = REPO / "configs"

TIME_COLS = ("timestamp", "time", "unixreviewtime", "date", "datetime")


def load_train() -> pd.DataFrame:
    df = pd.read_csv(RAW / "train.csv")
    missing = {"user_id", "item_id"} - set(df.columns)
    if missing:
        raise KeyError(f"train.csv에 {missing} column이 없음. 실제 column: {list(df.columns)}")
    return df


def load_test_users() -> np.ndarray:
    return pd.read_csv(RAW / "test.csv")["user_id"].to_numpy()


def find_time_col(df: pd.DataFrame):
    for c in df.columns:
        if c.lower() in TIME_COLS:
            return c
    return None


class Encoder:
    """raw user_id/item_id를 0..n-1 index로 매핑.

    user: train + test user 전체, item: train에 등장한 item 전체 (= 추천 후보).
    """

    def __init__(self, train_df: pd.DataFrame, test_users: np.ndarray):
        all_users = np.concatenate([train_df["user_id"].to_numpy(), np.asarray(test_users)])
        self.users = pd.Index(pd.unique(all_users))
        self.items = pd.Index(pd.unique(train_df["item_id"].to_numpy()))

    @property
    def n_users(self):
        return len(self.users)

    @property
    def n_items(self):
        return len(self.items)

    def user_index(self, ids) -> np.ndarray:
        idx = self.users.get_indexer(ids)
        if (idx < 0).any():
            raise KeyError("Encoder에 없는 user_id가 있음")
        return idx

    def item_index(self, ids) -> np.ndarray:
        idx = self.items.get_indexer(ids)
        if (idx < 0).any():
            raise KeyError("Encoder에 없는 item_id가 있음")
        return idx

    def matrix(self, df: pd.DataFrame) -> sp.csr_matrix:
        """binary user x item interaction matrix (중복 interaction은 1로)."""
        u = self.user_index(df["user_id"])
        i = self.item_index(df["item_id"])
        X = sp.csr_matrix(
            (np.ones(len(u), dtype=np.float32), (u, i)),
            shape=(self.n_users, self.n_items),
        )
        X.data[:] = 1.0
        return X
