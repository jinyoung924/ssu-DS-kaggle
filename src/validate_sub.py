"""submission 포맷 검사 (대회 Submission 규칙 기준).

  python -m src.validate_sub outputs/submissions/<file>.csv
"""
import argparse
import sys

import pandas as pd

from . import data


def check(path, k=20) -> list:
    errors = []
    sub = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(sub.columns) != ["user_id", "item_id"]:
        return [f"column이 user_id,item_id가 아님: {list(sub.columns)}"]

    test_users = set(pd.read_csv(data.RAW / "test.csv", dtype=str)["user_id"])
    users = set(sub["user_id"])
    if test_users - users:
        errors.append(f"test user {len(test_users - users)}명 누락")
    if users - test_users:
        errors.append(f"test.csv에 없는 user {len(users - test_users)}명 포함")
    if sub["user_id"].duplicated().any():
        errors.append(f"중복 user_id {sub['user_id'].duplicated().sum()}행")

    lists = sub["item_id"].str.split(" ")
    bad_len = (lists.str.len() != k).sum()
    if bad_len:
        errors.append(f"item이 정확히 {k}개가 아닌 행 {bad_len}개")
    dup = lists.apply(lambda x: len(set(x)) != len(x)).sum()
    if dup:
        errors.append(f"item이 중복된 행 {dup}개")

    train_items = set(pd.read_csv(data.RAW / "train.csv", usecols=["item_id"], dtype=str)["item_id"])
    unknown = len({i for row in lists for i in row} - train_items)
    if unknown:
        errors.append(f"train에 없는 item_id {unknown}종 (오타/변환 오류 확인)")
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()
    errors = check(args.path, args.k)
    if errors:
        print("format ERROR:\n  " + "\n  ".join(errors))
        sys.exit(1)
    print("format OK")


if __name__ == "__main__":
    main()
