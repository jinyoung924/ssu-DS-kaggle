"""대회 Recall@K: 전체 user의 hit 합 / sum(min(|GT_u|, K)) (micro)."""


def recall_at_k(recs, gt, k=20) -> float:
    """recs: user별 추천 item index 배열, gt: 같은 순서의 정답 item index set."""
    hits = denom = 0
    for r, g in zip(recs, gt):
        if not g:
            continue
        hits += len(g.intersection(r[:k].tolist()))
        denom += min(len(g), k)
    return hits / denom if denom else 0.0
