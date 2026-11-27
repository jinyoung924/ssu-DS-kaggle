"""Kaggle GPU용 상품 텍스트 임베딩 (11-27_exp.md Step 8). 이 파일 하나로 동작한다 (repo 코드 불필요).

Kaggle Notebook: 대회 데이터 추가 -> Accelerator: GPU -> Settings에서 Internet On (모델 다운로드)
-> 이 파일 내용을 셀에 붙여넣고 실행.
결과 (/kaggle/working):
  text_emb.npy  (n_items, d) float16, item_id 순서, L2 정규화
로컬: data/external/text_emb.npy에 넣고 textsim component(source: emb)로 blend (kaggle/README.md 참고).

로컬 동작 확인: QUICK=1 DATA_DIR=data/raw OUT_DIR=<폴더> python kaggle/text_embed_gpu.py
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np

MODEL = os.environ.get("EMB_MODEL", "BAAI/bge-small-en-v1.5")  # 384차원, 빠름. 더 크게: BAAI/bge-base-en-v1.5
QUICK = os.environ.get("QUICK") == "1"                        # 로컬 동작 확인용: 앞쪽 64개 아이템만
OUT = os.environ.get("OUT_DIR", "/kaggle/working")


def find_data():
    cands = [os.environ.get("DATA_DIR"), "/kaggle/input/ssu-data-science-amazon-top-k-recommendation",
             *glob.glob("/kaggle/input/*"), *glob.glob("/kaggle/input/*/*")]
    for d in cands:
        if d and os.path.exists(os.path.join(d, "item_meta.jsonl")):
            return d
    raise FileNotFoundError("item_meta.jsonl을 찾을 수 없음: 노트북에 대회 데이터를 추가했는지 확인")


def item_text(m):
    """title | store | 카테고리 경로 | features 앞 5개 (상품의 종류/스타일/브랜드가 드러나는 필드)."""
    cats = " > ".join(str(c) for c in (m.get("categories") or []))
    feats = " ".join(str(x) for x in (m.get("features") or [])[:5])
    return " | ".join(x for x in [m.get("title") or "", m.get("store") or "", cats, feats] if x)[:1500]


def main():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"], check=True)
        from sentence_transformers import SentenceTransformer
    import torch

    data_dir = find_data()
    os.makedirs(OUT, exist_ok=True)
    texts = {}
    with open(os.path.join(data_dir, "item_meta.jsonl"), encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            texts[int(m["item_id"])] = item_text(m)
    n = max(texts) + 1
    docs = [texts.get(i, "") for i in range(n)]
    if QUICK:
        docs = docs[:64]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"model={MODEL} device={dev} items={len(docs)}", flush=True)
    model = SentenceTransformer(MODEL, device=dev)
    model.max_seq_length = 256
    E = model.encode(docs, batch_size=128, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
    name = "text_emb_quick.npy" if QUICK else "text_emb.npy"
    np.save(os.path.join(OUT, name), E.astype(np.float16))
    print(f"saved {os.path.join(OUT, name)} shape={E.shape}", flush=True)


if __name__ == "__main__":
    main()
