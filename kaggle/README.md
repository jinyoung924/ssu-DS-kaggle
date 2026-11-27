# kaggle/ — GPU가 필요한 작업

로컬(CPU)에서 오래 걸리는 두 작업을 Kaggle Notebook의 무료 GPU로 돌리고, 결과 파일만 받아와 로컬 blend에 합친다.
두 스크립트 모두 **파일 하나로 동작**한다 (repo의 `src/` 불필요).

| 스크립트 | 하는 일 | 결과 파일 (`/kaggle/working`) | 예상 시간 |
|---|---|---|---|
| `lightgcn_gpu.py` | LightGCN config 3개를 로컬과 같은 split으로 학습하고, valid가 가장 좋은 config로 train 전체 재학습 | `lightgcn_{valid,full}_{user,item}.npy`, `lightgcn_log.json` | 30~60분 (추정) |
| `text_embed_gpu.py` | 상품 텍스트(title, store, 카테고리, features) → 문장 임베딩 (`BAAI/bge-small-en-v1.5`) | `text_emb.npy` | 5분 안팎 (추정) |

## 실행 순서

1. 대회 페이지 → **Code → New Notebook**. 대회 데이터는 `/kaggle/input/ssu-data-science-amazon-top-k-recommendation/`에 자동으로 붙는다.
2. 오른쪽 **Settings**
   - Accelerator: **GPU** (T4 또는 P100)
   - `text_embed_gpu.py`는 모델을 내려받으므로 **Internet: On**
3. 스크립트 내용을 셀에 붙여넣고 실행한다. 두 스크립트를 한 노트북에서 셀 두 개로 이어서 돌려도 된다.
4. 끝나면 오른쪽 **Output**(`/kaggle/working`)에서 결과 파일을 내려받아 로컬 **`data/external/`**에 넣는다 (git에는 올라가지 않음).

## 로컬에서 합치기

```bash
# 1. GPU component 2개를 기존 최고 blend에 더해 가중치 탐색 -> configs/blend_gpu.yaml
uv run python -m scripts.tune_blend configs/tune/gpu_components.yaml

# 2. 기록용 validation run + 제출 파일
uv run python -m src.train --config configs/blend_gpu.yaml --name gpu
uv run python -m src.predict --config configs/blend_gpu.yaml --name gpu
```

- `precomputed` component는 단계에 맞는 파일을 고른다. 검증할 때(`src.train`, `tune_blend`)는 `*_valid_*`, 제출할 때(`src.predict`)는 `*_full_*`.
- `textsim` component(`source: emb`)는 `text_emb.npy` 하나를 두 단계에 같이 쓴다. 텍스트는 split과 무관하기 때문이다.

## 주의

- **split 일치**: valid 임베딩은 로컬 split(랜덤 10%, seed 42)의 train으로 학습해야 로컬 튜닝 결과를 믿을 수 있다. 스크립트가 valid 개수와 checksum을 로컬 값과 비교하고, 다르면 `WARNING`을 출력한다. `src/common.py`의 split 설정을 바꾸면 스크립트의 `SPLIT`과 `EXPECTED`도 함께 고친다.
- LightGCN config를 바꾸려면 `lightgcn_gpu.py` 상단의 `GRID`를 고친다. GPU 할당량은 주당 약 30시간이다.
- 로컬 동작 확인 (CPU, 몇 분):
  ```bash
  QUICK=1 DATA_DIR=data/raw OUT_DIR=/tmp/kaggle_out uv run python kaggle/lightgcn_gpu.py
  QUICK=1 DATA_DIR=data/raw OUT_DIR=/tmp/kaggle_out uv run --with sentence-transformers python kaggle/text_embed_gpu.py
  ```
