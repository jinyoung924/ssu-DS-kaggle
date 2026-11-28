# [SSU-DataScience] Amazon top-k recommendation

Kaggle: https://www.kaggle.com/competitions/ssu-data-science-amazon-top-k-recommendation
Task: test user마다 top-20 item 추천 / Metric: Recall@20 / 하루 제출 5회

## Setup

```bash
uv sync                    # 기본 (numpy, pandas, scipy, pyyaml)
uv sync --extra torch      # LightGCN
uv sync --extra notebook   # EDA notebook
```

데이터: 대회 Data 탭의 파일 5개를 `data/raw/`에 넣는다.

```
data/raw/train.csv  test.csv  sample_submission.csv  item_meta.jsonl  train_reviews.jsonl
```

Kaggle CLI를 쓰는 경우 (`~/.kaggle/kaggle.json` 설정 후):

```bash
uv run kaggle competitions download -c ssu-data-science-amazon-top-k-recommendation -p data/raw
```

## Workflow

```bash
# 1. 로컬 validation 실험 -> experiments.csv에 valid Recall@20 기록
uv run python -m src.train --config configs/ease.yaml --name reg500

# 1-1. blend 가중치 탐색 (component는 한 번만 학습) -> configs/blend_*.yaml
uv run python -m scripts.tune_blend configs/tune/final_artifact.yaml

# 2. 괜찮은 run을 train 전체로 재학습 -> outputs/submissions/<run_id>.csv
uv run python -m src.predict --from-run <run_id>

# 3. 제출 (웹 업로드 또는 CLI)
uv run kaggle competitions submit -c ssu-data-science-amazon-top-k-recommendation \
    -f outputs/submissions/<run_id>.csv -m "<run_id>"

# 4. public LB 점수 기록 -> local score와 LB가 같은 방향으로 움직이는지 확인
uv run python -m src.experiments lb <submit_run_id> 0.0812
uv run python -m src.experiments show
```

## Local validation

`src/split.py`: user별로 interaction 일부(`ratio`, 기본 10%)를 valid로 뗀다.
test 정답이 그 user의 train에 없다는 조건을 그대로 따른다.

- 기본은 랜덤 분리 (`mode: random`, 대회 가이드 권장). `mode: time`이면 user별 마지막 interaction, `auto`면 time column이 있을 때 time
- interaction이 `min_items` 미만인 user는 전부 train에 남김
- 기본적으로 test.csv에 있는 user만 평가 (`eval.test_users_only`)

split 설정은 config의 `split:` 섹션으로 바꿀 수 있다 (기본값은 `src/common.py`의 `DEFAULTS`).
split을 바꾸면 이전 run과 점수를 직접 비교할 수 없으니 note에 남길 것.

## Models

| config | 설명 |
|---|---|
| `popular.yaml` | 인기순 (하한선) |
| `itemknn.yaml` | item-item cosine CF |
| `ease.yaml` | EASE, item 수가 많으면 `max_items`로 후보 제한 |
| `lightgcn.yaml` | LightGCN + BPR, `eval_every`마다 valid 측정 후 early stopping |
| `blend_best.yaml` | 11-27 blend (Ordering Artifact 사용). `scripts/tune_blend.py`가 생성 |
| `blend_noartifact.yaml` | Artifact 없는 최고 blend (보고서 비교용) |
| `blend_cease_artifact.yaml` | CEASE-A 중심 blend (valid 0.0727) |
| `rerank_v3.yaml` | **현재 최고**: CEASE-A blend 후보 top-100 → LightGBM 재정렬 (valid 0.0783). `scripts/rerank2.py`로 실행 |

재정렬은 split 여러 개로 학습하고, 유저를 묶음으로 나눠 평가·제출 유저를 학습에서 빼야 한다 (같은 유저로 학습하면 정답 쌍을 외워 점수가 부풀려짐). 실행 순서와 근거는 `개선안_01.md` 5-2장.

blend에 쓰는 component (`src/models/`, 실험 근거는 `11-27_exp.md`):

| model | 설명 |
|---|---|
| `easetime` | EASE의 Gram을 시간 감쇠로 가중 (`tau_days`). `reg_pop`(EDLAE 정규화), `user_pow` 옵션 |
| `cease` | 시간 감쇠 EASE의 Gram에 텍스트 유사도(`alpha`), 같은 브랜드(`beta`), 파일 위치 구간 동시등장(`pos_gamma`, `pos_bin`)을 더함. CEASE-A = `pos_gamma` 0.05, `pos_bin` 12 (단독 0.0698) |
| `fileorder` | Ordering Artifact: `train_reviews.jsonl`에서 가까이 등장한 유저들의 구매 빈도 |
| `textsim` | 상품 텍스트 유사도. `source: tfidf`(로컬) 또는 `emb`(kaggle/에서 만든 임베딩) |
| `affinity` | 유저의 브랜드/카테고리 분포 x 인기도 |
| `itemknn`, `rp3beta`, `poscooc`, `alive` | item 유사도 / 파일 위치 유사도 / 활동 기간 필터 (`poscooc`, `alive`, `rp3beta`는 효과 없음) |
| `precomputed` | kaggle/lightgcn_gpu.py가 GPU로 학습한 임베딩 |

새 모델 추가: `src/models/base.py`의 `Recommender`를 상속해 `_fit`, `score`를 구현하고 `src/models/__init__.py`의 `build_model`에 등록.

## Structure

```
configs/          실험별 hyperparameter (tune/: blend 가중치 탐색 spec)
data/raw/         대회 원본 (git 제외)
data/processed/   split, 메타데이터, 파일 순서 캐시
data/external/    kaggle/에서 받은 GPU 결과 (git 제외)
kaggle/           Kaggle GPU용 스크립트 (LightGCN, 텍스트 임베딩)
scripts/          tune_blend.py (blend 가중치 탐색), rerank2.py (재정렬, data/processed/rerank/에 split별 feature 캐시),
                  rerank_ens.py (재정렬 앙상블)
notebooks/        EDA
src/
  data.py         로드, id <-> index 매핑, interaction matrix
  features.py     메타데이터, 리뷰 파일의 유저 등장 순서
  split.py        로컬 validation split
  metrics.py      Recall@K (대회 정의)
  models/         Popular, ItemKNN, EASE, LightGCN, blend component들
  train.py        validation 실험
  predict.py      전체 재학습 + submission 생성
  validate_sub.py submission 포맷 검사
  experiments.py  experiments.csv 기록/조회
outputs/runs/         run별 config.yaml, metrics.json
outputs/submissions/  제출 csv (git 제외)
experiments.csv       전체 실험 기록 (local score, public LB)
```
