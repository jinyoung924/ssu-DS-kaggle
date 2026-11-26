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

`src/split.py`: user별로 interaction 일부(`ratio`, 기본 20%)를 valid로 뗀다.
test 정답이 그 user의 train에 없다는 조건을 그대로 따른다.

- time column이 있으면 user별 마지막 interaction, 없으면 랜덤
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

새 모델 추가: `src/models/base.py`의 `Recommender`를 상속해 `_fit`, `score`를 구현하고 `src/models/__init__.py`의 `build_model`에 등록.

## Structure

```
configs/          실험별 hyperparameter
data/raw/         대회 원본 (git 제외)
data/processed/   split 캐시
notebooks/        EDA
src/
  data.py         로드, id <-> index 매핑, interaction matrix
  split.py        로컬 validation split
  metrics.py      Recall@K (대회 정의)
  models/         Popular, ItemKNN, EASE, LightGCN
  train.py        validation 실험
  predict.py      전체 재학습 + submission 생성
  validate_sub.py submission 포맷 검사
  experiments.py  experiments.csv 기록/조회
outputs/runs/         run별 config.yaml, metrics.json
outputs/submissions/  제출 csv (git 제외)
experiments.csv       전체 실험 기록 (local score, public LB)
```
