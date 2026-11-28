# 제출 파일 기록

파일 이름은 `MMDD_NN_전략.csv`이다. 만든 날짜, 전체 제출 순서(01부터 이어지는 번호, 날짜가 바뀌어도 이어짐), 사용한 전략(영어)을 뜻한다. 이름순으로 정렬하면 만든 순서가 된다.

- csv 파일은 용량 때문에 git에 올리지 않는다(`.gitignore`). 이 표만 공유한다. 파일이 필요하면 만든 사람에게 받거나 아래 run id로 다시 만든다.
- 로컬 점수와 LB의 원본 기록은 루트의 `experiments.csv`에 있다. run id로 찾는다.

| # | 파일 | 전략 | 로컬 Recall@20 (seed 42) | LB | run id |
|---|---|---|---|---|---|
| 01 | `1127_01_ease_baseline.csv` | EASE (reg 500, 인기 상위 1만 아이템만 후보) | 0.04829 | 0.04943 | `1127-160145_ease` |
| 02 | `1127_02_blend_ordering_artifact.csv` | 시간 감쇠 EASE, ItemKNN, 텍스트 유사도, 브랜드·카테고리 선호를 섞은 blend + Ordering Artifact(리뷰 파일에서 가까이 등장한 유저들의 구매) | 0.07052 | – | `1127-174918_blend_best` |
| 03 | `1127_03_blend_no_artifact.csv` | 02에서 Artifact를 뺀 blend (보고서 비교용) | 0.06287 | – | `1127-175452_blend_noartifact` |
| 04 | `1127_04_blend_artifact_lightgcn.csv` | 02 + LightGCN (Kaggle GPU에서 학습) | 0.07148 | – | `1127-230139_blend_blend_lightgcn` |
| 05 | `1128_05_rerank_v2_binary_neg30pct.csv` | 재정렬 v2: CEASE blend 후보 top-100을 LightGBM으로 재정렬. binary, 오답 행 30%만 샘플링, 학습 split 8개, 유저 4묶음 | 0.07727 | – | `1128-023920_rerank2_N100` |
| 06 | `1128_06_rerank_v2_binary_allneg.csv` | 05와 같고 오답 행을 모두 사용 | 0.07722 | – | `1128-031829_rerank2_N100` |
| 07 | `1128_07_rerank_v2_lambdarank.csv` | 06의 학습 목표를 lambdarank로 | 0.07757 | – | `1128-034138_rerank2_N100` |
| 08 | `1128_08_rerank_v2_ensemble_binary_lambdarank.csv` | 06과 07의 점수 평균 | 0.07799 | – | `1128-034448_rerank_ens_2runs` |
| 09 | `1128_09_blend_cease_artifact.csv` | CEASE-A blend: Artifact를 CEASE의 Gram에 결합. 재정렬 없음 | 0.07273 | – | `1128-045008_blend_blend_cease_artifact` |
| 10 | `1128_10_rerank_v3_binary.csv` | 재정렬 v3: 09의 후보 top-100을 LightGBM으로 재정렬. binary, 오답 행 모두 | 0.07804 | – | `1128-050944_rerank2_N100` |
| 11 | `1128_11_rerank_v3_lambdarank.csv` | 10의 학습 목표를 lambdarank로 | 0.07829 (3개 평균 0.07757) | **0.07946** | `1128-053233_rerank2_N100` |
| 12 | `1128_12_rerank_v3_ensemble_binary_lambdarank.csv` | 10과 11의 점수 평균 | 0.07812 | – | `1128-053445_rerank_ens_2runs` |
| 13 | `1128_13_rerank_v3_lambdarank_16splits_3bags.csv` | **현재 최고.** 11 + 학습 split 16개 + 유저 묶음 나누기 3번 평균 | 0.07872 (3개 평균 0.07822) | 예상 약 0.080 | `1128-120025_rerank2_N100` |

- LB가 "–"인 파일은 `experiments.csv`에 LB 기록이 없다 (제출하지 않았거나, 제출했지만 기록하지 않음).
- 로컬 점수는 대부분 seed 42 split 하나에서 잰 값이다. 이 경우 0.001 안팎의 차이는 잡음이라 단순 비교하면 안 된다 (`docs/개선 계획_02.md` 1-2). 11과 13은 평가 split 3개의 평균도 함께 적었다.
- 08과 12의 앙상블은 서로 다른 모델을 섞는 방식이라 더 이상 쓰지 않는다 (`docs/개선 계획_02.md` 2-1).
- 재정렬 v2·v3의 설정은 `configs/rerank_v2.yaml`, `configs/rerank_v3.yaml`, 13의 재현 명령은 `docs/개선 계획_02.md` 4-2에 있다.

## 새 제출을 추가할 때

스크립트는 제출 파일을 `outputs/submissions/<run_id>.csv`로 만든다. 제출하기 전에 다음 번호로 이름을 바꾸고 이 표에 한 줄 추가한다.

```bash
mv outputs/submissions/<run_id>.csv outputs/submissions/MMDD_14_<strategy>.csv
# LB가 나오면 experiments.csv에도 기록 (run id 기준)
uv run python -m src.experiments lb <run_id> <public score>
```
