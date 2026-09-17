# Auto Research용 오프라인 실험 경로 설계

- Status: 승인됨 (#790, 구현 진행)
- 요청 맥락: CTR 학습 코드를 Auto Research용 오프라인 실험 구조로 리팩토링
- 이슈: #790 · 브랜치: `feat/790-offline-experiment-path`
- 관련: `docs/specs/2026-07-31-experiment-feature-override.md`(#405, prod 계약 불변 원칙의 근거)
  · `docs/specs/2026-08-03-paired-offline-experiment-comparison.md`(#454, production 승격 게이트)
  · `docs/specs/2026-08-06-experiment-result-reporting-contract.md`(#550, Auto Research 자율 실행 파이프라인)
  · `docs/specs/2026-08-13-repository-structure-redesign.md`(#754, 이 설계의 모든 경로가 따르는
  `autoresearch/<파이프라인 단계>/` 구조 — 최초 초안은 구 `src/` 구조로 작성되었다가 #754 반영 후
  전체 경로를 재매핑했다)
  · `docs/specs/2026-08-14-paper-grounded-autonomous-ml-research-harness.md`(#769, 완전 자율
  다중 trial 연구 루프 — 별도의 훨씬 큰 이니셔티브. 이 문서의 `run-experiment`는 사람이 직접
  1회 실행하는 수동 CLI이고 #769의 `harness-predict`는 Sealed Judge 전용 자동 진입점이라
  계약·목적이 겹치지 않는다)

## 배경 — 기존 구조 분석

Feature 변경 → Training Dataset 생성 → Model 학습 → Evaluation → Baseline 비교를
로컬에서 반복 가능하게 만들고 싶다는 요청을 코드 기준으로 분석한 결과, 요청 기능의
상당 부분이 **이미 구현되어 있다.** 다만 production 승격 게이트에 맞춰진 무거운
형태라 로컬 1회성 실험에는 맞지 않고, 핵심 기능 하나(feature 부분집합 선택)는
설계상 의도적으로 막혀 있다.

### 이미 존재하는 것

| 항목 | 위치 | 상태 |
|---|---|---|
| CLI 진입점 | `autoresearch/cli.py` (Typer) | `build-features`/`train-model`/`evaluate-model`/`run-pipeline`/`sweep-seeds`/`compare-paired-experiment`/`measure-degradation`/`promote-model`/`report-experiment-result` 모두 구현됨 |
| Feature override | `autoresearch/feature_engineering/model_contract.py` | `--extra-features`로 prod 21컬럼 **뒤에 추가만** 가능(`resolve_experiment_feature_columns`) |
| 모델 인터페이스 | `autoresearch/model_training/base.py` | `CTRModel` ABC(`fit`/`predict_proba`/`save`/`load`), `LGBMModel`이 유일한 구현체 |
| Evaluation | `autoresearch/model_evaluation/evaluate.py` | `roc_auc`/`pr_auc`/`log_loss` (`evaluate_held_out_metrics`) |
| MLflow 선택적 사용 | `train.py --experiment` | 트래킹 URI 미설정 시 로컬 파일 스토어(`file:./mlruns`) 자동 폴백(#406) — 필수 아님 |
| baseline 비교 | `autoresearch/model_evaluation/paired_experiment.py` | `compare-paired-experiment` — MLflow run ID·GCS 이미지 digest·code archive SHA·GCS promotion evidence store를 요구하는 **production 승격 게이트** |
| 실험 결과 반영 | `autoresearch/reporting/experiment_result_report.py` | `report-experiment-result`(#550, 현재 작업 중) — GitHub 이슈 → K8s Job → Agent Orchestration DB로 이어지는 Auto Research 자율 실행 파이프라인 전용 |

### 실제 gap

1. **Feature 부분집합 선택(ablation) 불가.** `train.py:781-783`은
   `extra_features`가 있으면 `MODEL_FEATURE_COLUMNS + extra`, 없으면
   `MODEL_FEATURE_COLUMNS` 그대로 둘 중 하나만 가능하다. 임의 부분집합(예: 21개
   중 1개 제거)은 지원하지 않는다. 이는 버그가 아니라 `docs/specs/2026-07-31-experiment-feature-override.md`가
   명시한 의도적 설계다 — ONNX 입력이 이름 없는 순서 배열이라 prod 접두부가
   흔들리면 서빙이 조용히 오예측하고, prod 계약이 불변이어야 서빙·시뮬레이션·
   일일추천이 안 깨진다(#396 실측: 피처 1개 추가만으로 테스트 43건 추가 실패).
2. **baseline 비교 도구가 로컬 1인 실험에는 과함.** `compare-paired-experiment`는
   이미 완료된 두 MLflow run을 GCS provenance로 재검증하는 production 게이트라,
   로컬에서 빠르게 반복하려는 목적과 다른 층위다.
3. **통합 ExperimentConfig 없음.** 기능은 있지만 CLI 플래그 여러 개
   (`--extra-features`/`--feature-service`/`--dataset-uri`/`--experiment-plan-receipt`/
   `--promotion-evidence-root`)로 흩어져 있다.
4. **Feast 없는 로컬 dataset 신규 조립 불가.** `build_training_dataset.py`는
   `_assemble_via_feast`가 유일 경로다(#359 C2에서 DuckDB 재계산·mock CSV 경로
   제거). 단, 로컬에 이미 `data/processed/training_dataset.csv`(437,371행, prod
   21컬럼 + `clicked`)가 존재해 **신규 조립 없이도** 실험을 시작할 수 있다.

## 사용자 결정 사항 (확인 완료)

- Feature ablation은 **완전히 별도의 로컬 실험 전용 경로**로 신설한다. 기존
  `--extra-features`(추가 전용) 확장이 아니라, prod 계약·ONNX·서빙에 영향 없는
  새 모듈로 분리한다.
- baseline 비교는 **로컬 JSON 2개(baseline/candidate) 단순 diff** 수준으로
  만든다. `compare-paired-experiment`(GCS/MLflow production 게이트)는 재사용하지
  않는다.
- Training Dataset은 **이미 있는 CSV를 입력으로만 받는다.** Feast 없는 신규 로컬
  조립 경로는 만들지 않는다(범위 최소화, Feast 구조 무변경 원칙에 가장 부합).
- 모델은 **`LGBMModel`만 재사용**한다. XGBoost 등 새 구현체는 이번 스코프에 넣지
  않는다(인터페이스는 이미 확장 가능하게 되어 있어 필요해지면 나중에 같은
  패턴으로 추가).

## 기존 학습 경로의 split/preprocessing semantics (보존 대상)

사용자 요청으로 구현 전에 `train.py`(`_train_from_resolved_dataset`, Step 2~7)와
`evaluate.py`를 정독해 학습 결과에 영향을 주는 로직을 전부 목록화했다.
`offline_experiment.py`는 아래 순서와 의미를 **그대로** 따르되, MLflow 로깅·
ONNX 변환·promotion evidence·snapshot 재사용·서빙용 아티팩트 저장은 생략한다.

| # | 단계 | 위치(현재 코드) | 의미 | offline_experiment 처리 |
|---|---|---|---|---|
| 1 | Train/Val/Test 분할 | `train.py:736-753` | `sklearn.train_test_split` 2단계: 전체→(train+val)/test 먼저(`test_size`, `stratify=clicked`), 그다음 (train+val)→train/val(`val_size/(1-test_size)`, 같은 `stratify`). **두 단계 모두 같은 `random_state`(=split_seed)** 사용 | 동일 로직 신규 작성(재사용 불가 — 함수가 분리되어 있지 않음). `test_size`/`val_size` 기본값은 prod `config.yaml`과 동일(0.2/0.2) |
| 2 | 분할별 라벨 검증 | `train.py:759-764`, `require_binary_labels`(`train.py:168`) | train/val/test 각각 양성·음성이 모두 있는지 확인, 없으면 즉시 실패 | **재사용**(`from autoresearch.model_training.train import require_binary_labels`) |
| 3 | Feature/Label 분리 | `train.py:813-816` | `X = df[feature_columns]`, `y = df["clicked"]` | feature_columns = ExperimentConfig의 `features` 그대로(순서 포함) |
| 4 | Negative downsampling | `train.py:821-844`, `downsample_negatives`(`autoresearch/model_training/downsampling.py:68`) | **train split에만** 적용. `sampling_rate<1.0`일 때만 동작(기본 1.0=no-op). val/test는 원분포 유지 | **재사용**(`from autoresearch.model_training.downsampling import downsample_negatives, apply_downsampling_calibration`). `sampling_rate`는 ExperimentConfig `model.params.sampling_rate`, 기본값은 prod `config.yaml`과 동일(1.0) |
| 5 | Categorical dtype 캐스팅 | `train.py:846-850`, `collect_categorical_categories`(`train.py:331`) | `CATEGORICAL_FEATURE_COLUMNS` 중 **실제 feature_columns에 포함된 것만** train+val union 카테고리로 `pd.Categorical` 캐스팅 | **재사용**(`from autoresearch.model_training.train import collect_categorical_categories`), 대상 컬럼은 `[c for c in CATEGORICAL_FEATURE_COLUMNS if c in features]`로 계산(ablation으로 categorical feature가 빠지면 자동으로 제외됨) |
| 6 | scale_pos_weight 결정 | `train.py:852-885`, `compute_auto_scale_pos_weight`(`train.py:201`) | downsampling 활성(`sampling_rate<1.0`)이면 강제로 1(명시적 다른 숫자값과 동시 지정 시 에러). 아니면 `"auto"`→`neg/pos` 비율 계산, 아니면 고정값 | **재사용**(`compute_auto_scale_pos_weight`), 강제/에러 분기 로직은 짧아서 신규 작성 |
| 7 | 모델 학습 | `train.py:920-928` | `LGBMModel(scale_pos_weight, n_estimators, learning_rate, num_leaves, random_state=model_seed).fit(X_train, y_train, categorical_features=categorical_columns)` | **재사용**(`LGBMModel`), 하이퍼파라미터 기본값은 prod `config.yaml`의 `model:` 섹션과 동일(n_estimators=200/learning_rate=0.05/num_leaves=31), `ExperimentConfig.model.params`로 override 가능 |
| 8 | Held-out(Test) 평가 | `evaluate.py:132-190`, `evaluate_held_out_metrics` | roc_auc/pr_auc는 원본 확률(downsampling에 불변), log_loss는 `apply_downsampling_calibration`으로 원분포 보정한 확률로 계산 | **부분 재사용 불가** — `evaluate_held_out_metrics`가 내부적으로 prod 고정 `CATEGORICAL_FEATURE_COLUMNS` 전체를 캐스팅하려 시도해(`_held_out_positive_proba`), categorical feature 하나가 ablation으로 빠지면 `KeyError`. 대신 같은 3줄 로직(raw_proba → `apply_downsampling_calibration`(재사용) → roc_auc/pr_auc/log_loss)을 `offline_experiment.py`에서 실제 `categorical_columns`(5번 항목과 동일 계산)만 캐스팅하도록 재작성 |
| 9 | seed 분리 | `train.py:649-660` | `split_seed`/`model_seed`/`sampler_seed` 3개 분리 가능하나, 셋 다 미지정 시 `random_state` 하나로 통일(legacy path) | `ExperimentConfig.experiment.seed` 하나만 받아 3개 모두에 동일 적용(legacy path와 동일 — 이 도구 스코프에서 3개를 분리할 이유가 없음) |

**모델 학습 결과에 영향을 주는 요소 중 offline_experiment가 적용하는 것**: train/val/test
분할 비율·seed·stratify, negative downsampling 시점·비율, scale_pos_weight 계산 방식,
categorical dtype 캐스팅, LightGBM 하이퍼파라미터. **적용하지 않는 것**(prod
orchestration 전용이라 이번 스코프 밖): MLflow run 로깅, ONNX 변환, promotion
evidence, snapshot manifest, model registry 등록.

NaN 처리는 별도 로직이 없다 — `training_dataset.csv`는 이미 cold-start 기본값
(`historical_category_affinity`의 `'unknown'` 등, `model_contract.py`의
`COLD_START_CATEGORICAL_DEFAULT`)이 채워진 채로 조립되어 있고, 학습 코드는 결측을
추가로 처리하지 않는다. offline_experiment도 동일하게 데이터셋을 있는 그대로 쓴다.

## 설계

### 아키텍처

```
examples/offline_experiments/*.yaml (ExperimentConfig)
        ↓
autoresearch/cli.py: run-experiment --config ... [--baseline-result ...] --output ...
        ↓
autoresearch/model_evaluation/offline_experiment.py   ← 신설, config/분할/학습/평가/비교 오케스트레이션
        ↓ (model_type 조회는 여기서만)
autoresearch/model_training/model_factory.py               ← 신설, model_type: str → CTRModel 구현체 매핑
        ↓ (재사용, 무수정)
CTRModel / LGBMModel (autoresearch/model_training/)
downsample_negatives / apply_downsampling_calibration (autoresearch/model_training/downsampling.py)
require_binary_labels / compute_auto_scale_pos_weight / collect_categorical_categories / load_config (autoresearch/model_training/train.py)
        ↓
ExperimentResult JSON (baseline diff 포함)
```

신규 모듈은 `autoresearch/model_evaluation/offline_experiment.py`와 `autoresearch/model_training/model_factory.py`
둘이며, `autoresearch/cli.py`에 명령 하나(`run-experiment`)를 추가하는 것 외에는 기존
파일을 수정하지 않는다. `train.py`/`evaluate.py`/`model_contract.py`/Feast/ONNX/
MLflow 승격 경로는 건드리지 않는다.

**모델 레지스트리를 별도 파일로 분리하는 이유**: `model_type: str → CTRModel`
매핑을 `offline_experiment.py` 안에 두면, 나중에 XGBoost 등 새 구현체를
추가할 때 config/학습/평가/비교 로직이 있는 바로 그 파일에 손을 대야 한다.
사용자가 명시한 제약("다른 CTRModel 구현체가 추가되어도 offline experiment의
config/training/evaluation/comparison 로직은 수정하지 않는 구조")을 문자
그대로 지키려면, 조회 지점을 `autoresearch/model_training/model_factory.py` 하나로 분리해 새
구현체 추가가 그 파일 한 줄 추가로 끝나야 한다.

### 재사용 vs 신규 경계

**재사용(모두 기존 함수를 import, 수정 없음)**:

- `autoresearch.model_training.base.CTRModel`, `autoresearch.model_training.lgbm_model.LGBMModel`
- `autoresearch.model_training.downsampling.downsample_negatives`, `apply_downsampling_calibration`
- `autoresearch.model_training.train.require_binary_labels`(분할별 라벨 검증)
- `autoresearch.model_training.train.compute_auto_scale_pos_weight`
- `autoresearch.model_training.train.collect_categorical_categories`(train+val categorical union 캐스팅)
- `autoresearch.model_training.train.load_config`(prod `config.yaml` 기본값 상속)
- `autoresearch.feature_engineering.model_contract.CATEGORICAL_FEATURE_COLUMNS`(categorical 여부 판별용 참조 상수 — 목록 자체를 바꾸지 않음)

**새로 작성**: train/val/test 분할, feature 부분집합 선택, scale_pos_weight
강제/에러 분기, held-out 평가 확률 계산, 모델 레지스트리(`autoresearch/model_training/model_factory.py`
— `MODEL_REGISTRY: dict[str, type[CTRModel]]`, 지금은 `{"lightgbm": LGBMModel}` 하나).

- 분할 로직을 새로 쓰는 이유: `train.py`의 분할은 MLflow 로깅·ONNX 변환·
  promotion evidence·snapshot 재사용과 한 함수(`_train_from_resolved_dataset`,
  200줄+)에 얽혀 있다. 여기서 분할만 뽑아 쓰려면 그 함수를 쪼개는 구조 변경이
  필요한데, 이는 "구조 변경과 동작 변경 분리" 원칙 및 "기존 Production 코드
  대규모 리팩토링 금지" 제약 둘 다에 걸린다. `sklearn.train_test_split` 직접
  호출 몇 줄로 새로 쓰는 편이 더 작은 변경이다. **분할 의미(2단계 stratified
  split, 같은 seed, stratify=clicked)는 그대로 재현한다** — "split 코드를 새로
  작성하더라도 의미는 동일하게 유지"라는 요청사항 1을 충족한다.
- held-out 평가를 `evaluate_held_out_metrics`째로 재사용하지 않는 이유: 그
  함수는 내부적으로 prod 고정 `CATEGORICAL_FEATURE_COLUMNS`(5개 전부)를
  캐스팅하려 시도해, ablation으로 categorical feature 하나가 빠지면
  `KeyError`로 죽는다. 같은 3줄 계산(raw_proba → `apply_downsampling_calibration`
  → roc_auc/pr_auc/log_loss)을 실제 `categorical_columns`만 캐스팅하도록
  다시 쓰되, 계산 로직 자체(어떤 지표를 raw로 재고 어떤 지표를 보정하는지)는
  동일하게 유지한다.
- feature 선택을 `model_contract.py`에 넣지 않는 이유: 그 모듈은 prod 불변
  계약 전용이다. `offline_experiment.py` 안에서 "CSV에 있는 컬럼 중 config가
  지정한 목록만 쓴다"는 단순 pandas 선택으로 처리하며, prod의
  `MODEL_FEATURE_COLUMNS`/ONNX 순서 계약과는 완전히 분리된 별개 경로다 —
  ablation이 무엇을 하든 prod 서빙에 영향이 없다.

### ExperimentConfig (YAML)

```yaml
experiment:
  name: remove_topic_similarity
  seed: 42                 # split_seed/model_seed/sampler_seed 모두에 동일 적용

model:
  type: lightgbm            # 현재는 lightgbm만. autoresearch/model_training/model_factory.py의 MODEL_REGISTRY에 한 줄
                             # 추가하는 것만으로 다른 CTRModel 구현체를 확장할 수 있게 열어둠
  params: {}                # LGBMModel 생성자 인자 override. 미지정 필드는 autoresearch/model_training/config.yaml의
                             # model: 섹션 기본값(n_estimators=200/learning_rate=0.05/num_leaves=31/
                             # scale_pos_weight=auto/sampling_rate=1.0)을 그대로 물려받는다.

data:
  path: data/processed/training_dataset.csv
  test_size: 0.2             # 미지정 시 config.yaml과 동일 기본값(0.2)
  val_size: 0.2               # 미지정 시 config.yaml과 동일 기본값(0.2)

features:                  # MODEL_FEATURE_COLUMNS를 완전히 대체 — 부분집합/재배열 자유
  - age_group
  - occupation
  # ...

evaluation:
  primary_metric: roc_auc  # roc_auc | pr_auc | log_loss
```

pydantic 모델(`ExperimentConfig`)로 파싱하며, `features`가 데이터셋 컬럼에 없거나
라벨 컬럼(`clicked`)을 포함하면 실행 전에 실패한다(fail-closed, 기존 저장소
관례와 일치). `model.params`/`data.test_size`/`data.val_size`의 기본값을
prod `config.yaml`에서 물려받는 것은 "feature 목록 외의 조건은 기존 학습과
동일해야 한다"는 요구를 설계 차원에서 보장하기 위함이다 — 실험자가 아무것도
지정하지 않으면 prod와 같은 조건에서 feature만 달라진다.

### ExperimentResult (JSON)

```json
{
  "experiment_name": "remove_topic_similarity",
  "status": "success",
  "model": "lightgbm",
  "features": ["age_group", "occupation", "..."],
  "metrics": {"roc_auc": 0.7421, "pr_auc": 0.031, "log_loss": 0.041},
  "comparison_conditions": {
    "data_path": "data/processed/training_dataset.csv",
    "row_count": 437371,
    "seed": 42,
    "test_size": 0.2,
    "val_size": 0.2,
    "sampling_rate": 1.0,
    "model_type": "lightgbm",
    "model_params": {
      "n_estimators": 200,
      "learning_rate": 0.05,
      "num_leaves": 31,
      "scale_pos_weight": 34.87
    }
  },
  "baseline": {
    "experiment_name": "baseline_full_features",
    "metrics": {"roc_auc": 0.7609}
  },
  "metrics_delta": {"roc_auc": -0.0188}
}
```

`comparison_conditions`는 metric 자체가 아니라 **이 결과가 다른 결과와 공정하게
비교 가능한 조건 하에서 나왔는지 판별하는 데 쓰는 필드**다(요청사항 4). 강한
provenance 검증(dataset fingerprint 등)은 이번 범위에 넣지 않는다 — 값 자체의
동등 비교로 충분하다.

`metrics`는 항상 `roc_auc`/`pr_auc`/`log_loss` 세 값을 전부 담는다(`evaluation.primary_metric`은
어떤 지표를 대표로 보라는 표시일 뿐 계산 범위를 줄이지 않는다). `metrics_delta`도
세 지표 모두에 대해 `candidate - baseline`을 계산한다.

`comparison_conditions.model_params`는 config에 적힌 원문(예: `scale_pos_weight: auto`)이
아니라 **실제로 학습에 쓰인 최종 값**을 기록한다 — `n_estimators`/`learning_rate`/
`num_leaves`는 config.yaml 기본값+override 병합 결과, `scale_pos_weight`는 6번 단계의
강제/`auto`/고정값 분기를 전부 거쳐 `LGBMModel` 생성자에 실제로 전달된 숫자다. 이번
스코프(feature ablation 비교)에서는 model 자체는 동일해야 하므로 이 값들도
baseline/candidate 간에 반드시 같아야 한다 — 나중에 model을 바꾸는 실험을 지원하게
되면 이 비교 정책(어떤 필드가 "같아야 함"인지)을 별도로 확장한다.

### 실패 처리 (요청사항 3)

`run-experiment`는 예외를 캐치해 `status: "failed"`, `error` 사유를 담은
ExperimentResult JSON을 `--output` 경로에 그대로 남긴 뒤, **반드시 non-zero
exit code로 종료**한다(`typer.Exit(code=1)`). 결과 파일을 남기는 것과 CLI가
실패로 보고하는 것은 별개 행동이며 이번 요청으로 후자를 명시적으로 보장한다 —
Agent가 stdout 문면을 파싱하지 않고 exit code만으로 성공/실패를 프로그램적으로
구분할 수 있어야 한다. 기존 CLI 명령들의 exit code 관례(`compare-paired-experiment`
등, 성공 0 / 실행 실패 1 / 인자·계약 오류 2)와 맞춰 `run-experiment`도
성공 0 / 학습·평가 실행 실패 1 / config·비교조건 검증 실패 2로 구분한다.

### baseline 비교 시 fail-fast (요청사항 4)

`--baseline-result`를 주면, delta를 계산하기 **전에** 두 결과의
`comparison_conditions`(`data_path`/`row_count`/`seed`/`test_size`/`val_size`/
`sampling_rate`/`model_type`/`model_params`)가 전부 일치하는지 검사한다. 하나라도 다르면 조용히
diff하지 않고 어떤 필드가 왜 다른지 명시한 오류로 즉시 중단하며(exit code 2 —
실행 실패가 아니라 비교 조건 자체가 성립하지 않는 것), `metrics_delta` 없는
`status: "failed"` 결과를 남긴다. `features`는 이 비교에서 의도적으로 제외한다
— ablation 실험의 목적 자체가 feature 목록을 다르게 하는 것이기 때문이다.

### CLI

```bash
# A — baseline (전체 21개 feature)
python -m autoresearch.cli run-experiment --config examples/offline_experiments/baseline.yaml \
  --output artifacts/experiments/baseline.json

# B — ablation (topic_similarity 제거), baseline과 자동 diff
python -m autoresearch.cli run-experiment --config examples/offline_experiments/ablation_remove_topic_similarity.yaml \
  --baseline-result artifacts/experiments/baseline.json \
  --output artifacts/experiments/ablation.json
```

`--baseline-result`를 주면 두 실행의 `metrics`를 비교해 `metrics_delta`를 채운다
— 별도 diff 명령을 만들지 않는다.

### 예제 config 위치

`examples/offline_experiments/`에 둔다. 새 top-level 디렉토리가 아니라 기존
`examples/`(이미 `ctr_pipeline_scaffold`가 있는 곳)를 재사용한다.
`experiments/`는 쓰지 않는다 — `CLAUDE.local.md` 규칙상 그 경로는 사용자 개인
벤치마크 로그용으로 이미 쓰이고 있어(로컬 `.git/info/exclude` 대상), 팀 공유
실험 config와 개인 포트폴리오 로그가 섞이는 걸 피한다.

### 검증 계획

설계 승인 후 아래 두 실험을 실제로 실행해 Dataset 소비 → Train → Evaluate →
결과 출력 전 구간이 동작함을 확인한다. 둘 다 로컬
`data/processed/training_dataset.csv`(437,371행, 이미 존재)로 즉시 실행
가능하며 Feast/MLflow/GCS가 필요 없다.

- **Experiment A (baseline)**: `examples/offline_experiments/baseline.yaml` —
  features 21개 전부(`MODEL_FEATURE_COLUMNS`와 동일 목록을 명시적으로 나열)
- **Experiment B (ablation)**: `examples/offline_experiments/ablation_remove_topic_similarity.yaml`
  — `topic_similarity` 제외 20개

## 비범위 (이번 설계가 하지 않는 것)

- 기존 Feast/Redis/FastAPI Serving 구조 삭제·대규모 수정
- Feast 없는 로컬 dataset 신규 조립 경로 추가
- `train.py`/`evaluate.py`/`model_contract.py`/`paired_experiment.py` 수정
- XGBoost 등 새 모델 구현체 추가
- 모델 성능 개선을 위한 임의 feature 신규 계산(파생 피처 계산 로직 추가는 하지 않음
  — config의 `features`는 데이터셋에 이미 존재하는 컬럼만 선택 가능)
- Label 정의, train/test split 비율의 prod 기본값, 평가 데이터 자체 변경
- Auto Research 자율 실행 파이프라인(GitHub 이슈 → K8s Job → Agent Orchestration,
  #516/#546/#550)과의 통합 — 이번 스코프는 순수 로컬 CLI 반복 실행이며, 필요해지면
  후속 스코프로 별도 설계한다
