# Auto Research 오프라인 실험 경로 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** feature 부분집합(ablation) 선택 → 로컬 CSV 학습 → held-out 평가 → baseline
대비 diff까지 로컬 CLI 한 줄로 반복 가능한 오프라인 실험 경로를 만든다.

**Architecture:** 신규 모듈 `autoresearch/model_evaluation/offline_experiment.py` 하나가
`ExperimentConfig`(YAML)를 읽어 split/downsampling/scale_pos_weight/categorical
캐스팅을 기존 `train.py`/`autoresearch/model_training/downsampling.py`의 semantics 그대로 재현하고
(가능한 함수는 직접 import해 재사용), `LGBMModel`로 학습해 held-out test로 평가한
뒤 `ExperimentResult`(JSON)를 낸다. `autoresearch/cli.py`에 `run-experiment` 명령 하나만
추가한다. Feast/ONNX/MLflow 승격 경로는 이 모듈이 전혀 참조하지 않는다.

**Tech Stack:** Python, pydantic v2, pandas, scikit-learn(`train_test_split`,
`roc_auc_score`, `average_precision_score`, `log_loss`), LightGBM(기존
`LGBMModel` wrapper 경유), Typer, pytest.

**Spec:** `docs/specs/2026-09-14-auto-research-offline-experiment-path.md`

**Issue:** #790 · **Branch:** `feat/790-offline-experiment-path`

## Global Constraints

- 무수정 대상: `autoresearch/model_training/train.py`, `autoresearch/model_evaluation/evaluate.py`,
  `autoresearch/feature_engineering/model_contract.py`, `autoresearch/model_evaluation/paired_experiment.py`, Feast/ONNX/서빙
  관련 모든 파일. 이 파일들에서는 **함수를 import해서 쓰기만** 한다.
- 새 top-level 디렉토리·패키지 금지. 기존 `examples/`, `autoresearch/model_training/`,
  `autoresearch/model_evaluation/`을 재사용한다.
- `experiments/` 디렉토리는 쓰지 않는다(개인 벤치마크 로그용, 이 clone의
  `.git/info/exclude` 대상). 예제 config는 `examples/offline_experiments/`.
- 새 모델 구현체(XGBoost 등) 추가 금지 — `LGBMModel`만 사용, 단 `type` 필드로
  확장 가능한 구조는 유지한다.
- `model_type: str → CTRModel` 조회는 **`autoresearch/model_training/model_factory.py`(신규) 하나에만**
  둔다. `autoresearch/model_evaluation/offline_experiment.py`(config/학습/평가/비교 로직)는
  `MODEL_REGISTRY`를 import해서 조회만 하며, 그 dict 자체를 정의하지 않는다 —
  나중에 다른 `CTRModel` 구현체를 추가할 때 `offline_experiment.py`가 한 줄도
  바뀌지 않아야 한다는 요구사항을 코드 구조로 강제하기 위함이다.
- 실패 시에도 `status: "failed"` `ExperimentResult` JSON을 `--output`에 남기고,
  CLI는 반드시 non-zero exit code로 종료한다: 성공 0 / config·baseline 비교조건
  검증 실패(`OfflineExperimentError`) 2 / 그 외 예기치 못한 실행 실패 1.
- `comparison_conditions`(`data_path`/`row_count`/`seed`/`test_size`/`val_size`/
  `sampling_rate`/`model_type`/`model_params`)가 baseline/candidate 간 하나라도
  다르면 `metrics_delta`를 계산하지 않고 즉시 실패한다. `features`는 이 비교에서
  제외한다(ablation의 목적 자체가 feature를 다르게 하는 것).
- `comparison_conditions.model_params`는 config 원문(`"auto"` 등)이 아니라
  학습에 **실제로 쓰인 최종 숫자값**을 기록한다.
- Python 함수는 반환 타입 포함 타입 힌트를 유지한다. 새 모듈 최상단에는
  `[파이프라인]`/`[기능]`/`[비책임]` 3단 docstring을 붙인다(저장소 기존 모듈
  관례, 예: `autoresearch/model_training/downsampling.py`, `autoresearch/feature_engineering/model_contract.py`).

---

## Task 1: 모델 레지스트리 + ExperimentConfig/ExperimentResult 모델 + 로더

**Files:**
- Create: `autoresearch/model_training/model_factory.py` (model_type → CTRModel 매핑, 이 파일 하나가 조회 지점의 전부)
- Test: `tests/model_training/test_model_factory.py`
- Create: `autoresearch/model_evaluation/offline_experiment.py` (이 태스크에서는 모델·로더·예외 클래스만)
- Test: `tests/model_evaluation/test_offline_experiment.py`

**Interfaces:**
- Produces (`autoresearch/model_training/model_factory.py`): `MODEL_REGISTRY: dict[str, type[CTRModel]]`
  (지금은 `{"lightgbm": LGBMModel}` 하나). 이후 다른 `CTRModel` 구현체를 추가할 때
  손대는 파일은 **이 파일뿐**이어야 한다 — `offline_experiment.py`는 이 dict를
  import해서 조회만 한다.
- Produces (`offline_experiment.py`): `OfflineExperimentError(message: str, *, reason_code: Literal["config_error", "comparison_error"])`,
  `ExperimentMeta(name: str, seed: int = 42)`,
  `ModelSection(type: str = "lightgbm", params: dict = {})`,
  `DataSection(path: str, test_size: float | None = None, val_size: float | None = None)`,
  `EvaluationSection(primary_metric: str = "roc_auc")`,
  `ExperimentConfig(experiment, model, data, features: list[str], evaluation)`,
  `ComparisonConditions(data_path, row_count, seed, test_size, val_size, sampling_rate, model_type, model_params: dict[str, float])`,
  `BaselineSummary(experiment_name: str, metrics: dict[str, float])`,
  `ExperimentResult(experiment_name, status, model=None, features=None, metrics=None, comparison_conditions=None, baseline=None, metrics_delta=None, error=None)`,
  `load_experiment_config(path: str | Path) -> ExperimentConfig`,
  `load_experiment_result(path: str | Path) -> ExperimentResult`,
  `write_result(result: ExperimentResult, path: str | Path) -> None`,
  `LABEL_COLUMN: str`

- [ ] **Step 1a: Write the failing registry test**

```python
# tests/model_training/test_model_factory.py
"""autoresearch/model_training/model_factory.py의 CTRModel 레지스트리 테스트."""
from autoresearch.model_training.base import CTRModel
from autoresearch.model_training.lgbm_model import LGBMModel
from autoresearch.model_training.model_factory import MODEL_REGISTRY


def test_lightgbm_registered_as_ctrmodel_subclass() -> None:
    assert MODEL_REGISTRY["lightgbm"] is LGBMModel
    assert issubclass(MODEL_REGISTRY["lightgbm"], CTRModel)
```

- [ ] **Step 1b: Run test to verify it fails**

Run: `uv run python -m pytest tests/model_training/test_model_factory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoresearch.model_training.model_factory'`

- [ ] **Step 1c: Write `autoresearch/model_training/model_factory.py`**

```python
# autoresearch/model_training/model_factory.py
"""CTRModel 구현체 이름 → 클래스 레지스트리.

[파이프라인] `autoresearch/model_evaluation/offline_experiment.py`가 `ExperimentConfig.model.type`
문자열로 학습에 쓸 `CTRModel` 구현체를 고르는 유일한 조회 지점이다.

[기능] `MODEL_REGISTRY`에 새 `CTRModel` 구현체를 한 줄 추가하는 것만으로
확장한다 — 이 파일 밖(`offline_experiment.py`의 config/학습/평가/비교 로직)은
전혀 수정하지 않는다.

[비책임] `CTRModel` 구현체 자체의 알고리즘·하이퍼파라미터는 각 구현체 모듈
(`lgbm_model.py` 등)이 소유한다.
"""

from __future__ import annotations

from autoresearch.model_training.base import CTRModel
from autoresearch.model_training.lgbm_model import LGBMModel

MODEL_REGISTRY: dict[str, type[CTRModel]] = {"lightgbm": LGBMModel}
```

- [ ] **Step 1d: Run registry test to verify it passes**

Run: `uv run python -m pytest tests/model_training/test_model_factory.py -v`
Expected: PASS (1 test)

- [ ] **Step 1: Write the failing tests**

```python
# tests/model_evaluation/test_offline_experiment.py
"""autoresearch/model_evaluation/offline_experiment.py의 config/result 모델·로더 테스트."""
import json

import pytest
import yaml

from autoresearch.model_evaluation import offline_experiment as oe


def _valid_config_dict() -> dict:
    return {
        "experiment": {"name": "baseline", "seed": 42},
        "model": {"type": "lightgbm", "params": {}},
        "data": {"path": "data/processed/training_dataset.csv"},
        "features": ["age_group", "recent_click_count_7d"],
        "evaluation": {"primary_metric": "roc_auc"},
    }


def test_load_experiment_config_valid(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_valid_config_dict()), encoding="utf-8")

    config = oe.load_experiment_config(config_path)

    assert config.experiment.name == "baseline"
    assert config.experiment.seed == 42
    assert config.model.type == "lightgbm"
    assert config.features == ["age_group", "recent_click_count_7d"]
    assert config.data.test_size is None
    assert config.data.val_size is None


def test_load_experiment_config_missing_file_raises_config_error(tmp_path) -> None:
    with pytest.raises(oe.OfflineExperimentError) as excinfo:
        oe.load_experiment_config(tmp_path / "missing.yaml")
    assert excinfo.value.reason_code == "config_error"


def test_load_experiment_config_rejects_empty_features(tmp_path) -> None:
    payload = _valid_config_dict()
    payload["features"] = []
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(oe.OfflineExperimentError) as excinfo:
        oe.load_experiment_config(config_path)
    assert excinfo.value.reason_code == "config_error"


def test_load_experiment_config_rejects_duplicate_features(tmp_path) -> None:
    payload = _valid_config_dict()
    payload["features"] = ["age_group", "age_group"]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(oe.OfflineExperimentError):
        oe.load_experiment_config(config_path)


def test_load_experiment_config_rejects_label_column_in_features(tmp_path) -> None:
    payload = _valid_config_dict()
    payload["features"] = ["age_group", oe.LABEL_COLUMN]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(oe.OfflineExperimentError):
        oe.load_experiment_config(config_path)


def test_load_experiment_config_rejects_unknown_model_type(tmp_path) -> None:
    payload = _valid_config_dict()
    payload["model"]["type"] = "xgboost"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(oe.OfflineExperimentError):
        oe.load_experiment_config(config_path)


def test_experiment_result_roundtrip(tmp_path) -> None:
    result = oe.ExperimentResult(
        experiment_name="baseline",
        status="success",
        model="lightgbm",
        features=["age_group"],
        metrics={"roc_auc": 0.7, "pr_auc": 0.1, "log_loss": 0.2},
        comparison_conditions=oe.ComparisonConditions(
            data_path="data/processed/training_dataset.csv",
            row_count=100,
            seed=42,
            test_size=0.2,
            val_size=0.2,
            sampling_rate=1.0,
            model_type="lightgbm",
            model_params={"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31, "scale_pos_weight": 1.0},
        ),
    )
    output_path = tmp_path / "result.json"

    oe.write_result(result, output_path)
    loaded = oe.load_experiment_result(output_path)

    assert loaded.experiment_name == "baseline"
    assert loaded.metrics == {"roc_auc": 0.7, "pr_auc": 0.1, "log_loss": 0.2}
    assert loaded.comparison_conditions.row_count == 100
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "success"


def test_load_experiment_result_rejects_invalid_json(tmp_path) -> None:
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(oe.OfflineExperimentError) as excinfo:
        oe.load_experiment_result(bad_path)
    assert excinfo.value.reason_code == "config_error"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/model_evaluation/test_offline_experiment.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoresearch.model_evaluation.offline_experiment'`

- [ ] **Step 3: Write the module (models/loaders part)**

```python
# autoresearch/model_evaluation/offline_experiment.py
"""Auto Research용 로컬 오프라인 feature/model 실험 경로.

[파이프라인] Training Dataset(CSV, 이미 존재) → feature 부분집합 선택 → 학습 →
held-out 평가 → baseline 대비 diff 구간을 담당한다. `build-features`(Feast PIT
조립)와 prod 승격 경로(`train.py`/`evaluate.py`/`paired_experiment.py`)의
**뒤**와 **바깥**에 위치하며, 이 모듈의 산출물은 champion 승격 대상이 아니다.

[기능] `ExperimentConfig`(YAML)를 파싱하고, prod `config.yaml` 기본값을
물려받아 병합한 뒤, train/val/test 분할·negative downsampling·scale_pos_weight
결정·categorical dtype 캐스팅을 `train.py`/`autoresearch/model_training/downsampling.py`의
기존 semantics 그대로 재현해 `LGBMModel`을 학습·평가하고 `ExperimentResult`
(JSON)를 만든다. `apply_baseline_comparison()`은 두 결과의 비교 조건이
일치하는지 확인한 뒤에만 metric delta를 계산한다(fail-fast).

[비책임] MLflow run 로깅, ONNX 변환, promotion evidence, snapshot manifest,
Model Registry 등록, Feast 조회는 다루지 않는다 — 이 경로는 Feast 없이
이미 존재하는 로컬 CSV만 입력으로 받는다. prod `MODEL_FEATURE_COLUMNS`/ONNX
입력 순서 계약은 건드리지 않는다(`autoresearch/feature_engineering/model_contract.py` 무수정).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from autoresearch.model_training.model_factory import MODEL_REGISTRY
from autoresearch.model_training.training_provenance import write_manifest_atomic

LABEL_COLUMN = "clicked"
_KNOWN_METRICS = ("roc_auc", "pr_auc", "log_loss")


class OfflineExperimentError(Exception):
    """offline_experiment의 설정·비교조건 검증 실패. exit code를 가르는 데 쓴다."""

    def __init__(self, message: str, *, reason_code: Literal["config_error", "comparison_error"]):
        super().__init__(message)
        self.reason_code = reason_code


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentMeta(_StrictModel):
    name: str
    seed: int = 42


class ModelSection(_StrictModel):
    type: str = "lightgbm"
    params: dict = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def _known_type(cls, value: str) -> str:
        if value not in MODEL_REGISTRY:
            raise ValueError(
                f"지원하지 않는 model.type: {value!r} (지원: {sorted(MODEL_REGISTRY)})"
            )
        return value


class DataSection(_StrictModel):
    path: str
    test_size: Optional[float] = None
    val_size: Optional[float] = None


class EvaluationSection(_StrictModel):
    primary_metric: str = "roc_auc"

    @field_validator("primary_metric")
    @classmethod
    def _known_metric(cls, value: str) -> str:
        if value not in _KNOWN_METRICS:
            raise ValueError(f"지원하지 않는 primary_metric: {value!r} (지원: {_KNOWN_METRICS})")
        return value


class ExperimentConfig(_StrictModel):
    experiment: ExperimentMeta
    model: ModelSection = ModelSection()
    data: DataSection
    features: list[str]
    evaluation: EvaluationSection = EvaluationSection()

    @field_validator("features")
    @classmethod
    def _valid_features(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("features가 비었습니다 — 최소 1개 이상 지정해야 합니다")
        if len(value) != len(set(value)):
            raise ValueError(f"features에 중복이 있습니다: {value}")
        if LABEL_COLUMN in value:
            raise ValueError(f"features에 라벨 컬럼({LABEL_COLUMN})을 포함할 수 없습니다")
        return value


class ComparisonConditions(_StrictModel):
    data_path: str
    row_count: int
    seed: int
    test_size: float
    val_size: float
    sampling_rate: float
    model_type: str
    model_params: dict[str, float]


class BaselineSummary(_StrictModel):
    experiment_name: str
    metrics: dict[str, float]


class ExperimentResult(_StrictModel):
    experiment_name: str
    status: Literal["success", "failed"]
    model: Optional[str] = None
    features: Optional[list[str]] = None
    metrics: Optional[dict[str, float]] = None
    comparison_conditions: Optional[ComparisonConditions] = None
    baseline: Optional[BaselineSummary] = None
    metrics_delta: Optional[dict[str, float]] = None
    error: Optional[str] = None


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """ExperimentConfig YAML을 읽고 구조적으로 검증한다.

    데이터셋 컬럼 존재 여부(features가 실제 CSV에 있는지)는 여기서 검증하지
    않는다 — 이 함수는 CSV를 읽지 않으며, 그 검증은 `run_experiment()`가 한다.
    """
    try:
        with open(path, "r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except OSError as error:
        raise OfflineExperimentError(
            f"config 파일을 읽을 수 없습니다: {path}", reason_code="config_error"
        ) from error
    try:
        return ExperimentConfig.model_validate(raw)
    except ValidationError as error:
        raise OfflineExperimentError(
            f"config 검증 실패({path}): {error}", reason_code="config_error"
        ) from error


def load_experiment_result(path: str | Path) -> ExperimentResult:
    """`--baseline-result`로 넘긴 ExperimentResult JSON을 읽는다."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as error:
        raise OfflineExperimentError(
            f"baseline 결과 파일을 읽을 수 없습니다: {path}", reason_code="config_error"
        ) from error
    except json.JSONDecodeError as error:
        raise OfflineExperimentError(
            f"baseline 결과가 올바른 JSON이 아닙니다: {path}", reason_code="config_error"
        ) from error
    try:
        return ExperimentResult.model_validate(payload)
    except ValidationError as error:
        raise OfflineExperimentError(
            f"baseline 결과 검증 실패({path}): {error}", reason_code="config_error"
        ) from error


def write_result(result: ExperimentResult, path: str | Path) -> None:
    """ExperimentResult를 원자적으로 JSON 파일에 게시한다(기존 아티팩트 저장 관례 재사용)."""
    write_manifest_atomic(result, Path(path))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/model_evaluation/test_offline_experiment.py tests/model_training/test_model_factory.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add autoresearch/model_training/model_factory.py tests/model_training/test_model_factory.py \
  autoresearch/model_evaluation/offline_experiment.py tests/model_evaluation/test_offline_experiment.py
git commit -m "feat: CTRModel 레지스트리와 offline experiment config/result 모델·로더를 추가한다"
```

---

## Task 2: 분할·downsampling·scale_pos_weight·학습·평가 헬퍼

**Files:**
- Modify: `autoresearch/model_evaluation/offline_experiment.py` (Task 1 파일에 이어서 작성)
- Test: `tests/model_evaluation/test_offline_experiment.py` (이어서 추가)

**Interfaces:**
- Consumes (Task 1): `OfflineExperimentError`, `ExperimentConfig`, `LABEL_COLUMN`, `MODEL_REGISTRY`
- Consumes (기존 저장소, 무수정 재사용): `autoresearch.model_training.train.require_binary_labels(labels, *, stage) -> tuple[int, int]`,
  `autoresearch.model_training.train.compute_auto_scale_pos_weight(labels) -> float`,
  `autoresearch.model_training.train.collect_categorical_categories(X_train, X_val, categorical_columns) -> dict`,
  `autoresearch.model_training.train.load_config(path) -> dict`,
  `autoresearch.model_training.downsampling.downsample_negatives(X, y, sampling_rate, random_state) -> tuple[pd.DataFrame, pd.Series, float]`,
  `autoresearch.model_training.downsampling.apply_downsampling_calibration(q, sampling_rate)`,
  `autoresearch.feature_engineering.model_contract.CATEGORICAL_FEATURE_COLUMNS: tuple[str, ...]`
- Produces: `_resolve_split_sizes(data: DataSection, prod_data_defaults: dict) -> tuple[float, float]`,
  `_split_dataset(dataset: pd.DataFrame, *, test_size: float, val_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`,
  `_resolve_scale_pos_weight(configured_spw: object, *, nominal_sampling_rate: float, y_train: pd.Series) -> float`,
  `_evaluate_held_out(model, dataset: pd.DataFrame, feature_columns: list[str], categorical_columns: list[str], *, sampling_rate: float) -> dict[str, float]`,
  `_effective_model_params(config: ExperimentConfig, prod_model_defaults: dict) -> dict`,
  `run_experiment(config: ExperimentConfig, *, project_root: Path) -> ExperimentResult`

- [ ] **Step 1: Write the failing tests**

```python
# tests/model_evaluation/test_offline_experiment.py 에 이어서 추가

import numpy as np
import pandas as pd
import yaml

from autoresearch.feature_engineering.model_contract import CATEGORICAL_FEATURE_COLUMNS


def _synthetic_dataset(n: int = 300, seed: int = 7) -> pd.DataFrame:
    """CATEGORICAL_FEATURE_COLUMNS의 age_group 포함, 수치 피처 2개 + 라벨."""
    rng = np.random.default_rng(seed)
    age_group = rng.choice(["10s", "20s", "30s"], size=n)
    recent_click_count_7d = rng.integers(0, 20, size=n)
    view_count = rng.integers(0, 100000, size=n)
    # 라벨은 양성 비율이 낮아도(~15%) stratify가 되도록 결정론적으로 생성.
    clicked = (rng.random(n) < 0.15).astype(int)
    return pd.DataFrame(
        {
            "age_group": age_group,
            "recent_click_count_7d": recent_click_count_7d,
            "view_count": view_count,
            "clicked": clicked,
        }
    )


def _write_config(tmp_path, dataset_path, *, features, seed=42, extra_model_params=None, sampling_rate=None):
    payload = {
        "experiment": {"name": "t", "seed": seed},
        "model": {"type": "lightgbm", "params": extra_model_params or {}},
        "data": {"path": str(dataset_path)},
        "features": features,
        "evaluation": {"primary_metric": "roc_auc"},
    }
    if sampling_rate is not None:
        payload["model"]["params"]["sampling_rate"] = sampling_rate
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def test_split_dataset_reproducible_with_same_seed() -> None:
    dataset = _synthetic_dataset()

    train1, val1, test1 = oe._split_dataset(dataset, test_size=0.2, val_size=0.2, seed=42)
    train2, val2, test2 = oe._split_dataset(dataset, test_size=0.2, val_size=0.2, seed=42)

    assert list(train1.index) == list(train2.index)
    assert list(val1.index) == list(val2.index)
    assert list(test1.index) == list(test2.index)
    assert len(train1) + len(val1) + len(test1) == len(dataset)


def test_split_dataset_rejects_too_large_sizes() -> None:
    dataset = _synthetic_dataset()
    with pytest.raises(oe.OfflineExperimentError) as excinfo:
        oe._split_dataset(dataset, test_size=0.6, val_size=0.6, seed=42)
    assert excinfo.value.reason_code == "config_error"


def test_resolve_scale_pos_weight_auto() -> None:
    y_train = pd.Series([0] * 90 + [1] * 10)
    weight = oe._resolve_scale_pos_weight("auto", nominal_sampling_rate=1.0, y_train=y_train)
    assert weight == pytest.approx(9.0)


def test_resolve_scale_pos_weight_forced_to_one_under_downsampling() -> None:
    y_train = pd.Series([0] * 90 + [1] * 10)
    weight = oe._resolve_scale_pos_weight("auto", nominal_sampling_rate=0.5, y_train=y_train)
    assert weight == 1


def test_resolve_scale_pos_weight_rejects_explicit_value_with_downsampling() -> None:
    y_train = pd.Series([0] * 90 + [1] * 10)
    with pytest.raises(oe.OfflineExperimentError) as excinfo:
        oe._resolve_scale_pos_weight(5.0, nominal_sampling_rate=0.5, y_train=y_train)
    assert excinfo.value.reason_code == "config_error"


def test_run_experiment_produces_metrics_and_comparison_conditions(tmp_path) -> None:
    dataset = _synthetic_dataset()
    dataset_path = tmp_path / "training_dataset.csv"
    dataset.to_csv(dataset_path, index=False)
    config_path = _write_config(
        tmp_path, dataset_path, features=["age_group", "recent_click_count_7d", "view_count"]
    )
    config = oe.load_experiment_config(config_path)

    result = oe.run_experiment(config, project_root=Path(__file__).resolve().parents[1])

    assert result.status == "success"
    assert result.model == "lightgbm"
    assert set(result.features) == {"age_group", "recent_click_count_7d", "view_count"}
    assert set(result.metrics) == {"roc_auc", "pr_auc", "log_loss"}
    assert 0.0 <= result.metrics["roc_auc"] <= 1.0
    assert result.comparison_conditions.row_count == len(dataset)
    assert result.comparison_conditions.seed == 42
    assert result.comparison_conditions.model_params["n_estimators"] == 200


def test_run_experiment_ablation_excludes_categorical_column_without_error(tmp_path) -> None:
    """categorical feature(age_group)를 뺀 부분집합도 KeyError 없이 평가까지 끝나야 한다."""
    dataset = _synthetic_dataset()
    dataset_path = tmp_path / "training_dataset.csv"
    dataset.to_csv(dataset_path, index=False)
    config_path = _write_config(tmp_path, dataset_path, features=["recent_click_count_7d", "view_count"])
    config = oe.load_experiment_config(config_path)

    result = oe.run_experiment(config, project_root=Path(__file__).resolve().parents[1])

    assert result.status == "success"
    assert "age_group" not in result.features


def test_run_experiment_missing_feature_column_raises_config_error(tmp_path) -> None:
    dataset = _synthetic_dataset()
    dataset_path = tmp_path / "training_dataset.csv"
    dataset.to_csv(dataset_path, index=False)
    config_path = _write_config(tmp_path, dataset_path, features=["age_group", "not_a_real_column"])
    config = oe.load_experiment_config(config_path)

    with pytest.raises(oe.OfflineExperimentError) as excinfo:
        oe.run_experiment(config, project_root=Path(__file__).resolve().parents[1])
    assert excinfo.value.reason_code == "config_error"


def test_run_experiment_downsampling_changes_realized_row_count_signal(tmp_path) -> None:
    dataset = _synthetic_dataset(n=600)
    dataset_path = tmp_path / "training_dataset.csv"
    dataset.to_csv(dataset_path, index=False)
    config_path = _write_config(
        tmp_path,
        dataset_path,
        features=["age_group", "recent_click_count_7d", "view_count"],
        sampling_rate=0.3,
    )
    config = oe.load_experiment_config(config_path)

    result = oe.run_experiment(config, project_root=Path(__file__).resolve().parents[1])

    assert result.status == "success"
    assert result.comparison_conditions.sampling_rate == 0.3
    assert result.comparison_conditions.model_params["scale_pos_weight"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/model_evaluation/test_offline_experiment.py -v -k "split_dataset or scale_pos_weight or run_experiment"`
Expected: FAIL with `AttributeError: module 'autoresearch.model_evaluation.offline_experiment' has no attribute '_split_dataset'` (등)

- [ ] **Step 3: Append the implementation**

```python
# autoresearch/model_evaluation/offline_experiment.py 끝에 이어서 추가

import math
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from autoresearch.feature_engineering.model_contract import CATEGORICAL_FEATURE_COLUMNS
from autoresearch.model_training.downsampling import apply_downsampling_calibration, downsample_negatives
from autoresearch.model_training.train import (
    collect_categorical_categories,
    compute_auto_scale_pos_weight,
    load_config,
    require_binary_labels,
)


def _resolve_split_sizes(data: DataSection, prod_data_defaults: dict) -> tuple[float, float]:
    """config에 없으면 prod config.yaml의 data.test_size/val_size를 그대로 쓴다."""
    test_size = data.test_size if data.test_size is not None else float(prod_data_defaults["test_size"])
    val_size = data.val_size if data.val_size is not None else float(prod_data_defaults["val_size"])
    if test_size + val_size >= 1:
        raise OfflineExperimentError(
            f"test_size({test_size}) + val_size({val_size}) >= 1 입니다 — "
            "train에 데이터가 남지 않습니다.",
            reason_code="config_error",
        )
    return test_size, val_size


def _split_dataset(
    dataset: pd.DataFrame, *, test_size: float, val_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """train.py Step 2와 동일한 2단계 stratified split(같은 seed, stratify=clicked)."""
    if test_size + val_size >= 1:
        raise OfflineExperimentError(
            f"test_size({test_size}) + val_size({val_size}) >= 1 입니다.",
            reason_code="config_error",
        )
    source_positions = np.arange(len(dataset))
    train_val_positions, test_positions = train_test_split(
        source_positions,
        test_size=test_size,
        random_state=seed,
        stratify=dataset[LABEL_COLUMN],
    )
    val_ratio_within_train_val = val_size / (1 - test_size)
    train_positions, val_positions = train_test_split(
        train_val_positions,
        test_size=val_ratio_within_train_val,
        random_state=seed,
        stratify=dataset.iloc[train_val_positions][LABEL_COLUMN],
    )
    train_df = dataset.iloc[train_positions].copy()
    val_df = dataset.iloc[val_positions].copy()
    test_df = dataset.iloc[test_positions].copy()
    for stage, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        require_binary_labels(split_df[LABEL_COLUMN], stage=stage)
    return train_df, val_df, test_df


def _resolve_scale_pos_weight(
    configured_spw: object, *, nominal_sampling_rate: float, y_train: pd.Series
) -> float:
    """train.py Step 5와 동일한 규칙(downsampling 활성 시 강제 1, 'auto', 고정값)."""
    if nominal_sampling_rate < 1.0:
        explicit_numeric = configured_spw != "auto" and float(configured_spw) != 1
        if explicit_numeric:
            raise OfflineExperimentError(
                f"downsampling(sampling_rate={nominal_sampling_rate})과 명시적 "
                f"scale_pos_weight={configured_spw}를 함께 설정했습니다 — 이중 보정입니다.",
                reason_code="config_error",
            )
        return 1.0
    if configured_spw == "auto":
        return float(compute_auto_scale_pos_weight(y_train))
    return float(configured_spw)


def _evaluate_held_out(
    model: object,
    dataset: pd.DataFrame,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str],
    *,
    sampling_rate: float,
) -> dict[str, float]:
    """evaluate.py의 held-out 지표 계산과 동일한 로직(raw roc/pr, calibrated log_loss).

    `evaluate_held_out_metrics`를 그대로 재사용하지 않는다 — 그 함수는 prod 고정
    `CATEGORICAL_FEATURE_COLUMNS` 5개 전부를 캐스팅하려 시도해, ablation으로
    categorical feature 하나가 feature_columns에서 빠지면 KeyError로 죽는다.
    """
    features = dataset[list(feature_columns)].copy()
    for column in categorical_columns:
        features[column] = features[column].astype("category")
    labels = dataset[LABEL_COLUMN]
    raw_proba = model.predict_proba(features)[:, 1]
    calibrated_proba = apply_downsampling_calibration(raw_proba, sampling_rate)
    return {
        "roc_auc": float(roc_auc_score(labels, raw_proba)),
        "pr_auc": float(average_precision_score(labels, raw_proba)),
        "log_loss": float(log_loss(labels, calibrated_proba)),
    }


def _effective_model_params(config: ExperimentConfig, prod_model_defaults: dict) -> dict:
    """prod config.yaml의 model: 섹션 기본값에 ExperimentConfig.model.params를 덮어쓴다."""
    merged = {
        "n_estimators": prod_model_defaults["n_estimators"],
        "learning_rate": prod_model_defaults["learning_rate"],
        "num_leaves": prod_model_defaults["num_leaves"],
        "scale_pos_weight": prod_model_defaults["scale_pos_weight"],
        "sampling_rate": prod_model_defaults["sampling_rate"],
    }
    merged.update(config.model.params)
    return merged


def run_experiment(config: ExperimentConfig, *, project_root: Path) -> ExperimentResult:
    """ExperimentConfig 하나를 실행해 ExperimentResult(성공)를 만든다.

    실패는 예외(`OfflineExperimentError` 또는 예기치 못한 표준 예외)로 전파한다 —
    이 함수는 실패 결과를 만들지 않는다. 실패 시 `status: "failed"` 결과를 만드는
    책임은 CLI 계층(`autoresearch/cli.py`)에 있다.
    """
    prod_config = load_config(str(project_root / "autoresearch" / "model_training" / "config.yaml"))

    data_path = config.data.path
    resolved_data_path = data_path if Path(data_path).is_absolute() else str(project_root / data_path)
    try:
        dataset = pd.read_csv(resolved_data_path)
    except OSError as error:
        raise OfflineExperimentError(
            f"데이터셋을 읽을 수 없습니다: {resolved_data_path}", reason_code="config_error"
        ) from error

    missing = [name for name in config.features if name not in dataset.columns]
    if missing:
        raise OfflineExperimentError(
            f"features에 데이터셋에 없는 컬럼이 있습니다: {missing}. "
            f"데이터셋 컬럼: {sorted(dataset.columns)}",
            reason_code="config_error",
        )

    feature_columns = list(config.features)
    categorical_columns = [c for c in CATEGORICAL_FEATURE_COLUMNS if c in feature_columns]

    test_size, val_size = _resolve_split_sizes(config.data, prod_config["data"])
    seed = config.experiment.seed
    train_df, val_df, test_df = _split_dataset(dataset, test_size=test_size, val_size=val_size, seed=seed)

    X_train = train_df[feature_columns].copy()
    y_train = train_df[LABEL_COLUMN].copy()
    X_val = val_df[feature_columns].copy()

    model_params = _effective_model_params(config, prod_config["model"])
    nominal_sampling_rate = float(model_params["sampling_rate"])
    if not math.isfinite(nominal_sampling_rate) or not (0.0 < nominal_sampling_rate <= 1.0):
        raise OfflineExperimentError(
            f"sampling_rate는 (0, 1] 범위여야 합니다: {nominal_sampling_rate}",
            reason_code="config_error",
        )

    realized_sampling_rate = 1.0
    if nominal_sampling_rate < 1.0:
        X_train, y_train, realized_sampling_rate = downsample_negatives(
            X_train, y_train, nominal_sampling_rate, random_state=seed
        )

    scale_pos_weight = _resolve_scale_pos_weight(
        model_params["scale_pos_weight"], nominal_sampling_rate=nominal_sampling_rate, y_train=y_train
    )

    collect_categorical_categories(X_train, X_val, categorical_columns)

    model_cls = MODEL_REGISTRY[config.model.type]
    model = model_cls(
        scale_pos_weight=scale_pos_weight,
        n_estimators=int(model_params["n_estimators"]),
        learning_rate=float(model_params["learning_rate"]),
        num_leaves=int(model_params["num_leaves"]),
        random_state=seed,
    )
    model.fit(X_train, y_train, categorical_features=categorical_columns)

    metrics = _evaluate_held_out(
        model, test_df, feature_columns, categorical_columns, sampling_rate=realized_sampling_rate
    )

    comparison_conditions = ComparisonConditions(
        data_path=config.data.path,
        row_count=len(dataset),
        seed=seed,
        test_size=test_size,
        val_size=val_size,
        sampling_rate=nominal_sampling_rate,
        model_type=config.model.type,
        model_params={
            "n_estimators": float(model_params["n_estimators"]),
            "learning_rate": float(model_params["learning_rate"]),
            "num_leaves": float(model_params["num_leaves"]),
            "scale_pos_weight": float(scale_pos_weight),
        },
    )

    return ExperimentResult(
        experiment_name=config.experiment.name,
        status="success",
        model=config.model.type,
        features=feature_columns,
        metrics=metrics,
        comparison_conditions=comparison_conditions,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/model_evaluation/test_offline_experiment.py -v`
Expected: PASS (모든 테스트, Task 1 것 포함)

- [ ] **Step 5: Commit**

```bash
git add autoresearch/model_evaluation/offline_experiment.py tests/model_evaluation/test_offline_experiment.py
git commit -m "feat: offline experiment 분할·downsampling·학습·평가 로직을 추가한다"
```

---

## Task 3: baseline 비교(fail-fast) 추가

**Files:**
- Modify: `autoresearch/model_evaluation/offline_experiment.py`
- Test: `tests/model_evaluation/test_offline_experiment.py`

**Interfaces:**
- Consumes (Task 1/2): `ExperimentResult`, `ComparisonConditions`, `OfflineExperimentError`, `BaselineSummary`
- Produces: `apply_baseline_comparison(result: ExperimentResult, baseline: ExperimentResult) -> ExperimentResult`

- [ ] **Step 1: Write the failing tests**

```python
# tests/model_evaluation/test_offline_experiment.py 에 이어서 추가

def _make_result(*, status="success", **comparison_overrides) -> oe.ExperimentResult:
    conditions = {
        "data_path": "data/processed/training_dataset.csv",
        "row_count": 100,
        "seed": 42,
        "test_size": 0.2,
        "val_size": 0.2,
        "sampling_rate": 1.0,
        "model_type": "lightgbm",
        "model_params": {"n_estimators": 200.0, "learning_rate": 0.05, "num_leaves": 31.0, "scale_pos_weight": 1.0},
    }
    conditions.update(comparison_overrides)
    return oe.ExperimentResult(
        experiment_name="candidate" if status == "success" else "failed_run",
        status=status,
        model="lightgbm",
        features=["age_group"],
        metrics={"roc_auc": 0.7, "pr_auc": 0.1, "log_loss": 0.2},
        comparison_conditions=oe.ComparisonConditions(**conditions) if status == "success" else None,
    )


def test_apply_baseline_comparison_computes_delta_when_conditions_match() -> None:
    baseline = _make_result()
    candidate = _make_result()
    candidate.metrics["roc_auc"] = 0.65
    candidate.features = ["recent_click_count_7d"]  # feature는 달라도 된다

    compared = oe.apply_baseline_comparison(candidate, baseline)

    assert compared.baseline.experiment_name == "candidate"
    assert compared.metrics_delta["roc_auc"] == pytest.approx(-0.05)


def test_apply_baseline_comparison_fails_on_seed_mismatch() -> None:
    baseline = _make_result(seed=42)
    candidate = _make_result(seed=7)

    with pytest.raises(oe.OfflineExperimentError) as excinfo:
        oe.apply_baseline_comparison(candidate, baseline)
    assert excinfo.value.reason_code == "comparison_error"


def test_apply_baseline_comparison_fails_on_model_params_mismatch() -> None:
    baseline = _make_result()
    candidate = _make_result(model_params={"n_estimators": 500.0, "learning_rate": 0.05, "num_leaves": 31.0, "scale_pos_weight": 1.0})

    with pytest.raises(oe.OfflineExperimentError) as excinfo:
        oe.apply_baseline_comparison(candidate, baseline)
    assert excinfo.value.reason_code == "comparison_error"


def test_apply_baseline_comparison_rejects_non_success_baseline() -> None:
    baseline = _make_result(status="failed")
    candidate = _make_result()

    with pytest.raises(oe.OfflineExperimentError) as excinfo:
        oe.apply_baseline_comparison(candidate, baseline)
    assert excinfo.value.reason_code == "comparison_error"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/model_evaluation/test_offline_experiment.py -v -k baseline_comparison`
Expected: FAIL with `AttributeError: module 'autoresearch.model_evaluation.offline_experiment' has no attribute 'apply_baseline_comparison'`

- [ ] **Step 3: Append the implementation**

```python
# autoresearch/model_evaluation/offline_experiment.py 끝에 이어서 추가

_COMPARISON_FIELDS = (
    "data_path",
    "row_count",
    "seed",
    "test_size",
    "val_size",
    "sampling_rate",
    "model_type",
    "model_params",
)


def apply_baseline_comparison(result: ExperimentResult, baseline: ExperimentResult) -> ExperimentResult:
    """baseline과 비교 조건이 완전히 같을 때만 metrics_delta를 계산한다(fail-fast).

    features는 비교 대상에서 제외한다 — ablation 실험의 목적 자체가 feature
    목록을 다르게 하는 것이라, feature 차이는 실패 사유가 아니다.
    """
    if baseline.status != "success" or baseline.comparison_conditions is None:
        raise OfflineExperimentError(
            "baseline 결과가 성공 상태가 아니라 비교할 수 없습니다.", reason_code="comparison_error"
        )
    if result.status != "success" or result.comparison_conditions is None:
        raise OfflineExperimentError(
            "candidate 결과가 성공 상태가 아니라 비교할 수 없습니다.", reason_code="comparison_error"
        )

    baseline_conditions = baseline.comparison_conditions.model_dump()
    candidate_conditions = result.comparison_conditions.model_dump()
    mismatches = [
        f"{field}: baseline={baseline_conditions[field]!r} candidate={candidate_conditions[field]!r}"
        for field in _COMPARISON_FIELDS
        if baseline_conditions[field] != candidate_conditions[field]
    ]
    if mismatches:
        raise OfflineExperimentError(
            "baseline과 candidate의 비교 조건이 다릅니다 — feature 목록 외 조건은 "
            "동일해야 공정한 비교입니다:\n" + "\n".join(mismatches),
            reason_code="comparison_error",
        )

    delta = {name: result.metrics[name] - baseline.metrics[name] for name in result.metrics}
    return result.model_copy(
        update={
            "baseline": BaselineSummary(experiment_name=baseline.experiment_name, metrics=baseline.metrics),
            "metrics_delta": delta,
        }
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/model_evaluation/test_offline_experiment.py -v`
Expected: PASS (전체)

- [ ] **Step 5: Commit**

```bash
git add autoresearch/model_evaluation/offline_experiment.py tests/model_evaluation/test_offline_experiment.py
git commit -m "feat: offline experiment baseline 비교에 fail-fast 조건 검증을 추가한다"
```

---

## Task 4: CLI 명령 `run-experiment`

**Files:**
- Modify: `autoresearch/cli.py` (명령 추가 + 모듈 docstring 갱신)
- Test: `tests/cli/test_cli.py`

**Interfaces:**
- Consumes (Task 1~3): `offline_experiment.load_experiment_config`, `load_experiment_result`,
  `run_experiment`, `apply_baseline_comparison`, `write_result`, `ExperimentResult`, `OfflineExperimentError`

- [ ] **Step 1: Write the failing tests**

```python
# tests/cli/test_cli.py 에 이어서 추가
import yaml


def _write_offline_experiment_dataset(tmp_path):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(3)
    n = 300
    dataset = pd.DataFrame(
        {
            "age_group": rng.choice(["10s", "20s", "30s"], size=n),
            "recent_click_count_7d": rng.integers(0, 20, size=n),
            "view_count": rng.integers(0, 100000, size=n),
            "clicked": (rng.random(n) < 0.15).astype(int),
        }
    )
    dataset_path = tmp_path / "training_dataset.csv"
    dataset.to_csv(dataset_path, index=False)
    return dataset_path


def _write_offline_experiment_config(tmp_path, dataset_path, *, name="baseline", features=None, seed=42):
    payload = {
        "experiment": {"name": name, "seed": seed},
        "model": {"type": "lightgbm", "params": {}},
        "data": {"path": str(dataset_path)},
        "features": features or ["age_group", "recent_click_count_7d", "view_count"],
        "evaluation": {"primary_metric": "roc_auc"},
    }
    config_path = tmp_path / f"{name}.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def test_run_experiment_succeeds_and_writes_result(tmp_path) -> None:
    dataset_path = _write_offline_experiment_dataset(tmp_path)
    config_path = _write_offline_experiment_config(tmp_path, dataset_path)
    output = tmp_path / "result.json"

    result = CliRunner().invoke(
        cli.app,
        ["run-experiment", "--config", str(config_path), "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert payload["experiment_name"] == "baseline"


def test_run_experiment_with_baseline_computes_delta(tmp_path) -> None:
    dataset_path = _write_offline_experiment_dataset(tmp_path)
    baseline_config = _write_offline_experiment_config(tmp_path, dataset_path, name="baseline")
    baseline_output = tmp_path / "baseline_result.json"
    CliRunner().invoke(
        cli.app, ["run-experiment", "--config", str(baseline_config), "--output", str(baseline_output)]
    )

    candidate_config = _write_offline_experiment_config(
        tmp_path, dataset_path, name="candidate", features=["recent_click_count_7d", "view_count"]
    )
    candidate_output = tmp_path / "candidate_result.json"
    result = CliRunner().invoke(
        cli.app,
        [
            "run-experiment",
            "--config",
            str(candidate_config),
            "--baseline-result",
            str(baseline_output),
            "--output",
            str(candidate_output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(candidate_output.read_text(encoding="utf-8"))
    assert payload["baseline"]["experiment_name"] == "baseline"
    assert "roc_auc" in payload["metrics_delta"]


def test_run_experiment_invalid_config_exits_two_without_output(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"experiment": {"name": "x"}}), encoding="utf-8")
    output = tmp_path / "result.json"

    result = CliRunner().invoke(
        cli.app, ["run-experiment", "--config", str(config_path), "--output", str(output)]
    )

    assert result.exit_code == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"


def test_run_experiment_mismatched_baseline_exits_two(tmp_path) -> None:
    dataset_path = _write_offline_experiment_dataset(tmp_path)
    baseline_config = _write_offline_experiment_config(tmp_path, dataset_path, name="baseline", seed=42)
    baseline_output = tmp_path / "baseline_result.json"
    CliRunner().invoke(
        cli.app, ["run-experiment", "--config", str(baseline_config), "--output", str(baseline_output)]
    )

    candidate_config = _write_offline_experiment_config(tmp_path, dataset_path, name="candidate", seed=7)
    candidate_output = tmp_path / "candidate_result.json"
    result = CliRunner().invoke(
        cli.app,
        [
            "run-experiment",
            "--config",
            str(candidate_config),
            "--baseline-result",
            str(baseline_output),
            "--output",
            str(candidate_output),
        ],
    )

    assert result.exit_code == 2
    payload = json.loads(candidate_output.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "seed" in payload["error"]


def test_run_experiment_help_exposes_options() -> None:
    result = CliRunner().invoke(cli.app, ["run-experiment", "--help"], color=False)
    help_output = unstyle(result.output)
    assert "--config" in help_output
    assert "--output" in help_output
    assert "--baseline-result" in help_output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/cli/test_cli.py -v -k run_experiment`
Expected: FAIL — `run-experiment` 명령이 아직 없어 typer가 "No such command" 오류를 낸다.

- [ ] **Step 3: Add the CLI command**

`autoresearch/cli.py`의 import 블록(파일 상단, 기존
`from autoresearch.model_evaluation import (...)` 구문)에 `offline_experiment`를
추가한다 — `offline_experiment.py`는 `autoresearch/model_evaluation/`에 있으므로
`model_training` 블록이 아니라 이 블록에 더한다:

```python
from autoresearch.model_training import (  # noqa: E402
    build_training_dataset,
    train,
)
from autoresearch.model_evaluation import (  # noqa: E402
    degradation_eval,
    evaluate,
    offline_experiment,
    paired_experiment,
    training_comparison,
)
```

파일 최상단 모듈 docstring의 `[파이프라인]` 절 명령 목록에 `run-experiment`를 추가한다:

```python
"""LightGBM 학습 파이프라인 Typer CLI.

[파이프라인] 피처 조립 → 학습 → 평가 → comparison → champion 승격 구간의
진입점(배선)을 담당한다: `python -m autoresearch.cli create-experiment-plan /
build-features / train-model / evaluate-model / run-pipeline / verify-comparison /
compare-paired-experiment / promote-model / sweep-seeds / measure-degradation /
run-experiment`.
```
(이하 기존 문단 유지, `run-experiment`는 Feast/MLflow/ONNX/승격 경로와 분리된
오프라인 실험 전용 명령이라는 문장 한 줄만 덧붙인다.)

`compare_paired_experiment` 명령 함수 뒤, `_experiment_client_module` 앞에 새 명령을 추가한다:

```python
@app.command("run-experiment")
def run_experiment_command(
    config: Path = typer.Option(..., "--config", help="ExperimentConfig YAML 경로"),
    output: Path = typer.Option(..., "--output", help="ExperimentResult JSON을 저장할 경로"),
    baseline_result: Optional[Path] = typer.Option(
        None,
        "--baseline-result",
        help="비교할 baseline ExperimentResult JSON 경로. 주면 비교 조건이 전부 "
        "일치할 때만 metrics_delta를 계산한다(#offline-experiment).",
    ),
) -> None:
    """로컬 CSV로 feature 부분집합·모델 실험을 실행하고 결과를 JSON으로 남긴다.

    Feast/MLflow/ONNX/promotion 경로와 완전히 분리된 오프라인 실험 전용 명령이다
    (`autoresearch/model_evaluation/offline_experiment.py`). 이 경로의 산출물은 champion 승격과
    무관하다.

    exit code: 성공 0, config·baseline 비교조건 검증 실패 2, 그 외 예기치 못한
    실행 실패 1. 실패해도 `status: "failed"` 결과 JSON은 항상 `--output`에 남는다.
    """
    experiment_name = "unknown"
    try:
        parsed_config = offline_experiment.load_experiment_config(config)
        experiment_name = parsed_config.experiment.name
        baseline = (
            offline_experiment.load_experiment_result(baseline_result)
            if baseline_result is not None
            else None
        )
        result = offline_experiment.run_experiment(parsed_config, project_root=Path(PROJECT_ROOT))
        if baseline is not None:
            result = offline_experiment.apply_baseline_comparison(result, baseline)
    except offline_experiment.OfflineExperimentError as error:
        failed_result = offline_experiment.ExperimentResult(
            experiment_name=experiment_name, status="failed", error=str(error)
        )
        offline_experiment.write_result(failed_result, output)
        typer.echo(f"[실험 실패] {error}", err=True)
        raise typer.Exit(code=2) from error
    except Exception as error:
        failed_result = offline_experiment.ExperimentResult(
            experiment_name=experiment_name,
            status="failed",
            error=f"{type(error).__name__}: 실행 중 예기치 못한 오류가 발생했습니다.",
        )
        offline_experiment.write_result(failed_result, output)
        typer.echo(f"[실험 실패] {type(error).__name__}: {error}", err=True)
        raise typer.Exit(code=1) from error

    offline_experiment.write_result(result, output)
    typer.echo(result.model_dump_json())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/cli/test_cli.py -v -k run_experiment`
Expected: PASS (6개)

전체 회귀도 확인한다:

Run: `uv run python -m pytest -v`
Expected: PASS (기존 테스트 전부 + 신규 테스트, 총 실패 0)

- [ ] **Step 5: Commit**

```bash
git add autoresearch/cli.py tests/cli/test_cli.py
git commit -m "feat: run-experiment CLI 명령을 추가한다"
```

---

## Task 5: 검증용 예제 config + 문서 인덱스 갱신

**Files:**
- Create: `examples/offline_experiments/baseline.yaml`
- Create: `examples/offline_experiments/ablation_remove_topic_similarity.yaml`
- Modify: `docs/README.md` (🏋️ 학습 파이프라인 절에 이 spec 링크 1줄 추가)

**Interfaces:**
- Consumes: 없음(설정 파일). Task 6에서 이 두 파일을 CLI로 직접 실행한다.

- [ ] **Step 1: baseline.yaml 작성**

`autoresearch/feature_engineering/model_contract.py`의 `MODEL_FEATURE_COLUMNS` 21개를 그대로 나열한다.

```yaml
# examples/offline_experiments/baseline.yaml
experiment:
  name: baseline_full_features
  seed: 42

model:
  type: lightgbm
  params: {}

data:
  path: data/processed/training_dataset.csv

features:
  - age_group
  - occupation
  - watch_time_band
  - recent_click_count_7d
  - recent_view_count_7d
  - recent_watch_time_7d
  - recent_like_count_7d
  - historical_category_affinity
  - total_event_count_7d
  - category_id
  - duration_sec
  - view_count
  - like_ratio
  - comment_ratio
  - days_since_upload
  - channel_subscriber_count
  - channel_view_count
  - channel_video_count
  - topic_similarity
  - preferred_category_match
  - historical_category_match

evaluation:
  primary_metric: roc_auc
```

- [ ] **Step 2: ablation_remove_topic_similarity.yaml 작성**

`baseline.yaml`과 동일하되 `topic_similarity`만 제외한 20개, `experiment.name`만 다르게.

```yaml
# examples/offline_experiments/ablation_remove_topic_similarity.yaml
experiment:
  name: ablation_remove_topic_similarity
  seed: 42

model:
  type: lightgbm
  params: {}

data:
  path: data/processed/training_dataset.csv

features:
  - age_group
  - occupation
  - watch_time_band
  - recent_click_count_7d
  - recent_view_count_7d
  - recent_watch_time_7d
  - recent_like_count_7d
  - historical_category_affinity
  - total_event_count_7d
  - category_id
  - duration_sec
  - view_count
  - like_ratio
  - comment_ratio
  - days_since_upload
  - channel_subscriber_count
  - channel_view_count
  - channel_video_count
  - preferred_category_match
  - historical_category_match

evaluation:
  primary_metric: roc_auc
```

- [ ] **Step 3: docs/README.md 갱신**

`## 🏋️ 학습 파이프라인 (Training)` 절, `training-experiment-provenance.md` 가이드
링크 다음 줄에 추가:

```markdown
- [Spec — Auto Research용 오프라인 실험 경로](specs/2026-09-14-auto-research-offline-experiment-path.md) —
  feature 부분집합 선택·baseline diff를 Feast/MLflow 없이 로컬 CLI로 반복하는 경로 계약
```

`## 유효한 Spec (살아있는 계약)` 목록에도 같은 문서를 한 줄 추가한다(다른 항목과
같은 형식으로).

- [ ] **Step 4: 파일 생성 확인**

Run: `uv run python -c "import yaml, pathlib; [yaml.safe_load(p.read_text(encoding='utf-8')) for p in pathlib.Path('examples/offline_experiments').glob('*.yaml')]; print('OK')"`
Expected: `OK` 출력(두 YAML이 유효하게 파싱됨)

- [ ] **Step 5: Commit**

```bash
git add examples/offline_experiments/baseline.yaml \
  examples/offline_experiments/ablation_remove_topic_similarity.yaml \
  docs/README.md
git commit -m "docs: offline experiment 예제 config와 문서 인덱스를 추가한다"
```

---

## Task 6: 실제 로컬 데이터로 검증 실행 (Experiment A/B)

**Files:** 없음(코드 변경 없음 — 실행 검증만). 결과 파일은 `artifacts/experiments/`
아래(git 미추적 산출물 디렉토리, 필요 시 `.gitignore`에 이미 걸린 `artifacts/`
관례를 따른다)에 남긴다.

**Interfaces:** Task 1~5에서 만든 `run-experiment` CLI를 그대로 실행한다.

- [ ] **Step 1: Experiment A(baseline) 실행**

Run:
```bash
uv run python -m autoresearch.cli run-experiment \
  --config examples/offline_experiments/baseline.yaml \
  --output artifacts/experiments/baseline.json
```
Expected: exit code 0, `artifacts/experiments/baseline.json`에 `status: "success"`와
`metrics.roc_auc`가 사람이 보기에 말이 되는 범위(0.5 초과, 1.0 미만)로 출력됨.

- [ ] **Step 2: Experiment B(ablation) 실행 — baseline과 diff**

Run:
```bash
uv run python -m autoresearch.cli run-experiment \
  --config examples/offline_experiments/ablation_remove_topic_similarity.yaml \
  --baseline-result artifacts/experiments/baseline.json \
  --output artifacts/experiments/ablation_remove_topic_similarity.json
```
Expected: exit code 0, 결과 JSON에 `baseline.experiment_name == "baseline_full_features"`와
`metrics_delta.roc_auc`가 채워짐(음수여도 정상 — `topic_similarity`는 라벨 생성과
직결된 핵심 신호이므로 제거 시 성능이 떨어지는 것이 기대되는 결과다).

- [ ] **Step 3: 사용자에게 두 결과 JSON 전문과 metrics_delta를 보여준다**

두 `artifacts/experiments/*.json` 파일 내용을 그대로 사용자에게 제시해, Feature 변경 →
Dataset 소비 → Train → Evaluate → 결과 출력 → baseline 비교 전 구간이 로컬에서
Feast/MLflow/GCS 없이 실제로 동작함을 확인시킨다.

---

## Self-Review 체크리스트 (계획 작성자가 직접 확인)

- **spec 커버리지**: split/preprocessing semantics 9단계 표(spec) → Task 2에서
  각 헬퍼 함수로 구현·테스트됨. `comparison_conditions`+`model_params`(spec 최신
  추가분) → Task 1(모델)·Task 2(계산)·Task 3(비교)에 반영됨. 실패 시 non-zero
  exit code → Task 4에서 exit code 1/2 분리 테스트로 커버됨. baseline diff
  fail-fast → Task 3. 두 검증 실험 → Task 6.
- **플레이스홀더 스캔**: 없음 — 모든 스텝에 실행 가능한 실제 코드/커맨드 포함.
- **타입 일관성**: `ExperimentConfig`/`ExperimentResult`/`ComparisonConditions`/
  `BaselineSummary` 필드명이 Task 1~4 전체에서 동일하게 쓰임(`features`,
  `comparison_conditions`, `metrics_delta` 등 재명명 없음).
