"""Auto Research용 로컬 오프라인 feature/model 실험 경로.

[파이프라인] Training Dataset(CSV, 이미 존재) → feature 부분집합 선택 → 학습 →
held-out 평가 → baseline 대비 diff 구간을 담당한다. `build-features`(Feast PIT
조립)과 prod 승격 경로(`train.py`/`evaluate.py`/`paired_experiment.py`)의
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
