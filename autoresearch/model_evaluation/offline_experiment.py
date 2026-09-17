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
import math
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from autoresearch.feature_engineering.model_contract import CATEGORICAL_FEATURE_COLUMNS
from autoresearch.model_training.downsampling import apply_downsampling_calibration, downsample_negatives
from autoresearch.model_training.model_factory import MODEL_REGISTRY
from autoresearch.model_training.train import (
    collect_categorical_categories,
    compute_auto_scale_pos_weight,
    load_config,
    require_binary_labels,
)
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
