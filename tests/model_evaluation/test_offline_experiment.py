"""autoresearch/model_evaluation/offline_experiment.py의 config/result 모델·로더 테스트."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from autoresearch.feature_engineering.model_contract import CATEGORICAL_FEATURE_COLUMNS
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


def test_load_experiment_config_rejects_unknown_model_params_key(tmp_path) -> None:
    """오타(learnign_rate)가 조용히 무시되지 않고 config_error로 드러나야 한다."""
    payload = _valid_config_dict()
    payload["model"]["params"] = {"learnign_rate": 0.1}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(oe.OfflineExperimentError) as excinfo:
        oe.load_experiment_config(config_path)
    assert excinfo.value.reason_code == "config_error"


@pytest.mark.parametrize("bad_test_size", [0, -0.1, 1.5])
def test_load_experiment_config_rejects_out_of_range_test_size(tmp_path, bad_test_size) -> None:
    payload = _valid_config_dict()
    payload["data"]["test_size"] = bad_test_size
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(oe.OfflineExperimentError) as excinfo:
        oe.load_experiment_config(config_path)
    assert excinfo.value.reason_code == "config_error"


@pytest.mark.parametrize("bad_val_size", [0, -0.1, 1.5])
def test_load_experiment_config_rejects_out_of_range_val_size(tmp_path, bad_val_size) -> None:
    payload = _valid_config_dict()
    payload["data"]["val_size"] = bad_val_size
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(oe.OfflineExperimentError) as excinfo:
        oe.load_experiment_config(config_path)
    assert excinfo.value.reason_code == "config_error"


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


def _synthetic_dataset(n: int = 300, seed: int = 7) -> pd.DataFrame:
    """CATEGORICAL_FEATURE_COLUMNS의 age_group 포함, 수치 피처 2개 + 라벨."""
    assert "age_group" in CATEGORICAL_FEATURE_COLUMNS
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

    result = oe.run_experiment(config, project_root=Path(__file__).resolve().parents[2])

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

    result = oe.run_experiment(config, project_root=Path(__file__).resolve().parents[2])

    assert result.status == "success"
    assert "age_group" not in result.features


def test_run_experiment_missing_feature_column_raises_config_error(tmp_path) -> None:
    dataset = _synthetic_dataset()
    dataset_path = tmp_path / "training_dataset.csv"
    dataset.to_csv(dataset_path, index=False)
    config_path = _write_config(tmp_path, dataset_path, features=["age_group", "not_a_real_column"])
    config = oe.load_experiment_config(config_path)

    with pytest.raises(oe.OfflineExperimentError) as excinfo:
        oe.run_experiment(config, project_root=Path(__file__).resolve().parents[2])
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

    result = oe.run_experiment(config, project_root=Path(__file__).resolve().parents[2])

    assert result.status == "success"
    assert result.comparison_conditions.sampling_rate == 0.3
    assert result.comparison_conditions.model_params["scale_pos_weight"] == 1.0


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


def test_apply_baseline_comparison_rejects_non_success_candidate() -> None:
    baseline = _make_result()
    candidate = _make_result(status="failed")

    with pytest.raises(oe.OfflineExperimentError) as excinfo:
        oe.apply_baseline_comparison(candidate, baseline)
    assert excinfo.value.reason_code == "comparison_error"


def test_apply_baseline_comparison_rejects_baseline_with_missing_metrics() -> None:
    baseline = oe.ExperimentResult(
        experiment_name="baseline",
        status="success",
        model="lightgbm",
        features=["age_group"],
        metrics=None,  # metrics is None despite status="success"
        comparison_conditions=oe.ComparisonConditions(
            data_path="data/processed/training_dataset.csv",
            row_count=100,
            seed=42,
            test_size=0.2,
            val_size=0.2,
            sampling_rate=1.0,
            model_type="lightgbm",
            model_params={"n_estimators": 200.0, "learning_rate": 0.05, "num_leaves": 31.0, "scale_pos_weight": 1.0},
        ),
    )
    candidate = _make_result()

    with pytest.raises(oe.OfflineExperimentError) as excinfo:
        oe.apply_baseline_comparison(candidate, baseline)
    assert excinfo.value.reason_code == "comparison_error"


def test_apply_baseline_comparison_rejects_candidate_with_missing_metrics() -> None:
    baseline = _make_result()
    candidate = oe.ExperimentResult(
        experiment_name="candidate",
        status="success",
        model="lightgbm",
        features=["age_group"],
        metrics=None,  # metrics is None despite status="success"
        comparison_conditions=oe.ComparisonConditions(
            data_path="data/processed/training_dataset.csv",
            row_count=100,
            seed=42,
            test_size=0.2,
            val_size=0.2,
            sampling_rate=1.0,
            model_type="lightgbm",
            model_params={"n_estimators": 200.0, "learning_rate": 0.05, "num_leaves": 31.0, "scale_pos_weight": 1.0},
        ),
    )

    with pytest.raises(oe.OfflineExperimentError) as excinfo:
        oe.apply_baseline_comparison(candidate, baseline)
    assert excinfo.value.reason_code == "comparison_error"


def test_apply_baseline_comparison_rejects_baseline_missing_metric_key() -> None:
    baseline = _make_result()
    baseline.metrics = {"roc_auc": 0.7, "pr_auc": 0.1}  # missing log_loss
    candidate = _make_result()

    with pytest.raises(oe.OfflineExperimentError) as excinfo:
        oe.apply_baseline_comparison(candidate, baseline)
    assert excinfo.value.reason_code == "comparison_error"
