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
