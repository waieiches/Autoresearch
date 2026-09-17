"""autoresearch/model_training/model_factory.py의 CTRModel 레지스트리 테스트."""
from autoresearch.model_training.base import CTRModel
from autoresearch.model_training.lgbm_model import LGBMModel
from autoresearch.model_training.model_factory import MODEL_REGISTRY


def test_lightgbm_registered_as_ctrmodel_subclass() -> None:
    assert MODEL_REGISTRY["lightgbm"] is LGBMModel
    assert issubclass(MODEL_REGISTRY["lightgbm"], CTRModel)
