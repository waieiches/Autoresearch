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
