"""Feature engineering pipeline for ML-driven signal generation."""

from .base_feature import FeatureGenerator, FeatureResult
from .pipeline import FeaturePipeline, PipelineResult
from .statistical import (
    HigherMomentFeature,
    HurstExponentFeature,
    ReturnFeature,
    VolatilityFeature,
    ZScoreFeature,
)
from .technical import (
    ATRFeature,
    BollingerBandFeature,
    MACDFeature,
    RSIFeature,
    SMAFeature,
)

__all__ = [
    "FeatureGenerator",
    "FeatureResult",
    "FeaturePipeline",
    "PipelineResult",
    "SMAFeature",
    "RSIFeature",
    "MACDFeature",
    "BollingerBandFeature",
    "ATRFeature",
    "ReturnFeature",
    "ZScoreFeature",
    "HigherMomentFeature",
    "HurstExponentFeature",
    "VolatilityFeature",
]
