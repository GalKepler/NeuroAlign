"""
Data preprocessing module for NeuroAlign.

Provides tools for preparing neuroimaging data for modeling.

Main components:
- FeatureStore: Storage and retrieval for regional brain features
- DataPreparationPipeline: End-to-end pipeline for data preparation

Example:
    >>> from neuroalign.data.preprocessing import FeatureStore
    >>> store = FeatureStore("data/processed")
    >>> gm = store.load_feature("gm_volume")
    >>> multi = store.load_features(["gm_volume", "ct_thickness"])
"""

from .config import DataPaths, ModalityConfig, OutputConfig, PipelineConfig
from .feature_store import (
    ANATOMICAL_METRICS,
    FeatureInfo,
    FeatureStore,
    LongFormatInfo,
    StoreManifest,
)
from .pipeline import DataPreparationPipeline, PipelineResult
from .transformers import AnatomicalWideTransformer, DiffusionWideTransformer

__all__ = [
    # Config
    "PipelineConfig",
    "DataPaths",
    "ModalityConfig",
    "OutputConfig",
    # Feature Store
    "FeatureStore",
    "FeatureInfo",
    "LongFormatInfo",
    "StoreManifest",
    "ANATOMICAL_METRICS",
    # Pipeline
    "DataPreparationPipeline",
    "PipelineResult",
    # Transformers (for backwards compatibility)
    "AnatomicalWideTransformer",
    "DiffusionWideTransformer",
]
