"""Shared data models for Car Design Space Explorer."""

from .data_models import (
    TrainingConfig,
    ModelArtifact,
    KPIPrediction,
    SurfacePrediction,
    GeometryMetrics,
    StructuralResult,
    CostParameters,
    CostResult,
    DesignVariantResult,
    SweepParameters,
    SweepResults,
    OrchestratorResponse,
)
from .a2a import A2AMessage, A2AResponse, A2AError

__all__ = [
    "TrainingConfig",
    "ModelArtifact",
    "KPIPrediction",
    "SurfacePrediction",
    "GeometryMetrics",
    "StructuralResult",
    "CostParameters",
    "CostResult",
    "DesignVariantResult",
    "SweepParameters",
    "SweepResults",
    "OrchestratorResponse",
    "A2AMessage",
    "A2AResponse",
    "A2AError",
]
