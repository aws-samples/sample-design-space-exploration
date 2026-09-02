"""Shared Pydantic data models for Car Design Space Explorer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Configuration for MLSimKit surrogate model training."""

    architecture: str = "MeshGraphNet"
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 0.001
    device: str = "cuda"


class ModelArtifact(BaseModel):
    """Reference to a trained model checkpoint."""

    model_path: str
    model_type: str  # "kpi" or "surface_variable"
    training_metrics: dict = Field(default_factory=dict)
    timestamp: str = ""


class KPIPrediction(BaseModel):
    """Aerodynamic KPI prediction result for a single variant."""

    variant_id: str
    drag_coefficient: float  # Cd
    side_force_coefficient: float  # Cs
    lift_coefficient: float  # Cl
    yaw_moment_coefficient: float  # Cmy
    inference_time_ms: float = 0.0


class SurfacePrediction(BaseModel):
    """Surface variable prediction (cpavg/cfxavg) mapped onto mesh."""

    variant_id: str
    cpavg_field: list[float] = Field(default_factory=list)
    cfxavg_field: list[float] = Field(default_factory=list)
    mesh_vertices: list[list[float]] = Field(default_factory=list)
    mesh_faces: list[list[int]] = Field(default_factory=list)
    vtk_file_path: str = ""
    png_heatmap_path: str = ""


class GeometryMetrics(BaseModel):
    """Structural and cost-relevant metrics computed from mesh geometry."""

    surface_area_m2: float
    vertex_count: int
    curvature_variation: float
    surface_patch_count: int
    max_draw_depth_mm: float
    has_undercuts: bool


class StructuralResult(BaseModel):
    """Structural feasibility evaluation result."""

    variant_id: str
    weight_kg: float
    stiffness_score: float  # 0.0 - 1.0
    recommended_thickness_mm: float
    feasibility_score: float  # 0.0 - 1.0
    is_feasible: bool
    constraint_violations: list[str] = Field(default_factory=list)
    status: str = "success"
    error_message: str | None = None


class CostParameters(BaseModel):
    """Manufacturing cost parameters retrieved from DynamoDB or MCP."""

    material_cost_per_kg: dict[str, float] = Field(default_factory=dict)
    stamping_cost_per_op: float = 150.0
    welding_cost_per_meter: float = 12.0
    tooling_base_cost: float = 50000.0
    complexity_multipliers: dict[str, float] = Field(
        default_factory=lambda: {
            "low_complexity": 1.0,
            "medium_complexity": 1.3,
            "high_complexity": 1.8,
        }
    )


class CostResult(BaseModel):
    """Manufacturing cost estimation result."""

    variant_id: str
    total_cost: float
    material_cost: float
    stamping_cost: float
    tooling_cost: float
    assembly_cost: float
    complexity_score: float
    status: str = "success"
    error_message: str | None = None


class DesignVariantResult(BaseModel):
    """Combined evaluation result for a single design variant."""

    variant_id: str
    kpi: KPIPrediction | None = None
    structural: StructuralResult | None = None
    cost: CostResult | None = None
    geometry_url: str = ""


class SweepParameters(BaseModel):
    """Parameter sweep configuration."""

    parameter_ranges: dict[str, tuple[float, float, int]] = Field(default_factory=dict)
    objectives: list[str] = Field(default_factory=lambda: ["aero", "structural", "cost"])


class SweepResults(BaseModel):
    """Results from a parameter sweep evaluation."""

    variants: list[DesignVariantResult] = Field(default_factory=list)
    pareto_front: list[DesignVariantResult] = Field(default_factory=list)
    top_by_aero: list[DesignVariantResult] = Field(default_factory=list)
    top_by_cost: list[DesignVariantResult] = Field(default_factory=list)
    top_by_structural: list[DesignVariantResult] = Field(default_factory=list)


class OrchestratorResponse(BaseModel):
    """Unified response from the Orchestrator Agent."""

    text_response: str
    variants: list[DesignVariantResult] = Field(default_factory=list)
    pareto_front: list[DesignVariantResult] | None = None
    recommended_variant: DesignVariantResult | None = None
    visualization_data: dict | None = None
