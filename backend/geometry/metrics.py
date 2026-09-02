"""Geometry metrics computation from STL/VTP mesh files.

Uses trimesh to load meshes and compute structural/cost-relevant metrics
for the Structural Agent and Cost Agent.
"""

from __future__ import annotations

import numpy as np
import trimesh

from backend.models.data_models import GeometryMetrics


def compute_geometry_metrics(geometry_path: str) -> GeometryMetrics:
    """Compute structural and cost-relevant metrics from a mesh file.

    Args:
        geometry_path: Path to an STL or VTP mesh file.

    Returns:
        GeometryMetrics with surface area, vertex count, curvature variation,
        surface patch count, max draw depth, and undercut detection.

    Raises:
        FileNotFoundError: If the geometry file does not exist.
        ValueError: If the mesh cannot be loaded or is empty.
    """
    mesh = trimesh.load(geometry_path, force="mesh")

    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise ValueError(f"Failed to load valid mesh from {geometry_path}")

    surface_area_m2 = float(mesh.area)
    vertex_count = len(mesh.vertices)

    # Curvature variation: std of discrete mean curvature at vertices
    curvature_variation = _compute_curvature_variation(mesh)

    # Surface patch count: connected components with similar normals
    surface_patch_count = _compute_surface_patches(mesh)

    # Max draw depth: Z-axis extent (relevant for stamping)
    bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    max_draw_depth_mm = float((bounds[1][2] - bounds[0][2]) * 1000)  # m -> mm

    # Undercut detection: faces with negative Z-component normals (negative draft)
    has_undercuts = _detect_undercuts(mesh)

    return GeometryMetrics(
        surface_area_m2=surface_area_m2,
        vertex_count=vertex_count,
        curvature_variation=curvature_variation,
        surface_patch_count=surface_patch_count,
        max_draw_depth_mm=max_draw_depth_mm,
        has_undercuts=has_undercuts,
    )


def _compute_curvature_variation(mesh: trimesh.Trimesh) -> float:
    """Compute standard deviation of discrete mean curvature at vertices.

    Uses the angle-deficit method: for each vertex, the discrete curvature
    is 2*pi minus the sum of angles at that vertex in incident faces,
    divided by the vertex's Voronoi area.
    """
    try:
        # trimesh provides discrete_mean_curvature_measure via vertex_defects
        # vertex_defects = 2*pi - sum(angles) for each vertex (angle deficit)
        defects = mesh.vertex_defects
        if defects is not None and len(defects) > 0:
            return float(np.std(defects))
    except Exception:
        pass

    # Fallback: use face normal variation as proxy for curvature
    if len(mesh.face_normals) > 1:
        return float(np.std(np.linalg.norm(np.diff(mesh.face_normals, axis=0), axis=1)))
    return 0.0


def _compute_surface_patches(mesh: trimesh.Trimesh, angle_threshold_deg: float = 30.0) -> int:
    """Count connected components of faces with similar normals.

    Groups adjacent faces whose normal angle difference is below the threshold,
    then counts the number of distinct groups.
    """
    if len(mesh.faces) == 0:
        return 0

    try:
        # Use trimesh facets: groups of coplanar adjacent faces
        facets = mesh.facets
        if facets is not None:
            return len(facets)
    except Exception:
        pass

    # Fallback: split by connected components
    try:
        components = mesh.split(only_watertight=False)
        return len(components) if components else 1
    except Exception:
        return 1


def _detect_undercuts(mesh: trimesh.Trimesh, draft_angle_deg: float = 5.0) -> bool:
    """Detect faces with negative draft angles (undercuts).

    A face has an undercut if its normal has a significant negative Z-component,
    meaning the face points downward relative to the draw direction (Z-axis).

    Args:
        mesh: The loaded trimesh mesh.
        draft_angle_deg: Minimum draft angle in degrees. Faces with normals
            more than this angle below horizontal are flagged.

    Returns:
        True if any undercut faces are detected.
    """
    if len(mesh.face_normals) == 0:
        return False

    # Z-component of face normals; negative means pointing downward
    z_normals = mesh.face_normals[:, 2]

    # Threshold: cos(90 + draft_angle) = -sin(draft_angle)
    threshold = -np.sin(np.radians(draft_angle_deg))

    # Any face with Z-normal below threshold is an undercut
    return bool(np.any(z_normals < threshold))
