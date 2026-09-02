"""Pure-geometry helpers for cross-section profile lofting.

No AWS, boto3, or agent framework imports — only numpy and trimesh.
This keeps the module fast to import and easy to test.
"""
from __future__ import annotations

import logging

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


def _car_profile(
    hw: float,
    cabin_hw: float,
    z_lo: float,
    z_shoulder: float,
    z_hi: float,
    n_corner: float = 3.0,
    n_pts: int = 48,
) -> np.ndarray:
    """Return a closed Y-Z cross-section profile for one X station.

    Traces four arcs clockwise (when viewed from front, i.e. from +x):
      A  underbody  — z_lo,      −hw  → +hw
      B  right side — z_lo→z_hi, +hw  (door panel + tumblehome)
      C  roof arc   — z_hi,      +cabin_hw → −cabin_hw
      D  left side  — z_hi→z_lo, −hw  (tumblehome + door panel, mirror of B)

    Args:
        hw:        sill half-width (Y extent at z_shoulder)
        cabin_hw:  cabin half-width above z_shoulder (tumblehome target)
        z_lo:      bottom of profile (ride_height)
        z_shoulder: top of lower-body slab / character line
        z_hi:      highest point of profile (roof top or hood top)
        n_corner:  superellipse exponent controlling sill-corner roundness
                   (2.0 = smooth oval, 4.0 = hard squared shoulder)
        n_pts:     total number of profile points; must be divisible by 4

    Returns:
        np.ndarray of shape (n_pts, 2) — columns are [y, z].
    """
    if n_pts % 4 != 0:
        raise ValueError(f"n_pts must be divisible by 4, got {n_pts}")

    q = n_pts // 4
    pts: list[list[float]] = []

    # ── A: underbody — z_lo, slight parabolic camber (5 mm at centre) ──────
    for i in range(q):
        t = i / q                             # 0 → 1 (exclusive of right corner)
        y = -hw + t * 2.0 * hw               # −hw → +hw
        z = z_lo + (1.0 - (2.0 * t - 1.0) ** 2) * 0.005
        pts.append([y, z])

    # ── B: right side — door panel then tumblehome ──────────────────────────
    # n_corner controls the blend shape near the character line (sill shoulder).
    # A higher n_corner sharpens the transition at z_shoulder (more squared corner).
    for i in range(q):
        t = i / q
        if t < 0.5:
            # Lower door panel: z_lo → z_shoulder at y ≈ hw (3 mm bulge)
            s = t * 2.0
            # Apply n_corner-based shaping: higher n_corner = more linear near sill
            shaped_s = s ** (2.0 / n_corner)
            y = hw + np.sin(np.pi * shaped_s) * 0.003
            z = z_lo + s * (z_shoulder - z_lo)
        else:
            # Tumblehome: z_shoulder → z_hi, hw → cabin_hw (cosine blend)
            s = (t - 0.5) * 2.0
            blend = (1.0 - np.cos(np.pi * s)) / 2.0
            y = hw + (cabin_hw - hw) * blend
            z = z_shoulder + s * (z_hi - z_shoulder)
        pts.append([y, z])

    # ── C: roof arc — z_hi, +cabin_hw → −cabin_hw (12 % of cabin height) ────
    # Scale by cabin height so the arc vanishes when z_hi ≈ z_shoulder.
    roof_camber = (z_hi - z_shoulder) * 0.12
    for i in range(q):
        t = i / q
        y = cabin_hw * (1.0 - 2.0 * t)
        z = z_hi + roof_camber * (1.0 - (2.0 * t - 1.0) ** 2)
        pts.append([y, z])

    # ── D: left side — mirror of B (tumblehome then door panel) ────────────
    for i in range(q):
        t = i / q
        if t < 0.5:
            # Tumblehome: z_hi → z_shoulder, −cabin_hw → −hw
            s = t * 2.0
            blend = (1.0 - np.cos(np.pi * s)) / 2.0
            y = -cabin_hw + (-hw + cabin_hw) * blend
            z = z_hi + (z_shoulder - z_hi) * s
        else:
            # Lower door panel: z_shoulder → z_lo at y ≈ −hw
            s = (t - 0.5) * 2.0
            shaped_s = s ** (2.0 / n_corner)
            y = -hw - np.sin(np.pi * shaped_s) * 0.003
            z = z_shoulder + s * (z_lo - z_shoulder)
        pts.append([y, z])

    return np.array(pts, dtype=float)


def _loft_body(stations: list[tuple[float, np.ndarray]]) -> trimesh.Trimesh:
    """Create a watertight body mesh by lofting between ordered cross-section stations.

    Each station is a (x_position, profile_yz) tuple where profile_yz is
    shape (N, 2) from _car_profile.  All stations must share the same N.

    The result has:
      - a quad-strip side surface between every adjacent station pair
      - a triangle-fan nose cap (station 0)
      - a triangle-fan tail cap (station -1)

    Returns a watertight trimesh.Trimesh with normals fixed.
    """
    if len(stations) < 2:
        raise ValueError("Need at least 2 stations to loft a body")

    n_pts = stations[0][1].shape[0]
    verts: list[list[float]] = []
    for x, profile in stations:
        for y, z in profile:
            verts.append([float(x), float(y), float(z)])

    faces: list[list[int]] = []
    n_st = len(stations)

    # ── Side surface: quad strip between adjacent stations ───────────────────
    for si in range(n_st - 1):
        b0 = si * n_pts
        b1 = (si + 1) * n_pts
        for j in range(n_pts):
            j1 = (j + 1) % n_pts
            faces.append([b0 + j,  b0 + j1, b1 + j1])
            faces.append([b0 + j,  b1 + j1, b1 + j])

    # ── Nose cap: fan from station-0 centroid ────────────────────────────────
    # Winding: [nose_idx, j1, j] — reversed relative to profile traversal order.
    # fix_normals() will orient both end caps consistently; the winding here
    # ensures a topologically closed shell for the repair pass.
    x0, prof0 = stations[0]
    nose_ctr = [float(x0), float(prof0[:, 0].mean()), float(prof0[:, 1].mean())]
    nose_idx = len(verts)
    verts.append(nose_ctr)
    for j in range(n_pts):
        j1 = (j + 1) % n_pts
        faces.append([nose_idx, j1, j])

    # ── Tail cap: fan from station-last centroid ─────────────────────────────
    xN, profN = stations[-1]
    tail_ctr = [float(xN), float(profN[:, 0].mean()), float(profN[:, 1].mean())]
    tail_idx = len(verts)
    verts.append(tail_ctr)
    base_last = (n_st - 1) * n_pts
    for j in range(n_pts):
        j1 = (j + 1) % n_pts
        faces.append([tail_idx, base_last + j1, base_last + j])

    mesh = trimesh.Trimesh(
        vertices=np.array(verts, dtype=float),
        faces=np.array(faces, dtype=np.int32),
        process=False,
    )
    mesh.fix_normals()
    # Safety net: fill any degenerate holes introduced by near-zero-length edges
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        mesh.fix_normals()
    return mesh


def _union_parts(parts: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    """Boolean union of all parts using manifold3d.

    Each part must overlap adjacent parts by at least 5 mm — this guarantees
    manifold3d always finds a non-degenerate intersection to merge.

    Falls back to trimesh.util.concatenate (visually correct, not watertight)
    if manifold3d raises for any reason, logging a WARNING.

    Args:
        parts: list of trimesh.Trimesh meshes to union. Must have at least one.

    Returns:
        Single trimesh.Trimesh — watertight if manifold3d succeeded.
    """
    if len(parts) == 1:
        return parts[0]

    try:
        import manifold3d as m3d

        manifolds = []
        for p in parts:
            m = m3d.Manifold(mesh=m3d.Mesh(
                vert_properties=np.asarray(p.vertices, dtype=np.float32),
                tri_verts=np.asarray(p.faces, dtype=np.uint32),
            ))
            manifolds.append(m)

        result = manifolds[0]
        for m in manifolds[1:]:
            result = result + m          # manifold3d operator+ is boolean union

        out = result.to_mesh()
        return trimesh.Trimesh(
            vertices=np.array(out.vert_properties, dtype=float),
            faces=np.array(out.tri_verts, dtype=np.int32),
        )

    except Exception as exc:
        logger.warning(
            "manifold3d union failed (%s) — falling back to concatenate. "
            "Result will NOT be watertight.", exc
        )
        result = trimesh.util.concatenate(parts)
        result.fix_normals()
        return result
