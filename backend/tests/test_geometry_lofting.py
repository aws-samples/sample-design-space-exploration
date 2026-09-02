"""Tests for backend/geometry/lofting.py — _car_profile, _loft_body, _union_parts."""
import json
import numpy as np
import pytest
import trimesh

from backend.geometry.lofting import _car_profile, _loft_body, _union_parts


# ---------------------------------------------------------------------------
# _car_profile
# ---------------------------------------------------------------------------

def test_car_profile_default_shape():
    pts = _car_profile(hw=0.20, cabin_hw=0.16, z_lo=0.05, z_shoulder=0.20, z_hi=0.35)
    assert pts.shape == (48, 2), "default n_pts=48, two columns (y, z)"


def test_car_profile_custom_n_pts():
    pts = _car_profile(hw=0.20, cabin_hw=0.16, z_lo=0.05, z_shoulder=0.20, z_hi=0.35,
                       n_pts=96)
    assert pts.shape == (96, 2)


def test_car_profile_n_pts_must_be_divisible_by_4():
    with pytest.raises((AssertionError, ValueError)):
        _car_profile(hw=0.20, cabin_hw=0.16, z_lo=0.05, z_shoulder=0.20, z_hi=0.35,
                     n_pts=50)


def test_car_profile_y_within_body_width():
    hw = 0.20
    pts = _car_profile(hw=hw, cabin_hw=0.16, z_lo=0.05, z_shoulder=0.20, z_hi=0.35)
    # Y must stay within hw ± 5 mm (small outward bulge allowed)
    assert pts[:, 0].max() <= hw + 0.006
    assert pts[:, 0].min() >= -hw - 0.006


def test_car_profile_z_within_height():
    z_lo, z_hi = 0.05, 0.35
    pts = _car_profile(hw=0.20, cabin_hw=0.16, z_lo=z_lo, z_shoulder=0.20, z_hi=z_hi)
    # Roof arc adds slight camber; Z must stay within z_hi + 15% of cabin height
    cabin_h = z_hi - 0.20
    assert pts[:, 1].min() >= z_lo - 1e-6
    assert pts[:, 1].max() <= z_hi + cabin_h * 0.16


def test_car_profile_hood_station_no_cabin():
    """When z_hi == z_shoulder + epsilon, profile has no visible greenhouse."""
    pts = _car_profile(hw=0.20, cabin_hw=0.18, z_lo=0.05, z_shoulder=0.20,
                       z_hi=0.21, n_pts=48)
    # All Z values should be at or near z_shoulder
    assert pts[:, 1].max() <= 0.21 + 0.01


def test_car_profile_n_corner_affects_shape():
    """Higher n_corner = squarer sill corners."""
    sharp = _car_profile(hw=0.20, cabin_hw=0.16, z_lo=0.05, z_shoulder=0.20,
                         z_hi=0.35, n_corner=4.0)
    smooth = _car_profile(hw=0.20, cabin_hw=0.16, z_lo=0.05, z_shoulder=0.20,
                          z_hi=0.35, n_corner=2.0)
    # The profiles should differ in the corner region
    assert not np.allclose(sharp, smooth, atol=1e-3)


# ---------------------------------------------------------------------------
# _loft_body
# ---------------------------------------------------------------------------

def _make_stations(n_pts=48):
    """Five stations that form a simple car-like lofted body for testing."""
    return [
        ( 0.52,  _car_profile(0.025, 0.025, 0.04, 0.04 + 0.10 * 0.28, 0.04 + 0.10 * 0.28, n_pts=n_pts)),
        ( 0.30,  _car_profile(0.185, 0.175, 0.04, 0.18, 0.18, n_pts=n_pts)),
        ( 0.10,  _car_profile(0.195, 0.145, 0.04, 0.18, 0.18 + 0.01, n_pts=n_pts)),
        (-0.10,  _car_profile(0.195, 0.145, 0.04, 0.18, 0.29, n_pts=n_pts)),
        (-0.35,  _car_profile(0.185, 0.125, 0.04, 0.18, 0.25, n_pts=n_pts)),
        (-0.52,  _car_profile(0.165, 0.110, 0.04, 0.16, 0.20, n_pts=n_pts)),
    ]


def test_loft_body_returns_trimesh():
    mesh = _loft_body(_make_stations())
    assert isinstance(mesh, trimesh.Trimesh)


def test_loft_body_watertight():
    mesh = _loft_body(_make_stations())
    assert mesh.is_watertight, f"Expected watertight mesh, euler={mesh.euler_number}"


def test_loft_body_is_volume():
    mesh = _loft_body(_make_stations())
    assert mesh.is_volume


def test_loft_body_euler_number():
    mesh = _loft_body(_make_stations())
    assert mesh.euler_number == 2, f"Expected genus-0 (euler=2), got {mesh.euler_number}"


def test_loft_body_vertex_count():
    n_pts, n_stations = 48, 6
    mesh = _loft_body(_make_stations(n_pts=n_pts))
    # Exactly n_pts × n_stations profile verts + 2 centroid cap verts
    assert len(mesh.vertices) == n_pts * n_stations + 2


def test_loft_body_x_bounds():
    stns = _make_stations()
    mesh = _loft_body(stns)
    assert abs(mesh.bounds[1][0] - 0.52) < 0.01
    assert abs(mesh.bounds[0][0] - (-0.52)) < 0.01


def test_loft_body_requires_two_stations():
    with pytest.raises((ValueError, IndexError)):
        _loft_body([_make_stations()[0]])

# ---------------------------------------------------------------------------
# _union_parts
# ---------------------------------------------------------------------------

def test_union_parts_two_overlapping_boxes():
    b1 = trimesh.creation.box([0.2, 0.1, 0.05])
    b2 = trimesh.creation.box([0.2, 0.1, 0.05])
    b2.apply_translation([0.19, 0, 0])          # 1 mm overlap
    result = _union_parts([b1, b2])
    assert isinstance(result, trimesh.Trimesh)
    # Union volume must be less than sum of individual volumes
    assert result.volume < b1.volume + b2.volume


def test_union_parts_single_part_passthrough():
    b = trimesh.creation.box([0.2, 0.1, 0.05])
    result = _union_parts([b])
    assert isinstance(result, trimesh.Trimesh)


def test_union_parts_watertight_result():
    b1 = trimesh.creation.box([0.3, 0.2, 0.1])
    b2 = trimesh.creation.box([0.1, 0.2, 0.05])
    b2.apply_translation([0.25, 0, 0.025])      # sits on top with overlap
    result = _union_parts([b1, b2])
    assert result.is_watertight


def test_union_parts_fallback_on_bad_input(monkeypatch):
    """If manifold3d raises, falls back to trimesh.util.concatenate gracefully."""
    import backend.geometry.lofting as lofting_mod
    import manifold3d as m3d

    original = m3d.Manifold
    def _bad_manifold(*args, **kwargs):
        raise RuntimeError("simulated manifold failure")

    monkeypatch.setattr(m3d, "Manifold", _bad_manifold)
    b1 = trimesh.creation.box([0.2, 0.1, 0.05])
    b2 = trimesh.creation.box([0.2, 0.1, 0.05])
    b2.apply_translation([0.19, 0, 0])
    # Should not raise — falls back to concatenate
    result = _union_parts([b1, b2])
    assert isinstance(result, trimesh.Trimesh)


# ---------------------------------------------------------------------------
# _generate_parametric_car  (integration — imports geometry_agent helpers)
# ---------------------------------------------------------------------------
# NOTE: geometry_agent.py imports boto3/bedrock which won't be available in CI.
# These tests import only the pure function after patching the heavy modules.

import sys
import types
from unittest.mock import MagicMock

def _patch_heavy_imports():
    """Stub out AWS/agent modules so geometry_agent.py can be imported in tests.

    CRITICAL: strands must be a real module-like object with `tool` set to an
    identity decorator.  If `tool` is a MagicMock, the @tool decorator replaces
    every decorated function with a MagicMock — making them unreachable in tests.
    """
    import types as _types

    # strands: real module with identity `tool` decorator
    # Force-replace even if already present (may be partially imported)
    strands_mock = _types.ModuleType("strands")
    strands_mock.tool = lambda fn: fn          # identity — preserves real functions
    strands_mock.Agent = MagicMock()           # geometry_agent does: from strands import Agent, tool
    sys.modules["strands"] = strands_mock
    # Sub-modules also need stubs
    for sub in ("strands.agent", "strands.agent.agent",
                "strands.agent.conversation_manager",
                "strands.hooks", "strands.hooks.events",
                "strands.models",
                "strands.multiagent", "strands.multiagent.a2a"):
        sys.modules[sub] = MagicMock()

    # Everything else can be plain MagicMocks
    for mod in [
        "boto3", "botocore", "botocore.config",
        "bedrock_agentcore", "bedrock_agentcore.tools",
        "bedrock_agentcore.tools.code_interpreter_client",
        "a2a", "a2a.server", "a2a.server.tasks",
        "a2a.server.context", "a2a.types",
        "fastapi", "fastapi.responses",
        "uvicorn",
        "requests",
        "httpx",
        "starlette", "starlette.types",
    ]:
        sys.modules[mod] = MagicMock()

_patch_heavy_imports()

from backend.agents.geometry_agent import _generate_parametric_car  # noqa: E402


@pytest.mark.parametrize("segment", ["sedan", "sport", "suv", "hatchback", "mini_suv"])
def test_parametric_car_watertight(segment):
    mesh = _generate_parametric_car({"segment": segment})
    assert mesh.is_watertight, (
        f"{segment}: expected watertight mesh, euler={mesh.euler_number}, "
        f"faces={len(mesh.faces)}"
    )


@pytest.mark.parametrize("segment", ["sedan", "sport", "suv", "hatchback", "mini_suv"])
def test_parametric_car_is_volume(segment):
    mesh = _generate_parametric_car({"segment": segment})
    assert mesh.is_volume, f"{segment}: mesh.is_volume is False"


def test_suv_taller_than_sport():
    suv   = _generate_parametric_car({"segment": "suv"})
    sport = _generate_parametric_car({"segment": "sport"})
    suv_h   = suv.bounds[1][2]   - suv.bounds[0][2]
    sport_h = sport.bounds[1][2] - sport.bounds[0][2]
    assert suv_h > sport_h * 1.15, (
        f"SUV ({suv_h:.3f} m) should be at least 15% taller than sport ({sport_h:.3f} m)"
    )


def test_sedan_has_trunk_shelf():
    sedan = _generate_parametric_car({"segment": "sedan"})
    # Trunk shelf: rear 15% of car, some vertices above z_shoulder
    rear_x = sedan.bounds[0][0] + (sedan.bounds[1][0] - sedan.bounds[0][0]) * 0.15
    rear_verts = sedan.vertices[sedan.vertices[:, 0] <= rear_x]
    ride_h = 0.04
    z_shoulder_est = ride_h + (sedan.bounds[1][2] - ride_h) * 0.50
    assert np.any(rear_verts[:, 2] > z_shoulder_est), "Sedan should have a raised trunk shelf"


def test_sport_lower_nose_than_suv():
    sport = _generate_parametric_car({"segment": "sport"})
    suv   = _generate_parametric_car({"segment": "suv"})
    # Front 10% of car
    sport_front_z_max = sport.vertices[
        sport.vertices[:, 0] > sport.bounds[0][0] + (sport.bounds[1][0] - sport.bounds[0][0]) * 0.90
    ][:, 2].max()
    suv_front_z_max = suv.vertices[
        suv.vertices[:, 0] > suv.bounds[0][0] + (suv.bounds[1][0] - suv.bounds[0][0]) * 0.90
    ][:, 2].max()
    assert sport_front_z_max < suv_front_z_max, "Sport nose should be lower than SUV nose"


def test_rear_slant_increases_rear_drop():
    low  = _generate_parametric_car({"segment": "sedan", "rear_slant": 5})
    high = _generate_parametric_car({"segment": "sedan", "rear_slant": 30})
    # Higher rear_slant → lower rear roof
    low_rear_top  = low.vertices[low.vertices[:, 0] < 0][:, 2].max()
    high_rear_top = high.vertices[high.vertices[:, 0] < 0][:, 2].max()
    assert high_rear_top < low_rear_top, "Higher rear_slant should lower the rear roofline"


def test_simulation_quality_more_faces():
    viewer = _generate_parametric_car({"segment": "sedan", "quality": "viewer"})
    sim    = _generate_parametric_car({"segment": "sedan", "quality": "simulation"})
    assert len(sim.faces) > len(viewer.faces) * 1.5, (
        "simulation quality should produce significantly more faces"
    )

# ---------------------------------------------------------------------------
# generate_car_design tool response fields
# ---------------------------------------------------------------------------

def test_generate_car_design_response_fields(monkeypatch, tmp_path):
    """generate_car_design returns simulation_ready and simulation_warning fields."""
    import backend.agents.geometry_agent as ga

    # Stub S3 upload
    monkeypatch.setattr(ga, "_upload_mesh_to_s3", lambda mesh, vid: f"s3://bucket/{vid}.stl")
    monkeypatch.setattr(ga, "GEOMETRY_S3_BUCKET", "bucket")

    result = json.loads(ga.generate_car_design(json.dumps({"segment": "sedan"})))
    assert result["status"] == "success"
    assert "simulation_ready" in result
    assert "simulation_warning" in result
    assert "is_watertight" in result
    assert result["simulation_warning"] is False      # sedan is in-distribution


def test_generate_car_design_simulation_warning_for_sport(monkeypatch):
    import backend.agents.geometry_agent as ga
    monkeypatch.setattr(ga, "_upload_mesh_to_s3", lambda mesh, vid: f"s3://bucket/{vid}.stl")
    monkeypatch.setattr(ga, "GEOMETRY_S3_BUCKET", "bucket")

    result = json.loads(ga.generate_car_design(json.dumps({"segment": "sport"})))
    assert result["simulation_warning"] is True       # sport extrapolates


def test_generate_car_design_simulation_warning_for_suv(monkeypatch):
    import backend.agents.geometry_agent as ga
    monkeypatch.setattr(ga, "_upload_mesh_to_s3", lambda mesh, vid: f"s3://bucket/{vid}.stl")
    monkeypatch.setattr(ga, "GEOMETRY_S3_BUCKET", "bucket")

    result = json.loads(ga.generate_car_design(json.dumps({"segment": "suv"})))
    assert result["simulation_warning"] is True       # suv extrapolates


def test_generate_car_design_simulation_warning_for_hatchback(monkeypatch):
    import backend.agents.geometry_agent as ga
    monkeypatch.setattr(ga, "_upload_mesh_to_s3", lambda mesh, vid: f"s3://bucket/{vid}.stl")
    monkeypatch.setattr(ga, "GEOMETRY_S3_BUCKET", "bucket")

    result = json.loads(ga.generate_car_design(json.dumps({"segment": "hatchback"})))
    assert result["simulation_warning"] is True       # hatchback extrapolates
