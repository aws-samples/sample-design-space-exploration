#!/usr/bin/env python3
"""
Geometry Agent — Strands A2A Server for geometry modification and visualization.

Allows engineers to modify car body STL meshes via natural language
(e.g. "add side mirrors", "extend the bonnet"). Uses Stability AI Stable Diffusion 3.5 Large
for visual design preview generation and trimesh for actual 3D mesh
modifications. Uploads modified STL to S3 for downstream aero evaluation.

Architecture:
- Strands Agent framework with @tool decorated functions
- A2A Server for agent-to-agent communication
- Stability AI Stable Diffusion 3.5 Large (stability.sd3-5-large-v1:0) for image generation
- trimesh for 3D mesh operations (boolean unions, scaling, transforms)
- S3 for geometry storage
- Deployed to Bedrock AgentCore Runtime
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import tempfile
import time
import uuid

import boto3
import numpy as np
import trimesh
import uvicorn
from bedrock_agentcore.tools.code_interpreter_client import code_session
from botocore.config import Config
from fastapi import FastAPI
from strands import Agent, tool
from strands.agent.agent import ConcurrentInvocationMode
from strands.hooks.events import BeforeInvocationEvent
from strands.multiagent.a2a import A2AServer

# A2A TaskStore — strip history from responses to prevent payload bloat
from a2a.server.tasks import TaskStore, InMemoryTaskStore
from a2a.server.context import ServerCallContext
from a2a.types import Task as A2ATask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AGENT_PORT = int(os.environ.get("GEOMETRY_AGENT_PORT", "9000"))
MODEL_ID = os.environ.get("MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
IMAGE_MODEL_ID = os.environ.get("IMAGE_MODEL_ID", "stability.sd3-5-large-v1:0")
IMAGE_MODEL_REGION = os.environ.get("IMAGE_MODEL_REGION", "us-west-2")
GEOMETRY_S3_BUCKET = os.environ.get("GEOMETRY_S3_BUCKET", "")
GEOMETRY_S3_PREFIX = os.environ.get("GEOMETRY_S3_PREFIX", "geometries/")

if not GEOMETRY_S3_BUCKET:
    logger.warning(
        "GEOMETRY_S3_BUCKET is not configured. S3 upload/download operations "
        "will fail. Set the GEOMETRY_S3_BUCKET environment variable."
    )

# Bedrock Runtime client for image generation. Stable Diffusion 3.5 Large is
# currently served by Amazon Bedrock in us-west-2, independently of the agent region.
bedrock_runtime = boto3.client(
    "bedrock-runtime",
    region_name=IMAGE_MODEL_REGION,
    config=Config(read_timeout=300),
)
s3_client = boto3.client("s3", region_name=AWS_REGION)


# ---------------------------------------------------------------------------
# Helper: Load mesh from S3 or local path
# ---------------------------------------------------------------------------

def _load_mesh(geometry_path: str) -> trimesh.Trimesh:
    """Load an STL/VTP mesh from S3 URI or local path."""
    if geometry_path.startswith("s3://"):
        parts = geometry_path.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
            s3_client.download_file(bucket, key, tmp.name)
            tmp.flush()
            return trimesh.load(tmp.name, force="mesh")
    else:
        return trimesh.load(geometry_path, force="mesh")


def _upload_mesh_to_s3(mesh: trimesh.Trimesh, variant_id: str) -> str:
    """Export mesh as STL and upload to S3. Returns the S3 URI."""
    if not GEOMETRY_S3_BUCKET:
        raise ValueError(
            "GEOMETRY_S3_BUCKET is not configured. Set the GEOMETRY_S3_BUCKET "
            "environment variable to upload geometry files to S3."
        )
    s3_key = f"{GEOMETRY_S3_PREFIX}{variant_id}.stl"
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        mesh.export(tmp.name, file_type="stl")
        tmp.flush()
        s3_client.upload_file(tmp.name, GEOMETRY_S3_BUCKET, s3_key)
    s3_uri = f"s3://{GEOMETRY_S3_BUCKET}/{s3_key}"
    logger.info(f"Uploaded modified mesh to {s3_uri}")
    return s3_uri


def _render_mesh_to_png_bytes(mesh: trimesh.Trimesh) -> bytes:
    """Render a trimesh mesh to a PNG image using pyrender (offscreen).

    Falls back to a simple matplotlib projection if pyrender is unavailable.
    """
    try:
        import pyrender

        scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
        py_mesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(py_mesh)

        # Camera setup — look at mesh center
        bounds = mesh.bounds
        center = mesh.centroid
        extent = np.max(bounds[1] - bounds[0])
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
        cam_pose = np.eye(4)
        cam_pose[:3, 3] = center + [0, 0, extent * 1.8]
        scene.add(camera, pose=cam_pose)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=cam_pose)

        renderer = pyrender.OffscreenRenderer(1024, 768)
        color, _ = renderer.render(scene)
        renderer.delete()

        from PIL import Image
        img = Image.fromarray(color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    except Exception as e:
        logger.warning(f"pyrender unavailable ({e}), falling back to matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure(figsize=(12, 9), dpi=100)
        ax = fig.add_subplot(111, projection="3d")

        vertices = mesh.vertices
        faces = mesh.faces

        # Subsample faces for performance
        max_faces = 5000
        if len(faces) > max_faces:
            idx = np.random.choice(len(faces), max_faces, replace=False)
            faces_sub = faces[idx]
        else:
            faces_sub = faces

        polys = vertices[faces_sub]
        collection = Poly3DCollection(polys, alpha=0.7, edgecolor="gray", linewidth=0.1)
        collection.set_facecolor([0.6, 0.7, 0.85])
        ax.add_collection3d(collection)

        bounds = mesh.bounds
        ax.set_xlim(bounds[0][0], bounds[1][0])
        ax.set_ylim(bounds[0][1], bounds[1][1])
        ax.set_zlim(bounds[0][2], bounds[1][2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=25, azim=-60)
        ax.set_title("Car Body Geometry")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Bedrock image generation helper
# ---------------------------------------------------------------------------

def _decode_image_response(response_body: dict) -> bytes:
    """Validate a Stability AI response and decode its first PNG image."""
    if response_body.get("error"):
        raise RuntimeError(f"Image generation error: {response_body['error']}")

    finish_reasons = response_body.get("finish_reasons", [])
    failure_reasons = [reason for reason in finish_reasons if reason is not None]
    if failure_reasons:
        raise RuntimeError(f"Image generation failed: {failure_reasons[0]}")

    images = response_body.get("images") or []
    if not images:
        raise RuntimeError("Image generation returned no images")
    return base64.b64decode(images[0])


def _invoke_image_model(
    input_image_b64: str,
    text_prompt: str,
    negative_text: str = "",
    similarity_strength: float = 0.7,
) -> bytes:
    """Call Stable Diffusion 3.5 Large for an image-to-image design preview.

    Args:
        input_image_b64: Base64-encoded PNG of the current car body render.
        text_prompt: Description of the desired modification.
        negative_text: What to avoid in the generated image.
        similarity_strength: 0.2 (creative) to 1.0 (faithful). Default 0.7.

    Returns:
        PNG bytes of the generated design preview image.
    """
    if not 0.2 <= similarity_strength <= 1.0:
        raise ValueError("similarity_strength must be between 0.2 and 1.0")

    # Stability's strength is inverse to Nova's similarityStrength: 0 preserves
    # the input image, while 1 allows the generated image to diverge completely.
    body = {
        "prompt": text_prompt,
        "mode": "image-to-image",
        "image": input_image_b64,
        "strength": round(1.0 - similarity_strength, 4),
        "output_format": "png",
        "seed": 42,
    }
    if negative_text:
        body["negative_prompt"] = negative_text

    response = bedrock_runtime.invoke_model(
        modelId=IMAGE_MODEL_ID,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )
    response_body = json.loads(response["body"].read())
    return _decode_image_response(response_body)


# ---------------------------------------------------------------------------
# Strands @tool functions
# ---------------------------------------------------------------------------

@tool
def render_current_geometry(variant_id: str, geometry_path: str) -> str:
    """Render the current car body geometry to a PNG image.

    Loads the STL mesh, renders it from a 3/4 perspective view, and
    uploads the PNG to S3 for viewing.

    Args:
        variant_id: Unique identifier for the design variant.
        geometry_path: S3 URI or local path to the STL file.

    Returns:
        JSON with the S3 URI of the rendered PNG and mesh statistics.
    """
    start = time.time()
    try:
        mesh = _load_mesh(geometry_path)
        png_bytes = _render_mesh_to_png_bytes(mesh)

        # Upload render to S3
        render_key = f"{GEOMETRY_S3_PREFIX}renders/{variant_id}_current.png"
        s3_client.put_object(
            Bucket=GEOMETRY_S3_BUCKET,
            Key=render_key,
            Body=png_bytes,
            ContentType="image/png",
        )

        bounds = mesh.bounds
        return json.dumps({
            "variant_id": variant_id,
            "render_s3_uri": f"s3://{GEOMETRY_S3_BUCKET}/{render_key}",
            "mesh_stats": {
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "surface_area_m2": round(float(mesh.area), 4),
                "bounding_box_mm": {
                    "x": round(float(bounds[1][0] - bounds[0][0]) * 1000, 1),
                    "y": round(float(bounds[1][1] - bounds[0][1]) * 1000, 1),
                    "z": round(float(bounds[1][2] - bounds[0][2]) * 1000, 1),
                },
                "is_watertight": mesh.is_watertight,
            },
            "elapsed_ms": round((time.time() - start) * 1000, 1),
            "status": "success",
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "variant_id": variant_id,
            "status": "error",
            "error_message": str(e),
        })


@tool
def generate_design_preview(
    variant_id: str,
    geometry_path: str,
    modification_prompt: str,
    negative_prompt: str = "",
    similarity_strength: float = 0.7,
) -> str:
    """Generate a visual design preview using Stability AI Stable Diffusion 3.5 Large.

    Renders the current STL mesh to PNG, then uses image-to-image generation
    to generate a concept image showing the requested modification. This is a
    visual preview only — no mesh changes are made.

    Args:
        variant_id: Unique identifier for the design variant.
        geometry_path: S3 URI or local path to the STL file.
        modification_prompt: Natural language description of the desired change.
            Example: "Add aerodynamic side mirrors to this car body"
        negative_prompt: What to avoid in the generated image.
        similarity_strength: How close to the original (0.2=creative, 1.0=faithful).

    Returns:
        JSON with S3 URIs for both the original render and the Stable Diffusion 3.5 Large preview.
    """
    start = time.time()
    try:
        # Step 1: Render current mesh to PNG
        mesh = _load_mesh(geometry_path)
        png_bytes = _render_mesh_to_png_bytes(mesh)
        input_b64 = base64.b64encode(png_bytes).decode("utf-8")

        # Step 2: Call Stable Diffusion 3.5 Large for design preview
        full_prompt = (
            f"Professional automotive design rendering of a car body. "
            f"{modification_prompt}. "
            f"Photorealistic studio lighting, clean background, "
            f"engineering visualization style, high detail."
        )
        preview_bytes = _invoke_image_model(
            input_image_b64=input_b64,
            text_prompt=full_prompt,
            negative_text=negative_prompt or "blurry, low quality, distorted, cartoon",
            similarity_strength=similarity_strength,
        )

        # Step 3: Upload both images to S3
        original_key = f"{GEOMETRY_S3_PREFIX}renders/{variant_id}_original.png"
        preview_key = f"{GEOMETRY_S3_PREFIX}renders/{variant_id}_preview.png"

        s3_client.put_object(
            Bucket=GEOMETRY_S3_BUCKET, Key=original_key,
            Body=png_bytes, ContentType="image/png",
        )
        s3_client.put_object(
            Bucket=GEOMETRY_S3_BUCKET, Key=preview_key,
            Body=preview_bytes, ContentType="image/png",
        )

        return json.dumps({
            "variant_id": variant_id,
            "original_render_s3_uri": f"s3://{GEOMETRY_S3_BUCKET}/{original_key}",
            "preview_render_s3_uri": f"s3://{GEOMETRY_S3_BUCKET}/{preview_key}",
            "modification_prompt": modification_prompt,
            "similarity_strength": similarity_strength,
            "elapsed_ms": round((time.time() - start) * 1000, 1),
            "status": "success",
            "note": "This is a visual concept preview only. Use modify_geometry to apply actual mesh changes.",
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "variant_id": variant_id,
            "status": "error",
            "error_message": str(e),
        })


@tool
def modify_geometry(
    variant_id: str,
    geometry_path: str,
    operation: str,
    parameters_json: str,
) -> str:
    """Apply a 3D mesh modification to a car body STL geometry.

    Supports operations: add_primitive (boolean union with box/cylinder/sphere),
    scale_region, translate_region, extend_surface, add_side_mirrors,
    extend_bonnet, add_rear_spoiler, add_diffuser.

    Args:
        variant_id: Unique identifier for the design variant.
        geometry_path: S3 URI or local path to the STL file.
        operation: The modification operation to perform. One of:
            "add_side_mirrors", "extend_bonnet", "add_rear_spoiler",
            "add_diffuser", "add_primitive", "scale_region",
            "scale_body", "translate_body".
        parameters_json: JSON string with operation-specific parameters.
            - add_side_mirrors: {"mirror_width": 0.15, "mirror_height": 0.08, "mirror_depth": 0.05, "y_offset": 0.0}
            - extend_bonnet: {"extension_m": 0.1}
            - add_rear_spoiler: {"width_m": 0.8, "height_m": 0.05, "depth_m": 0.15}
            - add_diffuser: {"angle_deg": 10, "depth_m": 0.3}
            - add_primitive: {"shape": "box"|"cylinder"|"sphere", "size": [x,y,z], "position": [x,y,z]}
            - scale_body: {"scale_x": 1.0, "scale_y": 1.0, "scale_z": 1.0}
            - translate_body: {"dx": 0, "dy": 0, "dz": 0}

    Returns:
        JSON with the S3 URI of the modified STL, mesh stats, and a render.
    """
    start = time.time()
    try:
        mesh = _load_mesh(geometry_path)
        params = json.loads(parameters_json)
        new_variant_id = f"{variant_id}_mod_{uuid.uuid4().hex[:6]}"

        if operation == "add_side_mirrors":
            mesh = _add_side_mirrors(mesh, params)
        elif operation == "extend_bonnet":
            mesh = _extend_bonnet(mesh, params)
        elif operation == "add_rear_spoiler":
            mesh = _add_rear_spoiler(mesh, params)
        elif operation == "add_diffuser":
            mesh = _add_diffuser(mesh, params)
        elif operation == "add_primitive":
            mesh = _add_primitive(mesh, params)
        elif operation == "scale_body":
            mesh = _scale_body(mesh, params)
        elif operation == "translate_body":
            mesh = _translate_body(mesh, params)
        else:
            return json.dumps({
                "variant_id": variant_id,
                "status": "error",
                "error_message": f"Unknown operation: {operation}. Supported: add_side_mirrors, extend_bonnet, add_rear_spoiler, add_diffuser, add_primitive, scale_body, translate_body",
            })

        # Upload modified mesh
        s3_uri = _upload_mesh_to_s3(mesh, new_variant_id)

        # Render the modified mesh
        png_bytes = _render_mesh_to_png_bytes(mesh)
        render_key = f"{GEOMETRY_S3_PREFIX}renders/{new_variant_id}_modified.png"
        s3_client.put_object(
            Bucket=GEOMETRY_S3_BUCKET, Key=render_key,
            Body=png_bytes, ContentType="image/png",
        )

        bounds = mesh.bounds
        return json.dumps({
            "variant_id": new_variant_id,
            "original_variant_id": variant_id,
            "operation": operation,
            "parameters": params,
            "modified_stl_s3_uri": s3_uri,
            "modified_render_s3_uri": f"s3://{GEOMETRY_S3_BUCKET}/{render_key}",
            "mesh_stats": {
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "surface_area_m2": round(float(mesh.area), 4),
                "bounding_box_mm": {
                    "x": round(float(bounds[1][0] - bounds[0][0]) * 1000, 1),
                    "y": round(float(bounds[1][1] - bounds[0][1]) * 1000, 1),
                    "z": round(float(bounds[1][2] - bounds[0][2]) * 1000, 1),
                },
                "is_watertight": mesh.is_watertight,
            },
            "elapsed_ms": round((time.time() - start) * 1000, 1),
            "status": "success",
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "variant_id": variant_id,
            "status": "error",
            "error_message": str(e),
        })


# ---------------------------------------------------------------------------
# Coordinate-convention helpers
# ---------------------------------------------------------------------------
# generate_car_design builds bodies with:
#   front = +x  (bounds[1][0] = x_max)
#   rear  = −x  (bounds[0][0] = x_min)
#   up    = +z,  lateral = y
# ALL modifier functions MUST use these accessors so the convention is
# defined in exactly one place and cannot drift.
#
# Important: WindsorML run_N.stl and user-uploaded meshes may use a different
# orientation.  For robustness we detect which end carries the cabin mass
# (_detect_front_x) rather than hard-coding max/min.

def _detect_front_x(mesh: trimesh.Trimesh) -> str:
    """Return 'max' if front=+x (parametric convention) or 'min' if front=-x.

    Heuristic: the front half of a car is lower (nose) and the cabin sits over
    the rear half.  We compare the mean Z of vertices in each x-half; the half
    with the LOWER mean Z is the front (nose slopes downward).
    Falls back to 'max' (parametric convention) if the split is ambiguous.
    """
    verts = mesh.vertices
    mid_x = (verts[:, 0].min() + verts[:, 0].max()) / 2
    z_pos = verts[verts[:, 0] >= mid_x, 2].mean() if np.any(verts[:, 0] >= mid_x) else 0.0
    z_neg = verts[verts[:, 0] <  mid_x, 2].mean() if np.any(verts[:, 0] <  mid_x) else 0.0
    # Lower mean-Z half is the front (nose)
    return "max" if z_pos <= z_neg else "min"


def _front_x(bounds, convention: str = "max") -> float:
    """X coordinate of the front face."""
    return float(bounds[1][0]) if convention == "max" else float(bounds[0][0])


def _rear_x(bounds, convention: str = "max") -> float:
    """X coordinate of the rear face."""
    return float(bounds[0][0]) if convention == "max" else float(bounds[1][0])


# ---------------------------------------------------------------------------
# Module-level mesh-building helpers (also used inside _generate_parametric_car)
# ---------------------------------------------------------------------------

def _hex_mesh(v: list) -> trimesh.Trimesh:
    """Closed hexahedron from 8 corner vertices.

    Order: [0..3] front face (BL BR TR TL), [4..7] rear face (BL BR TR TL).
    B=bottom T=top L=−y R=+y.
    """
    v = np.array(v, dtype=float)
    faces = np.array([
        [0, 2, 1], [0, 3, 2],  # front
        [4, 5, 6], [4, 6, 7],  # rear
        [0, 1, 5], [0, 5, 4],  # bottom
        [3, 7, 6], [3, 6, 2],  # top
        [0, 4, 7], [0, 7, 3],  # left  (−y)
        [1, 2, 6], [1, 6, 5],  # right (+y)
    ])
    return trimesh.Trimesh(vertices=v, faces=faces, process=True)


def _prism_y_mesh(xz_tri: list, y1: float, y2: float) -> trimesh.Trimesh:
    """Triangular prism extruded along Y.

    xz_tri: three (x, z) points defining the triangle cross-section.
    """
    (x1, z1), (x2, z2), (x3, z3) = xz_tri
    v = np.array([
        [x1, y1, z1], [x2, y1, z2], [x3, y1, z3],
        [x1, y2, z1], [x2, y2, z2], [x3, y2, z3],
    ], dtype=float)
    faces = np.array([
        [0, 1, 2], [3, 5, 4],
        [0, 3, 4], [0, 4, 1],
        [1, 4, 5], [1, 5, 2],
        [2, 5, 3], [2, 3, 0],
    ])
    return trimesh.Trimesh(vertices=v, faces=faces, process=True)


# ---------------------------------------------------------------------------
# Mesh modification operations
# ---------------------------------------------------------------------------

def _add_side_mirrors(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """Add curved side mirror housings at the A-pillar / windshield base edge."""
    w = params.get("mirror_width",  0.10)   # Y protrusion from body side
    h = params.get("mirror_height", 0.065)  # Z blade height (cylinder radius × 2)
    d = params.get("mirror_depth",  0.055)  # X fore-aft depth (unused in shape, kept for API compat)

    bounds = mesh.bounds
    conv = _detect_front_x(mesh)
    car_length = bounds[1][0] - bounds[0][0]
    car_height = bounds[1][2] - bounds[0][2]

    # A-pillar / windshield base: ~27% back from the front face
    mirror_x = _front_x(bounds, conv) - car_length * 0.27 * (1 if conv == "max" else -1)

    # ── Find actual body edge Y at door/A-pillar height ───────────────────────
    # Sample vertices in a ±10% length window around the A-pillar X, at door
    # height (excludes wheels near ground and roof).
    zone_hw   = car_length * 0.10
    z_door_lo = bounds[0][2] + car_height * 0.35
    z_door_hi = bounds[0][2] + car_height * 0.68
    x_lo = min(mirror_x - zone_hw, mirror_x + zone_hw)
    x_hi = max(mirror_x - zone_hw, mirror_x + zone_hw)
    x_mask = (mesh.vertices[:, 0] >= x_lo) & (mesh.vertices[:, 0] <= x_hi)
    z_mask = (mesh.vertices[:, 2] >= z_door_lo) & (mesh.vertices[:, 2] <= z_door_hi)
    door_verts = mesh.vertices[x_mask & z_mask]
    if len(door_verts) > 0:
        body_max_y = float(np.abs(door_verts[:, 1]).max())
    else:
        body_max_y = (bounds[1][1] - bounds[0][1]) / 2 * 0.85  # fallback

    # Z: window ledge / A-pillar base height (~60% of total body height from floor)
    mirror_z = bounds[0][2] + car_height * 0.60

    # ── Curved mirror housing ─────────────────────────────────────────────────
    # Cylinder axis along Y (protrudes sideways from body).
    # Cross-section in X-Z plane is circular → smooth aerodynamic profile
    # when viewed from front or side.
    parts = [mesh]
    rot_y = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    for y_sign in [-1, 1]:
        housing = trimesh.creation.cylinder(radius=h / 2, height=w, sections=20)
        housing.apply_transform(rot_y)
        # Inner face of housing flush with body edge; housing protrudes outward
        cy = y_sign * body_max_y + y_sign * w / 2
        housing.apply_translation([mirror_x, cy, mirror_z])
        parts.append(housing)

    combined = trimesh.util.concatenate(parts)
    logger.info(
        f"Added curved side mirrors at A-pillar x={mirror_x:.3f}, body_y=±{body_max_y:.3f}"
    )
    return combined


def _extend_bonnet(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """Extend the front bonnet/hood by pushing front vertices forward (+x)."""
    extension = params.get("extension_m", 0.1)
    bounds = mesh.bounds
    conv = _detect_front_x(mesh)
    car_length = bounds[1][0] - bounds[0][0]

    fx = _front_x(bounds, conv)
    # Threshold: front 25% of the car
    if conv == "max":
        front_threshold = fx - car_length * 0.25
        vertices = mesh.vertices.copy()
        front_mask = vertices[:, 0] > front_threshold
        if np.any(front_mask):
            t = (vertices[front_mask, 0] - front_threshold) / (fx - front_threshold)
            vertices[front_mask, 0] += t * extension   # push forward (+x)
    else:
        front_threshold = fx + car_length * 0.25
        vertices = mesh.vertices.copy()
        front_mask = vertices[:, 0] < front_threshold
        if np.any(front_mask):
            t = (front_threshold - vertices[front_mask, 0]) / (front_threshold - fx)
            vertices[front_mask, 0] -= t * extension   # push forward (−x)

    modified = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)
    logger.info(f"Extended bonnet by {extension}m")
    return modified


def _add_rear_spoiler(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """Add a rear spoiler sitting on the trunk deck / tail of the car."""
    width  = params.get("width_m",  0.8)
    height = params.get("height_m", 0.05)
    depth  = params.get("depth_m",  0.15)

    bounds = mesh.bounds
    conv = _detect_front_x(mesh)
    car_length = bounds[1][0] - bounds[0][0]
    car_width  = bounds[1][1] - bounds[0][1]
    car_height = bounds[1][2] - bounds[0][2]

    rx   = _rear_x(bounds, conv)
    sign = 1 if conv == "max" else -1   # direction pointing away from rear face

    # ── Find the actual trunk deck surface Z ──────────────────────────────────
    # Sample vertices in the rear 30 % of the car at mid-body height.
    # This range (40–75 % of total height) excludes wheels (ground level) and
    # the cabin roof so we isolate the trunk/deck surface height.
    rear_threshold = rx - sign * car_length * 0.30
    z_lo = bounds[0][2] + car_height * 0.40
    z_hi = bounds[0][2] + car_height * 0.75
    if conv == "max":
        rear_verts = mesh.vertices[mesh.vertices[:, 0] >= rear_threshold]
    else:
        rear_verts = mesh.vertices[mesh.vertices[:, 0] <= rear_threshold]
    deck_verts = rear_verts[(rear_verts[:, 2] >= z_lo) & (rear_verts[:, 2] <= z_hi)]
    if len(deck_verts) > 0:
        trunk_deck_z = float(deck_verts[:, 2].max())
    else:
        trunk_deck_z = bounds[0][2] + car_height * 0.60  # fallback

    # X: blade leading edge sits at the rear face; half-depth overhangs inward
    spoiler_x = rx + sign * depth / 2
    # Y: centered laterally
    spoiler_y = (bounds[0][1] + bounds[1][1]) / 2
    # Z: blade rests directly on trunk deck surface
    spoiler_z = trunk_deck_z + height / 2

    spoiler = trimesh.creation.box(extents=[depth, min(width, car_width * 0.90), height])
    spoiler.apply_translation([spoiler_x, spoiler_y, spoiler_z])

    combined = trimesh.util.concatenate([mesh, spoiler])
    logger.info(
        f"Added rear spoiler: {width}m × {height}m × {depth}m "
        f"at rear x={rx:.3f}, trunk_deck_z={trunk_deck_z:.3f}"
    )
    return combined


def _add_diffuser(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """Add a rear diffuser on the underside of the rear of the car."""
    angle_deg = params.get("angle_deg", 10)
    depth     = params.get("depth_m",  0.3)

    bounds = mesh.bounds
    conv = _detect_front_x(mesh)
    car_width = bounds[1][1] - bounds[0][1]

    diffuser_h = max(depth * np.tan(np.radians(angle_deg)), 0.02)

    # X: centred over the rear section, inset from the tail face
    rx   = _rear_x(bounds, conv)
    sign = 1 if conv == "max" else -1
    diffuser_x = rx + sign * depth / 2

    diffuser_y = (bounds[0][1] + bounds[1][1]) / 2

    # Z: sits at ground level (z_min), rising upward — never below ground
    diffuser_z = bounds[0][2] + diffuser_h / 2

    diffuser = trimesh.creation.box(extents=[depth, car_width * 0.6, diffuser_h])
    diffuser.apply_translation([diffuser_x, diffuser_y, diffuser_z])

    combined = trimesh.util.concatenate([mesh, diffuser])
    logger.info(f"Added diffuser: angle={angle_deg}°, depth={depth}m at rear x={rx:.3f}")
    return combined


def _add_primitive(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """Add a primitive shape (box, cylinder, sphere) to the mesh."""
    shape = params.get("shape", "box")
    size = params.get("size", [0.1, 0.1, 0.1])
    position = params.get("position", [0, 0, 0])

    if shape == "box":
        prim = trimesh.creation.box(extents=size)
    elif shape == "cylinder":
        prim = trimesh.creation.cylinder(radius=size[0], height=size[1])
    elif shape == "sphere":
        prim = trimesh.creation.icosphere(radius=size[0])
    else:
        raise ValueError(f"Unknown shape: {shape}")

    prim.apply_translation(position)
    combined = trimesh.util.concatenate([mesh, prim])
    logger.info(f"Added {shape} at {position}")
    return combined


def _scale_body(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """Scale the entire car body."""
    sx = params.get("scale_x", 1.0)
    sy = params.get("scale_y", 1.0)
    sz = params.get("scale_z", 1.0)

    matrix = np.eye(4)
    matrix[0, 0] = sx
    matrix[1, 1] = sy
    matrix[2, 2] = sz

    mesh.apply_transform(matrix)
    logger.info(f"Scaled body: x={sx}, y={sy}, z={sz}")
    return mesh


def _translate_body(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """Translate the entire car body."""
    dx = params.get("dx", 0)
    dy = params.get("dy", 0)
    dz = params.get("dz", 0)
    mesh.apply_translation([dx, dy, dz])
    logger.info(f"Translated body: dx={dx}, dy={dy}, dz={dz}")
    return mesh


@tool
def list_base_variants(limit: int = 20) -> str:
    """List available base car body variants from S3.

    Scans the geometry S3 prefix for available STL files that can be
    used as starting points for modifications.

    Args:
        limit: Maximum number of variants to return.

    Returns:
        JSON with list of available variant IDs and their S3 URIs.
    """
    try:
        response = s3_client.list_objects_v2(
            Bucket=GEOMETRY_S3_BUCKET,
            Prefix=GEOMETRY_S3_PREFIX,
            MaxKeys=limit * 2,
        )
        variants = []
        for obj in response.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".stl"):
                name = key.split("/")[-1].replace(".stl", "")
                variants.append({
                    "variant_id": name,
                    "s3_uri": f"s3://{GEOMETRY_S3_BUCKET}/{key}",
                    "size_bytes": obj["Size"],
                    "last_modified": obj["LastModified"].isoformat(),
                })
            if len(variants) >= limit:
                break

        return json.dumps({
            "variants": variants,
            "count": len(variants),
            "bucket": GEOMETRY_S3_BUCKET,
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error_message": str(e)})


@tool
def generate_text_to_image_preview(
    description: str,
    negative_prompt: str = "",
) -> str:
    """Generate a car design concept image from a text description using Stable Diffusion 3.5 Large.

    Uses text-to-image mode (no input image required). Useful when the engineer
    wants to explore a completely new design concept before creating geometry.

    Args:
        description: Natural language description of the desired car design.
            Example: "A sleek sports car with aggressive side mirrors and a large rear spoiler"
        negative_prompt: What to avoid in the generated image.

    Returns:
        JSON with the S3 URI of the generated concept image.
    """
    start = time.time()
    try:
        full_prompt = (
            f"Professional automotive design rendering. {description}. "
            f"Photorealistic studio lighting, clean white background, "
            f"engineering CAD visualization style, high detail, 3/4 front view."
        )
        body = {
            "prompt": full_prompt,
            "mode": "text-to-image",
            "aspect_ratio": "16:9",
            "output_format": "png",
            "seed": 42,
            "negative_prompt": negative_prompt or "blurry, low quality, distorted, cartoon",
        }

        response = bedrock_runtime.invoke_model(
            modelId=IMAGE_MODEL_ID,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response["body"].read())
        image_bytes = _decode_image_response(response_body)

        # Upload to S3
        concept_id = f"concept_{uuid.uuid4().hex[:8]}"
        concept_key = f"{GEOMETRY_S3_PREFIX}concepts/{concept_id}.png"
        s3_client.put_object(
            Bucket=GEOMETRY_S3_BUCKET, Key=concept_key,
            Body=image_bytes, ContentType="image/png",
        )

        return json.dumps({
            "concept_id": concept_id,
            "concept_s3_uri": f"s3://{GEOMETRY_S3_BUCKET}/{concept_key}",
            "description": description,
            "elapsed_ms": round((time.time() - start) * 1000, 1),
            "status": "success",
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error_message": str(e)})


# ---------------------------------------------------------------------------
# Deterministic Parametric Car Generation (trimesh — runs in agent container)
# ---------------------------------------------------------------------------


def _generate_parametric_car(params: dict) -> trimesh.Trimesh:
    """Build a parametric car body using cross-section profile lofting.

    Coordinate system (WindsorML):
      X  front = +half_L,  rear = −half_L
      Y  left  = −W/2,     right = +W/2
      Z  ground = ride_height, roof = z_roof
    """
    import math
    try:
        from backend.geometry.lofting import _car_profile, _loft_body, _union_parts
    except ModuleNotFoundError:
        from geometry.lofting import _car_profile, _loft_body, _union_parts

    # ── Parameters ────────────────────────────────────────────────────────────
    ride_height    = float(params.get("ride_height", 0.05))
    diffuser_angle = float(params.get("diffuser_angle", 5))
    front_overhang = float(params.get("front_overhang", 0.85))
    rear_slant     = float(params.get("rear_slant", 15))
    boat_tail      = float(params.get("boat_tail_angle", params.get("boat_tail", 5)))
    segment        = str(params.get("segment", "sedan")).lower()
    quality        = str(params.get("quality", "viewer")).lower()
    n_pts          = 96 if quality == "simulation" else 48

    # ── Segment proportions ────────────────────────────────────────────────────
    if segment == "suv" or ride_height > 0.07:
        L, W, H          = 1.15,  0.42,  0.38
        wheel_r, wheel_w = 0.090, 0.055
        chassis_frac     = 0.42
        hood_frac        = 0.20
        trunk_frac       = 0.10
        cabin_w_ratio    = 0.94
        nose_z_frac      = 0.70
        rear_drop        = 0.10
        n_corner         = 4.0
        ride_height      = max(ride_height, 0.08)

    elif segment in ("sport", "sports_car"):
        L, W, H          = 1.05,  0.43,  0.23
        wheel_r, wheel_w = 0.076, 0.055
        chassis_frac     = 0.58
        hood_frac        = 0.40
        trunk_frac       = 0.18
        cabin_w_ratio    = 0.78
        nose_z_frac      = 0.28
        rear_drop        = 0.82
        n_corner         = 2.2
        ride_height      = min(ride_height, 0.025)

    elif segment == "hatchback":
        L, W, H          = 0.92,  0.38,  0.33
        wheel_r, wheel_w = 0.072, 0.045
        chassis_frac     = 0.38
        hood_frac        = 0.18
        trunk_frac       = 0.02
        cabin_w_ratio    = 0.91
        nose_z_frac      = 0.55
        rear_drop        = 0.72
        n_corner         = 2.5
        ride_height      = max(ride_height, 0.04)

    elif segment in ("mini_suv", "mini suv"):
        L, W, H          = 1.00,  0.40,  0.34
        wheel_r, wheel_w = 0.080, 0.048
        chassis_frac     = 0.41
        hood_frac        = 0.22
        trunk_frac       = 0.12
        cabin_w_ratio    = 0.92
        nose_z_frac      = 0.62
        rear_drop        = 0.22
        n_corner         = 3.5
        ride_height      = max(ride_height, 0.065)

    else:  # sedan
        L, W, H          = 1.044, 0.389, 0.289
        wheel_r, wheel_w = 0.080, 0.050
        chassis_frac     = 0.62
        hood_frac        = 0.34
        trunk_frac       = 0.26
        cabin_w_ratio    = 0.78
        nose_z_frac      = 0.50
        rear_drop        = 0.35
        n_corner         = 3.0
        ride_height      = max(ride_height, 0.04)

    # ── User-parameter adjustments ────────────────────────────────────────────
    rear_drop    = max(0.0, min(0.95, rear_drop + (rear_slant - 15.0) / 35.0 * 0.25))
    rear_w_scale = max(0.70, 1.0 - (boat_tail / 25.0) * 0.30)
    overhang_scale = front_overhang / 0.85
    L = L * min(max(overhang_scale, 0.85), 1.2)

    # ── Derived geometry ──────────────────────────────────────────────────────
    chassis_h  = H * chassis_frac
    cabin_h    = H * (1.0 - chassis_frac)
    half_L     = L / 2.0
    hw         = W / 2.0
    cabin_hw   = W * cabin_w_ratio / 2.0
    tail_hw    = hw * rear_w_scale
    x_nose     =  half_L
    x_ws_base  =  half_L - L * hood_frac
    x_rw_base  = -half_L + L * trunk_frac
    x_tail     = -half_L
    z_floor    = ride_height
    z_shoulder = ride_height + chassis_h
    z_roof     = z_shoulder + cabin_h
    cabin_len  = x_ws_base - x_rw_base

    # ── Station 4/5/6/7 z_hi depends on segment and rear_drop ────────────────
    z5_hi = z_shoulder + cabin_h * (1.0 - rear_drop)
    # Station 4 (65% along cabin) blends from z_roof toward z5_hi so that
    # rear_slant visibly lowers the rear cabin even at mid-cabin stations.
    z4_hi = z_roof * 0.70 + z5_hi * 0.30

    if segment in ("sedan",):
        z6_hi = z_shoulder + cabin_h * 0.18
        z7_hi = z_shoulder + cabin_h * 0.05
    elif segment in ("sport", "sports_car"):
        z6_hi = z_shoulder + cabin_h * max(1.0 - rear_drop - 0.10, 0.05)
        z7_hi = z_shoulder + cabin_h * max(1.0 - rear_drop - 0.20, 0.02)
    elif segment == "suv":
        z6_hi = z_shoulder
        z7_hi = z_shoulder
    elif segment == "hatchback":
        z6_hi = z5_hi * 0.95
        z7_hi = z5_hi * 0.75
    else:  # mini_suv
        z6_hi = z_shoulder + cabin_h * 0.08
        z7_hi = z_shoulder + cabin_h * 0.03

    # ── Build 8 loft stations ─────────────────────────────────────────────────
    def stn(x, hw_s, chw_s, z_sh_s, z_hi_s, nc=None):
        return (x, _car_profile(
            hw_s, chw_s, z_floor, z_sh_s, z_hi_s,
            n_corner=nc if nc is not None else n_corner,
            n_pts=n_pts,
        ))

    stations = [
        stn(x_nose,
            0.025, 0.025,
            z_floor + chassis_h * nose_z_frac * 0.5,
            z_floor + chassis_h * nose_z_frac,
            nc=2.5),
        stn(x_nose - L * 0.15,
            hw * 0.85, hw * 0.82,
            z_shoulder, z_shoulder),
        stn(x_ws_base,
            hw, hw,
            z_shoulder, z_shoulder + 0.01),
        stn(x_ws_base - cabin_len * 0.25,
            hw, cabin_hw,
            z_shoulder, z_roof),
        stn(x_ws_base - cabin_len * 0.65,
            hw * 0.99, cabin_hw,
            z_shoulder, z4_hi),
        stn(x_rw_base,
            hw * 0.97, cabin_hw,
            z_shoulder, z5_hi),
        stn(x_tail + L * trunk_frac * 0.5,
            tail_hw * 0.99, cabin_hw * 0.85,
            z_shoulder, z6_hi,
            nc=n_corner * 0.9),
        stn(x_tail,
            tail_hw, tail_hw * 0.88,
            z_shoulder * 0.92, z7_hi,
            nc=n_corner * 0.85),
    ]

    # Add extra stations in simulation quality mode
    if quality == "simulation":
        extra = []
        for i in range(len(stations) - 1):
            extra.append(stations[i])
            x0, p0 = stations[i]
            x1, p1 = stations[i + 1]
            xm = (x0 + x1) / 2.0
            pm = (p0 + p1) / 2.0
            extra.append((xm, pm))
        extra.append(stations[-1])
        stations = extra

    body = _loft_body(stations)

    # ── Add-ons (5 mm penetration into body surface for union) ───────────────
    addons = [body]

    # Grille housing — protrudes forward from nose
    _grille_spec = {
        "sport": (0.54, 0.22), "sports_car": (0.54, 0.22),
        "suv": (0.72, 0.44), "hatchback": (0.52, 0.26), "mini_suv": (0.62, 0.36),
    }
    g_wf, g_hf = _grille_spec.get(segment, (0.56, 0.30))
    g_h = chassis_h * g_hf
    grille = trimesh.creation.box(extents=[0.015, W * g_wf, g_h])
    grille.apply_translation([x_nose + 0.015 / 2, 0, z_floor + 0.018 + g_h / 2])
    addons.append(grille)

    # Rear bumper — extends 5 mm into tail face
    _bumper_h_frac = {
        "suv": 0.22, "sport": 0.12, "sports_car": 0.12,
        "hatchback": 0.18, "mini_suv": 0.20,
    }
    bmp_h = chassis_h * _bumper_h_frac.get(segment, 0.15)
    bumper = trimesh.creation.box(extents=[0.027, W * 0.92, bmp_h])
    bumper.apply_translation([x_tail - 0.027 / 2 + 0.005, 0, z_floor + bmp_h / 2])
    addons.append(bumper)

    # Trunk lid (sedan/mini_suv) — base 5 mm below z_shoulder
    if segment in ("sedan", "mini_suv", "mini suv") and trunk_frac >= 0.15:
        trunk_h   = cabin_h * 0.18
        trunk_len = L * trunk_frac * 0.6
        trunk_box = trimesh.creation.box(extents=[trunk_len, W * 0.90, trunk_h + 0.005])
        trunk_box.apply_translation([
            x_tail + trunk_len / 2 + L * 0.02,
            0,
            z_shoulder - 0.005 + (trunk_h + 0.005) / 2,
        ])
        addons.append(trunk_box)

    # Bonnet crease / power dome — base 5 mm into hood surface
    _bonnet_crease = {
        "sport": (0.030, 0.016), "sports_car": (0.030, 0.016),
        "suv": (0.065, 0.013), "hatchback": (0.028, 0.006), "mini_suv": (0.048, 0.010),
    }
    cr_hw, cr_h = _bonnet_crease.get(segment, (0.038, 0.009))
    hood_mid_x  = (x_nose + x_ws_base) / 2.0
    hood_len    = x_nose - x_ws_base

    def _bonnet_ridge(cy_offset):
        ridge = trimesh.creation.box(extents=[hood_len, cr_hw * 2, cr_h + 0.005])
        ridge.apply_translation([
            hood_mid_x,
            cy_offset,
            z_shoulder - 0.005 + (cr_h + 0.005) / 2,
        ])
        return ridge

    if segment in ("sport", "sports_car"):
        addons.extend([_bonnet_ridge(-0.06), _bonnet_ridge(0.06)])
    else:
        addons.append(_bonnet_ridge(0.0))

    # Diffuser — top 5 mm into underbody
    if diffuser_angle > 2:
        diff_depth = L * 0.12
        diff_h = min(
            diff_depth * math.tan(math.radians(diffuser_angle)),
            chassis_h * 0.3,
        )
        diffuser = trimesh.creation.box(extents=[diff_depth, W * 0.85, diff_h + 0.005])
        diffuser.apply_translation([
            x_tail + diff_depth / 2 + 0.01,
            0,
            z_floor + (diff_h + 0.005) / 2 - 0.005,
        ])
        addons.append(diffuser)

    # viewer quality: fast concatenate (Three.js renders shells fine, no union needed)
    # simulation quality: manifold3d boolean union for watertight MLSimKit mesh
    if quality == "simulation":
        body_with_addons = _union_parts(addons)
    else:
        body_with_addons = trimesh.util.concatenate(addons)
        body_with_addons.fix_normals()

    # ── Wheels (concatenated, not unioned) ────────────────────────────────────
    wheel_parts = [body_with_addons]
    wheelbase = L * 0.65
    track     = W * 0.86
    for x_sign in [-1, 1]:
        for y_sign in [-1, 1]:
            wheel_cx = x_sign * wheelbase / 2
            wheel_cy = y_sign * track / 2
            tire = trimesh.creation.cylinder(radius=wheel_r, height=wheel_w, sections=24)
            tire.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
            tire.apply_translation([wheel_cx, wheel_cy, wheel_r])
            wheel_parts.append(tire)
            rim = trimesh.creation.cylinder(
                radius=wheel_r * 0.65, height=wheel_w * 0.18, sections=16
            )
            rim.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
            rim.apply_translation([wheel_cx, wheel_cy + y_sign * wheel_w * 0.40, wheel_r])
            wheel_parts.append(rim)

    result = trimesh.util.concatenate(wheel_parts)
    result.fix_normals()
    return result





@tool
def generate_car_design(parameter_json: str) -> str:
    """Generate a parametric 3D car model from design parameters.

    Builds a deterministic car mesh using cross-section profile lofting.
    Faster and more reliable than Code Interpreter for standard parametric designs.

    Args:
        parameter_json: JSON string with design parameters:
            - segment: Vehicle type — sedan, sport, suv, hatchback, mini_suv (default sedan)
            - ride_height: Ground clearance in meters (default 0.05)
            - rear_slant: Rear window angle in degrees (0-35, default 15)
            - diffuser_angle: Rear diffuser angle in degrees (0-20, default 5)
            - front_overhang: Front overhang in meters (0.5-1.2, default 0.85)
            - boat_tail_angle: Rear taper angle in degrees (0-15, default 5)
            - quality: "viewer" (default, fast) or "simulation" (higher resolution)

    Returns:
        JSON with:
          stl_s3_uri, variant_id, segment, dimensions,
          is_watertight, is_volume,
          euler_number — reflects the full concatenated mesh topology (body + wheels);
            expect 18 (9 genus-0 shells: 1 body-with-addons + 4 tires + 4 rims),
          simulation_ready  — False if mesh is not watertight,
          simulation_warning — True for sport/hatchback/suv (outside WindsorML distribution),
          vertices, faces, elapsed_ms
    """
    start = time.time()
    try:
        params = json.loads(parameter_json)
        mesh = _generate_parametric_car(params)
        variant_id = f"parametric_{uuid.uuid4().hex[:8]}"
        s3_uri = _upload_mesh_to_s3(mesh, variant_id)

        bounds = mesh.bounds
        dims = (
            f"{round(float(bounds[1][0] - bounds[0][0]), 2)}m × "
            f"{round(float(bounds[1][1] - bounds[0][1]), 2)}m × "
            f"{round(float(bounds[1][2] - bounds[0][2]), 2)}m"
        )
        segment = params.get("segment", "sedan")
        simulation_warning = str(segment).lower() not in ("sedan", "mini_suv", "mini suv")

        return json.dumps({
            "status":             "success",
            "stl_s3_uri":         s3_uri,
            "variant_id":         variant_id,
            "segment":            segment,
            "dimensions":         dims,
            "is_watertight":      bool(mesh.is_watertight),
            "is_volume":          bool(mesh.is_volume),
            "euler_number":       int(mesh.euler_number),
            "simulation_ready":   bool(mesh.is_watertight and mesh.is_volume),
            "simulation_warning": simulation_warning,
            "vertices":           len(mesh.vertices),
            "faces":              len(mesh.faces),
            "elapsed_ms":         round((time.time() - start) * 1000, 1),
        }, indent=2)
    except Exception as e:
        logger.error(f"Parametric generation failed: {e}")
        return json.dumps({"status": "error", "error_message": str(e)})


# ---------------------------------------------------------------------------
# Code Interpreter — custom STL generation (fallback for non-standard shapes)
# ---------------------------------------------------------------------------

CODE_INTERPRETER_BUCKET = GEOMETRY_S3_BUCKET


@tool
def execute_python_for_stl(code: str) -> str:
    """Execute Python code in Code Interpreter to generate an STL file.

    The code should use numpy + struct to construct a 3D mesh and write it
    as binary STL. The sandbox has NO network access — only numpy and struct
    are available. Do NOT use trimesh, cadquery, or pip install.

    Args:
        code: Python code that generates a mesh and saves it to 'output.stl'.
              Must use only numpy and struct. Write binary STL format directly.

    Returns:
        JSON with stl_s3_uri on success, or error details on failure.
    """
    try:
        variant_id = f"parametric_{uuid.uuid4().hex[:8]}"
        stl_filename = f"{variant_id}.stl"
        s3_key = f"geometries/{stl_filename}"

        # Generate presigned PUT URL for the sandbox to upload the STL.
        # NOTE: The presigned URL is ~1200 chars. We write it to a file in the
        # sandbox rather than embedding it in the code string, because the code
        # string becomes part of the tool_use block in the conversation and
        # bloats the context, contributing to MaxTokensReachedException.
        presigned_put_url = s3_client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": CODE_INTERPRETER_BUCKET,
                "Key": s3_key,
                "ContentType": "application/octet-stream",
            },
            ExpiresIn=300,
        )

        # First code block: write the presigned URL to a file (keeps it out of
        # the main code string that the model sees in its conversation context)
        url_setup_code = f'''
import os
with open('/tmp/_upload_url.txt', 'w') as f:
    f.write("""{presigned_put_url}""")
print("URL_READY")
'''

        # S3 upload code — reads URL from file instead of embedding it inline
        s3_upload_code = '''
import requests, os
try:
    with open('/tmp/_upload_url.txt') as f:
        _url = f.read().strip()
    stl_path = 'output.stl' if os.path.exists('output.stl') else '/tmp/output.stl'
    with open(stl_path, 'rb') as f:
        r = requests.put(_url, data=f.read(), headers={"Content-Type": "application/octet-stream"})
    print("[S3_SUCCESS]" if r.status_code == 200 else f"[S3_ERROR]HTTP {r.status_code}[/S3_ERROR]")
except Exception as e:
    print(f"[S3_ERROR]{str(e)}[/S3_ERROR]")
'''

        # Ensure code writes to output.stl (add fallback if not present)
        if "output.stl" not in code:
            code = code + "\nwrite_stl(triangles, 'output.stl')\n"

        full_code = code + s3_upload_code

        logger.info(f"Executing code interpreter for STL generation: {variant_id}")

        with code_session(AWS_REGION) as session:
            # Step 1: Write the presigned URL to a file in the sandbox
            # (keeps the ~1200-char URL out of the main code string)
            url_resp = session.invoke("executeCode", {
                "code": url_setup_code,
                "language": "python",
                "clearContext": False,
            })
            # Drain the url setup stream
            for event in url_resp["stream"]:
                pass

            # Step 2: Run the user's mesh code + S3 upload
            response = session.invoke("executeCode", {
                "code": full_code,
                "language": "python",
                "clearContext": False,
            })

            result_text = ""
            upload_success = False

            for event in response["stream"]:
                if "result" in event:
                    result = event["result"]
                    if isinstance(result, dict):
                        if "structuredContent" in result:
                            result_text = str(result["structuredContent"].get("stdout", "")) + str(result["structuredContent"].get("stderr", ""))
                        else:
                            result_text = str(result.get("stdout", "")) + str(result.get("stderr", ""))
                    else:
                        result_text = str(result)

                    if "[S3_SUCCESS]" in result_text:
                        upload_success = True
                    if "[S3_ERROR]" in result_text:
                        error = result_text.split("[S3_ERROR]")[1].split("[/S3_ERROR]")[0]
                        logger.error(f"S3 upload error: {error}")

            if upload_success:
                s3_uri = f"s3://{CODE_INTERPRETER_BUCKET}/{s3_key}"
                logger.info(f"STL uploaded: {s3_uri}")
                # Return ONLY the s3:// URI. The orchestrator will generate
                # the presigned URL and [STL] tag — keeping this response
                # small prevents A2A payload overflow.
                return json.dumps({
                    "status": "success",
                    "variant_id": variant_id,
                    "stl_s3_uri": s3_uri,
                }, indent=2)
            else:
                return json.dumps({
                    "status": "error",
                    "error_message": f"Code executed but STL upload failed. Output: {result_text}",
                })

    except Exception as e:
        logger.error(f"Code interpreter error: {e}")
        return json.dumps({"status": "error", "error_message": str(e)})


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = f"""You are the Geometry Agent for the Car Design Space Explorer.

You modify car body STL meshes, generate visual previews with Stable Diffusion 3.5 Large, and create parametric designs via Code Interpreter.

## Tools
1. **render_current_geometry** — Render STL to PNG.
2. **generate_design_preview** — Stable Diffusion 3.5 Large visual concept (no mesh change).
3. **modify_geometry** — Apply 3D mesh modifications. Operations: add_side_mirrors, extend_bonnet, add_rear_spoiler, add_diffuser, add_primitive, scale_body, translate_body.
4. **list_base_variants** — List available STL files in S3.
5. **generate_text_to_image_preview** — Concept image from text only.
6. **generate_car_design** — PREFERRED for parametric car generation. Builds a deterministic car mesh (chassis+cabin+wheels) from parameters. Fast, reliable, no Code Interpreter needed.
7. **execute_python_for_stl** — FALLBACK for custom/creative shapes that generate_car_design can't handle. Runs Python (numpy+struct only) in Code Interpreter. NO trimesh/pip install.

## Parametric Design Routing
- Standard parametric queries (ride_height, rear_slant, diffuser_angle, front_overhang, boat_tail_angle) → use **generate_car_design**
- Custom/creative shapes, complex geometry, non-standard designs → use **execute_python_for_stl**

## Parametric Windsor Body
Parameters: ride_height (default 0.05m), rear_slant (0-35°), diffuser_angle (0-20°), front_overhang (0.5-1.2m), boat_tail_angle (0-15°), segment (sedan/sport/suv/hatchback/mini_suv).

## CRITICAL: Car Composition Rules
generate_car_design builds the car from explicit geometric primitives — NEVER a single box:
1. Lower body slab — tapered at nose and tail
2. Cabin greenhouse box — narrower than body, segment-specific length/position
3. Windshield prism — rakes from hood top up to cabin roof
4. Rear window section — slope controlled by rear_slant; taper by boat_tail_angle
5. Four wheels — cylinders with rim discs
6. Optional: trunk lid shelf (sedan/mini_suv), diffuser wedge

Parameter → Geometry Mapping:
- segment → Sets silhouette: sedan (3-box), sport (long hood, fastback, slammed), suv (tall boxy, high ground clearance), hatchback (short hood, steep rear drop, no trunk), mini_suv (compact crossover)
- ride_height → Ground clearance; sport is clamped low (≤0.025m), suv clamped high (≥0.08m)
- rear_slant (0-35°) → Higher = more aggressive roofline drop toward tail (fastback/hatch effect)
- boat_tail_angle (0-25°) → Higher = narrower rear end; tapers lower body and rear window section
- front_overhang (0.5-1.2m) → Scales overall body length
- diffuser_angle (0-20°) → Depth of angled wedge at rear underside

When using generate_car_design, always include segment if the user mentions a vehicle type (e.g., "sports car" → segment: "sport").

## CRITICAL: Code Interpreter — numpy + struct ONLY
The Code Interpreter sandbox has NO network access. `pip install` will fail.
Only `numpy` and `struct` are available. Do NOT import trimesh, cadquery, or any external library.

To write STL binary format with numpy + struct:
```python
import numpy as np, struct

def write_stl(triangles, filename='output.stl'):
    \"\"\"Write binary STL. triangles: Nx3x3 array of vertex coords.\"\"\"
    triangles = np.asarray(triangles, dtype=np.float32)
    n = len(triangles)
    with open(filename, 'wb') as f:
        f.write(b'\\0' * 80)  # header
        f.write(struct.pack('<I', n))
        for tri in triangles:
            v0, v1, v2 = tri
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            normal = normal / norm if norm > 0 else np.zeros(3)
            f.write(struct.pack('<3f', *normal))
            for v in tri:
                f.write(struct.pack('<3f', *v))
            f.write(struct.pack('<H', 0))

# Build a box: 8 vertices → 12 triangles
def box(cx, cy, cz, sx, sy, sz):
    \"\"\"Axis-aligned box centered at (cx,cy,cz) with half-sizes (sx,sy,sz).\"\"\"
    v = np.array([[cx+dx, cy+dy, cz+dz]
                  for dx in [-sx,sx] for dy in [-sy,sy] for dz in [-sz,sz]], dtype=np.float32)
    faces = [[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
             [2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]]
    return np.array([[v[f[0]], v[f[1]], v[f[2]]] for f in faces], dtype=np.float32)
```
Use box() for body, wheel wells, etc. Concatenate triangle arrays with np.concatenate().
Always save as 'output.stl'.

## Response Rules
- Keep total response under 800 chars. Verbose responses crash the pipeline.
- When execute_python_for_stl succeeds, respond with ONLY the stl_s3_uri. Example: "Generated parametric STL at s3://bucket/key"
- Do NOT generate presigned URLs, [STL] tags, or list parameters back — the Orchestrator handles that.
- Do NOT call additional tools after execute_python_for_stl succeeds.
- For modify_geometry, return the JSON output directly.
- Variants: run_1 through run_355. Modified variants: {{original}}_mod_{{hash}}.
"""

# ---------------------------------------------------------------------------
# Agent + A2A Server setup
# ---------------------------------------------------------------------------
logger.info("Creating Geometry Agent...")

from strands.models import BedrockModel as _BedrockModel

geometry_model = _BedrockModel(
    model_id=MODEL_ID,
    max_tokens=2048,  # Reduced from 8192. generate_car_design needs ~150 chars.
    # execute_python_for_stl (fallback) may need more for code generation.
)

from strands.agent.conversation_manager import SlidingWindowConversationManager as _SWCM

# Conversation manager keeps context lean within a single A2A request.
# The geometry agent is stateless (each A2A call is independent), but within
# one call the model may invoke multiple tools, growing the context.
# window_size=6 keeps only the most recent turns to prevent MaxTokensReached.
_geometry_conversation_manager = _SWCM(window_size=6)

agent = Agent(
    name="Geometry Agent",
    description=(
        "Modifies car body STL meshes via natural language instructions. "
        "Generates visual design previews using Stability AI Stable Diffusion 3.5 Large and applies "
        "actual 3D mesh modifications using trimesh. Supports adding side mirrors, "
        "extending bonnets, adding spoilers/diffusers, and arbitrary primitives."
    ),
    system_prompt=SYSTEM_PROMPT,
    model=geometry_model,
    tools=[
        render_current_geometry,
        generate_design_preview,
        modify_geometry,
        list_base_variants,
        generate_text_to_image_preview,
        generate_car_design,
        execute_python_for_stl,
    ],
    conversation_manager=_geometry_conversation_manager,
    concurrent_invocation_mode=ConcurrentInvocationMode.UNSAFE_REENTRANT,
)

# Clear conversation history before each A2A request so the agent is stateless
# across requests. Without this, messages accumulate and context grows unbounded.
def _clear_messages(event: BeforeInvocationEvent):
    event.agent.messages.clear()

agent.add_hook(_clear_messages)
logger.info("✅ Geometry Agent created")

# OTel guard — patch from_converse to handle missing 'output' key
try:
    from opentelemetry.instrumentation.botocore.extensions import bedrock_utils
    _original_from_converse = bedrock_utils._Choice.from_converse

    @classmethod
    def _safe_from_converse(cls, response, capture_content=False):
        try:
            return _original_from_converse.__func__(cls, response, capture_content)
        except (KeyError, TypeError, Exception) as e:
            logger.warning(f"[otel_guard] ConverseStream response issue: {e} — returning empty choice")
            try:
                return cls(finish_reason="error", message=None, index=0)
            except TypeError:
                return cls(finish_reason="error", message=None)

    bedrock_utils._Choice.from_converse = _safe_from_converse
    logger.info("✅ OTel from_converse patched for KeyError guard")
except Exception as e:
    logger.warning(f"OTel patch skipped: {e}")

runtime_url = os.environ.get("AGENTCORE_RUNTIME_URL", f"http://127.0.0.1:{AGENT_PORT}/")
logger.info(f"Runtime URL: {runtime_url}")


# ---------------------------------------------------------------------------
# HistoryStrippingTaskStore — prevents A2A response payload bloat
# ---------------------------------------------------------------------------
class HistoryStrippingTaskStore(TaskStore):
    """TaskStore that strips conversation history from tasks on retrieval."""

    def __init__(self) -> None:
        self._inner = InMemoryTaskStore()

    async def save(self, task: A2ATask, context: ServerCallContext | None = None) -> None:
        await self._inner.save(task, context)

    async def get(self, task_id: str, context: ServerCallContext | None = None) -> A2ATask | None:
        task = await self._inner.get(task_id, context)
        if task is not None:
            task.history = None
        return task

    async def delete(self, task_id: str, context: ServerCallContext | None = None) -> None:
        await self._inner.delete(task_id, context)


a2a_server = A2AServer(
    agent=agent,
    http_url=runtime_url,
    serve_at_root=True,
    task_store=HistoryStrippingTaskStore(),
    enable_a2a_compliant_streaming=False,  # Synchronous — one complete payload
)

# FastAPI app
app = FastAPI()


@app.get("/ping")
def ping():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agent": "geometry_agent",
        "version": "1.0.0",
        "features": [
            "a2a_protocol",
            "stable_diffusion_preview",
            "stl_mesh_modification",
            "side_mirror_addition",
            "bonnet_extension",
            "rear_spoiler",
            "diffuser",
            "primitive_boolean_union",
            "s3_geometry_storage",
        ],
        "image_model": IMAGE_MODEL_ID,
        "image_model_region": IMAGE_MODEL_REGION,
        "geometry_bucket": GEOMETRY_S3_BUCKET,
    }


_a2a_app = a2a_server.to_fastapi_app()


# ---------------------------------------------------------------------------
# ASGI middleware: wrap non-JSON-RPC payloads arriving at POST /
# and truncate oversized responses
# ---------------------------------------------------------------------------
import uuid as _uuid
from starlette.types import ASGIApp, Receive, Scope, Send


MAX_RESPONSE_BYTES = 60_000  # Synchronous A2A — safe to allow larger payloads


def _truncate_a2a_response(body: bytes) -> bytes:
    """Truncate an A2A JSON-RPC response if it exceeds MAX_RESPONSE_BYTES."""
    if len(body) <= MAX_RESPONSE_BYTES:
        return body

    try:
        data = json.loads(body)
    except (json.JSONDecodeError, Exception):
        return body[:MAX_RESPONSE_BYTES]

    result = data.get("result", {})
    if not isinstance(result, dict):
        return json.dumps(data).encode()[:MAX_RESPONSE_BYTES]

    # 1. ALWAYS strip history
    result.pop("history", None)
    task_obj = result.get("task", result) if "task" in result else result
    task_obj.pop("history", None)

    # 2. Merge ALL artifacts' text into one, collapse to single artifact
    artifacts = task_obj.get("artifacts", result.get("artifacts", []))
    if artifacts:
        all_text = ""
        for artifact in artifacts:
            for part in artifact.get("parts", []):
                if isinstance(part, dict) and part.get("text"):
                    all_text += part["text"]
                elif isinstance(part, dict) and part.get("kind") == "text" and "text" in part:
                    all_text += part["text"]
        trimmed = all_text.strip()[:2000]
        collapsed = [{
            "artifactId": artifacts[0].get("artifactId", ""),
            "name": artifacts[0].get("name", "agent_response"),
            "parts": [{"kind": "text", "text": trimmed}],
        }]
        if "artifacts" in task_obj:
            task_obj["artifacts"] = collapsed
        elif "artifacts" in result:
            result["artifacts"] = collapsed

    # 3. Nuke status.message if artifacts exist
    status = task_obj.get("status", result.get("status", {}))
    if isinstance(status, dict) and artifacts:
        status.pop("message", None)

    truncated = json.dumps(data, separators=(",", ":")).encode()

    if len(truncated) > MAX_RESPONSE_BYTES:
        truncated = truncated[:MAX_RESPONSE_BYTES]

    logger.info(f"Truncated A2A response from {len(body)} to {len(truncated)} bytes")
    return truncated


class A2APayloadNormalizer:
    """ASGI middleware that normalizes A2A payloads and truncates responses."""

    def __init__(self, wrapped_app: ASGIApp):
        self.app = wrapped_app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http" or scope.get("method", "") != "POST":
            await self.app(scope, receive, send)
            return

        body_parts = []
        while True:
            message = await receive()
            body_parts.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        raw_body = b"".join(body_parts)

        needs_wrap = False
        try:
            parsed = json.loads(raw_body)
            if (isinstance(parsed, dict)
                    and parsed.get("jsonrpc") == "2.0"
                    and "method" in parsed):
                msg = parsed.get("params", {}).get("message", {})
                if msg and "messageId" not in msg:
                    msg["messageId"] = str(_uuid.uuid4())
                    raw_body = json.dumps(parsed).encode()
            else:
                needs_wrap = True
        except (json.JSONDecodeError, Exception):
            needs_wrap = True

        if needs_wrap:
            try:
                parsed = json.loads(raw_body)
                if isinstance(parsed, dict):
                    prompt = parsed.get("prompt", parsed.get("input", {}).get("prompt", ""))
                    if not prompt:
                        prompt = json.dumps(parsed)
                else:
                    prompt = str(parsed)
            except Exception:
                prompt = raw_body.decode("utf-8", errors="replace")

            a2a_envelope = {
                "jsonrpc": "2.0",
                "id": str(_uuid.uuid4()),
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": prompt}],
                        "messageId": str(_uuid.uuid4()),
                    }
                },
            }
            raw_body = json.dumps(a2a_envelope).encode()
            logger.info(f"Wrapped raw payload into A2A JSON-RPC envelope")

        body_sent = False

        async def wrapped_receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": raw_body, "more_body": False}
            return {"type": "http.disconnect"}

        # --- OUTBOUND: intercept response and truncate if needed ---
        response_headers_sent = False
        response_body_parts = []

        async def capturing_send(message):
            nonlocal response_headers_sent
            if message["type"] == "http.response.start":
                response_headers_sent = message
            elif message["type"] == "http.response.body":
                response_body_parts.append(message.get("body", b""))
                if not message.get("more_body", False):
                    full_body = b"".join(response_body_parts)
                    truncated_body = _truncate_a2a_response(full_body)

                    if response_headers_sent:
                        headers = list(response_headers_sent.get("headers", []))
                        new_headers = []
                        for h_name, h_val in headers:
                            if h_name.lower() == b"content-length":
                                new_headers.append((h_name, str(len(truncated_body)).encode()))
                            else:
                                new_headers.append((h_name, h_val))
                        response_headers_sent["headers"] = new_headers
                        await send(response_headers_sent)

                    await send({
                        "type": "http.response.body",
                        "body": truncated_body,
                        "more_body": False,
                    })
            else:
                await send(message)

        await self.app(scope, wrapped_receive, capturing_send)


_wrapped_a2a_app = A2APayloadNormalizer(_a2a_app)
app.mount("/", _wrapped_a2a_app)


# ---------------------------------------------------------------------------
# /invocations route — AgentCore Runtime forwards payloads here
# ---------------------------------------------------------------------------
from fastapi import Request
from fastapi.responses import JSONResponse
import httpx


@app.post("/invocations")
async def invocations(request: Request):
    """Handle /invocations calls (belt-and-suspenders fallback).

    For A2A protocol agents, AgentCore sends JSON-RPC to POST / directly.
    This route exists as a safety net. It handles two payload formats:
    1. Already valid A2A JSON-RPC → pass through to the A2A app
    2. Simple {"prompt": "..."} → wrap into JSON-RPC and forward
    """
    try:
        body = await request.json()
    except Exception:
        raw = await request.body()
        try:
            body = json.loads(raw)
        except Exception:
            body = {"prompt": raw.decode("utf-8", errors="replace")}

    import uuid as _uuid
    if (isinstance(body, dict)
            and body.get("jsonrpc") == "2.0"
            and body.get("method") in ("message/send", "message/stream")):
        a2a_payload = body
        msg = a2a_payload.get("params", {}).get("message", {})
        if "messageId" not in msg:
            msg["messageId"] = str(_uuid.uuid4())
    else:
        prompt = body.get("prompt", body.get("input", {}).get("prompt", "")) if isinstance(body, dict) else str(body)
        if not prompt and isinstance(body, dict):
            prompt = json.dumps(body)

        a2a_payload = {
            "jsonrpc": "2.0",
            "id": str(_uuid.uuid4()),
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": prompt}],
                    "messageId": str(_uuid.uuid4()),
                }
            },
        }

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_a2a_app), base_url="http://internal"
    ) as client:
        headers = {}
        for k, v in request.headers.items():
            if k.lower() not in ("host", "content-length", "content-type", "transfer-encoding"):
                headers[k] = v
        headers["content-type"] = "application/json"

        resp = await client.post("/", json=a2a_payload, headers=headers, timeout=600.0)

    return JSONResponse(content=resp.json(), status_code=resp.status_code)


if __name__ == "__main__":
    # AgentCore containers require an all-interface bind; ingress is protected
    # by the Runtime JWT authorizer and AgentCore network boundary.
    host, port = "0.0.0.0", AGENT_PORT  # nosec B104
    print()
    print("=" * 60)
    print("Geometry Agent — Car Design Space Explorer")
    print(f"  Agent Card: http://{host}:{port}/.well-known/agent-card.json")
    print(f"  Image Model: {IMAGE_MODEL_ID} ({IMAGE_MODEL_REGION})")
    print(f"  S3 Bucket: {GEOMETRY_S3_BUCKET}")
    print("=" * 60)
    uvicorn.run(app, host=host, port=port)
