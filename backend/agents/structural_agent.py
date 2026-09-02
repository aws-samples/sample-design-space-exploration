#!/usr/bin/env python3
"""
Structural Agent — Strands A2A Server for structural feasibility evaluation.

Evaluates car body design variants for structural feasibility using
rule-based heuristics: weight, stiffness, thickness recommendation,
and manufacturing constraint checks.

Architecture:
- Strands Agent framework with @tool decorated functions
- A2A Server for agent-to-agent communication
- FastAPI with health check endpoint
- Deployed to Bedrock AgentCore Runtime
"""

from __future__ import annotations

import json
import logging
import os

import tempfile

import boto3
import numpy as np
import trimesh
import uvicorn
from fastapi import FastAPI
from strands import Agent, tool
from strands.models import BedrockModel
from strands.agent.agent import ConcurrentInvocationMode
from strands.hooks.events import BeforeInvocationEvent
from strands.multiagent.a2a import A2AServer
from strands.agent.conversation_manager import SlidingWindowConversationManager

# A2A TaskStore — strip history from responses to prevent payload bloat
from a2a.server.tasks import TaskStore, InMemoryTaskStore
from a2a.server.context import ServerCallContext
from a2a.types import Task as A2ATask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AGENT_PORT = int(os.environ.get("STRUCTURAL_AGENT_PORT", "9000"))
MODEL_ID = os.environ.get("MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
MAX_TOOL_RESULT_CHARS = 5000  # geometry_metrics from STL can be verbose; truncate to prevent context overflow
GEOMETRY_S3_BUCKET = os.environ.get("GEOMETRY_S3_BUCKET", "")
GEOMETRY_S3_PREFIX = os.environ.get("GEOMETRY_S3_PREFIX", "geometries/")

if not GEOMETRY_S3_BUCKET:
    logger.warning(
        "GEOMETRY_S3_BUCKET is not configured. Mesh loading from S3 will fail. "
        "Set the GEOMETRY_S3_BUCKET environment variable."
    )

# S3 client for loading STL files
s3_client = boto3.client("s3", region_name=AWS_REGION)

# ---------------------------------------------------------------------------
# Material constants
# ---------------------------------------------------------------------------
MATERIAL_DENSITIES: dict[str, float] = {
    "steel": 7850.0,
    "aluminum": 2700.0,
    "carbon_fiber": 1600.0,
}

BASE_THICKNESS_MM: dict[str, float] = {
    "steel": 1.0,
    "aluminum": 1.5,
    "carbon_fiber": 2.0,
}

MAX_CURVATURE_VARIATION = 2.0
MAX_DRAW_DEPTH_MM = 300.0

# WindsorML reference maximums for normalization
MAX_VERTEX_COUNT = 500_000
MAX_CURVATURE = 3.0
MAX_PATCHES = 200
MAX_DEPTH = 500.0


# ---------------------------------------------------------------------------
# Helper: Load mesh from S3
# ---------------------------------------------------------------------------

def _load_mesh_from_s3(variant_id: str) -> trimesh.Trimesh:
    """Download an STL file from S3 using a variant_id-derived key and load it."""
    if not GEOMETRY_S3_BUCKET:
        raise ValueError(
            "GEOMETRY_S3_BUCKET is not configured. Set the GEOMETRY_S3_BUCKET "
            "environment variable to load geometry files from S3."
        )
    s3_key = f"{GEOMETRY_S3_PREFIX}{variant_id}.stl"
    logger.info(f"Loading mesh from s3://{GEOMETRY_S3_BUCKET}/{s3_key}")
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        s3_client.download_file(GEOMETRY_S3_BUCKET, s3_key, tmp.name)
        tmp.flush()
        mesh = trimesh.load(tmp.name, force="mesh")
    logger.info(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    return mesh


def _normalize_geometry_path(geometry_path: str) -> str:
    """Convert presigned HTTPS URL to s3:// URI if needed."""
    if geometry_path.startswith("https://") and ".s3." in geometry_path and "amazonaws.com" in geometry_path:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(geometry_path)
            bucket = parsed.hostname.split(".s3.")[0]
            key = parsed.path.lstrip("/")
            return f"s3://{bucket}/{key}"
        except Exception:
            pass
    return geometry_path


def _load_mesh_from_uri(s3_uri: str) -> trimesh.Trimesh:
    """Download an STL file from an explicit s3:// URI and load it."""
    parts = s3_uri.replace("s3://", "").split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    bucket, key = parts
    logger.info(f"Loading mesh from {s3_uri}")
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        s3_client.download_file(bucket, key, tmp.name)
        tmp.flush()
        mesh = trimesh.load(tmp.name, force="mesh")
    logger.info(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    return mesh


# ---------------------------------------------------------------------------
# Strands @tool functions — the agent's callable capabilities
# ---------------------------------------------------------------------------

@tool
def compute_geometry_metrics(variant_id: str, geometry_path: str = "") -> str:
    """Compute geometry metrics for a car body design variant from its STL file.

    Downloads the STL mesh from S3, analyzes it with trimesh, and returns
    all geometry metrics needed by the Structural and Cost agents.

    Args:
        variant_id: Unique identifier for the design variant (e.g. "run_125").
        geometry_path: Optional explicit s3:// URI to the STL file. When provided,
            this path is used directly instead of deriving it from variant_id.
            Required for parametric or uploaded variants whose S3 key may differ
            from the standard geometries/{variant_id}.stl pattern.

    Returns:
        JSON string with geometry metrics: surface_area_m2, vertex_count,
        curvature_variation, surface_patch_count, max_draw_depth_mm, has_undercuts.
    """
    try:
        geometry_path = _normalize_geometry_path(geometry_path) if geometry_path else ""
        if geometry_path and geometry_path.startswith("s3://"):
            mesh = _load_mesh_from_uri(geometry_path)
        else:
            mesh = _load_mesh_from_s3(variant_id)

        # Surface area in m² (STL units assumed mm, convert)
        bounds = mesh.bounds  # [[xmin,ymin,zmin],[xmax,ymax,zmax]]
        extent = bounds[1] - bounds[0]
        max_dim = max(extent)

        # Heuristic: if max dimension > 50, assume mm; otherwise assume meters
        if max_dim > 50:
            # STL is in mm — convert area from mm² to m²
            surface_area_m2 = mesh.area / 1_000_000.0
            max_draw_depth_mm = extent[2]  # Z-extent already in mm
        else:
            # STL is in meters
            surface_area_m2 = mesh.area
            max_draw_depth_mm = extent[2] * 1000.0  # convert m to mm

        vertex_count = len(mesh.vertices)

        # Curvature variation: std of discrete mean curvature
        # Use vertex normals divergence as proxy
        try:
            # trimesh discrete mean curvature via vertex defect
            angles = mesh.face_angles  # (n_faces, 3)
            vertex_angle_sum = np.zeros(vertex_count)
            for i in range(3):
                np.add.at(vertex_angle_sum, mesh.faces[:, i], angles[:, i])
            # Angular defect = 2π - sum of angles at vertex
            angular_defect = 2.0 * np.pi - vertex_angle_sum
            # Use mixed Voronoi area per vertex as denominator
            face_areas = mesh.area_faces
            vertex_area = np.zeros(vertex_count)
            for i in range(3):
                np.add.at(vertex_area, mesh.faces[:, i], face_areas / 3.0)
            vertex_area = np.maximum(vertex_area, 1e-12)
            mean_curvature = angular_defect / vertex_area
            curvature_variation = float(np.std(mean_curvature))
        except Exception as e:
            logger.warning(f"Curvature computation failed, using fallback: {e}")
            curvature_variation = 0.5  # reasonable default

        # Surface patch count: number of connected components
        try:
            components = mesh.split(only_watertight=False)
            surface_patch_count = len(components)
        except Exception:
            surface_patch_count = 1

        # Undercut detection: check for faces with normals pointing
        # significantly downward (negative Z) — indicates negative draft angles
        face_normals_z = mesh.face_normals[:, 2]
        # Faces with normal Z < -0.5 (more than 60° from vertical) = undercut
        undercut_faces = np.sum(face_normals_z < -0.5)
        undercut_ratio = undercut_faces / len(mesh.faces) if len(mesh.faces) > 0 else 0
        has_undercuts = bool(undercut_ratio > 0.05)  # >5% undercut faces

        result = {
            "variant_id": variant_id,
            "surface_area_m2": round(surface_area_m2, 6),
            "vertex_count": vertex_count,
            "curvature_variation": round(curvature_variation, 6),
            "surface_patch_count": surface_patch_count,
            "max_draw_depth_mm": round(max_draw_depth_mm, 2),
            "has_undercuts": has_undercuts,
            "status": "success",
        }
        logger.info(f"Geometry metrics for {variant_id}: {result}")
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"compute_geometry_metrics failed for {variant_id}: {e}")
        return json.dumps({
            "variant_id": variant_id,
            "status": "error",
            "error_message": str(e),
        }, indent=2)


@tool
def evaluate_structural_feasibility(
    variant_id: str,
    surface_area_m2: float,
    vertex_count: int,
    curvature_variation: float,
    surface_patch_count: int,
    max_draw_depth_mm: float,
    has_undercuts: bool,
    material: str = "steel",
) -> str:
    """Evaluate structural feasibility of a car body design variant.

    Computes weight, stiffness score, recommended thickness, feasibility score,
    and checks manufacturing constraints (curvature, draw depth, undercuts).

    Args:
        variant_id: Unique identifier for the design variant (e.g. "run_15").
        surface_area_m2: Total mesh surface area in square meters.
        vertex_count: Number of mesh vertices.
        curvature_variation: Standard deviation of discrete curvature at vertices.
        surface_patch_count: Number of connected surface patches.
        max_draw_depth_mm: Maximum Z-extent of geometry in millimeters.
        has_undercuts: Whether the geometry contains negative draft angles.
        material: Material type — one of "steel", "aluminum", "carbon_fiber".

    Returns:
        JSON string with structural evaluation results.
    """
    try:
        density = MATERIAL_DENSITIES.get(material, MATERIAL_DENSITIES["steel"])
        base_thickness = BASE_THICKNESS_MM.get(material, BASE_THICKNESS_MM["steel"])

        # Complexity score (0-1)
        complexity = (
            0.3 * min(vertex_count / MAX_VERTEX_COUNT, 1.0)
            + 0.3 * min(curvature_variation / MAX_CURVATURE, 1.0)
            + 0.2 * min(surface_patch_count / MAX_PATCHES, 1.0)
            + 0.2 * min(max_draw_depth_mm / MAX_DEPTH, 1.0)
        )
        complexity = min(max(complexity, 0.0), 1.0)

        # Recommended thickness
        recommended_thickness = base_thickness * (1.0 + 0.2 * complexity)

        # Weight: surface_area × thickness(m) × density
        weight_kg = surface_area_m2 * (recommended_thickness / 1000.0) * density

        # Stiffness score (0-1)
        curvature_penalty = min(curvature_variation / MAX_CURVATURE_VARIATION, 1.0) * 0.5
        thickness_penalty = min(1.0 / max(recommended_thickness, 0.1), 1.0) * 0.5
        stiffness_score = max(1.0 - curvature_penalty - thickness_penalty, 0.0)

        # Constraint violations
        violations = []
        if curvature_variation > MAX_CURVATURE_VARIATION:
            violations.append(
                f"Excessive curvature variation: {curvature_variation:.2f} (max {MAX_CURVATURE_VARIATION})"
            )
        if max_draw_depth_mm > MAX_DRAW_DEPTH_MM:
            violations.append(
                f"Draw depth exceeds limit: {max_draw_depth_mm:.1f}mm (max {MAX_DRAW_DEPTH_MM}mm)"
            )
        if has_undercuts:
            violations.append("Geometry contains undercuts (negative draft angles)")

        # Feasibility score
        curvature_ratio = min(curvature_variation / MAX_CURVATURE_VARIATION, 1.0)
        depth_ratio = min(max_draw_depth_mm / MAX_DRAW_DEPTH_MM, 1.0)
        feasibility_score = (
            stiffness_score * 0.4
            + (1.0 - curvature_ratio) * 0.3
            + (1.0 - depth_ratio) * 0.3
        )
        is_feasible = feasibility_score >= 0.5 and len(violations) == 0

        result = {
            "variant_id": variant_id,
            "weight_kg": round(weight_kg, 3),
            "stiffness_score": round(stiffness_score, 4),
            "recommended_thickness_mm": round(recommended_thickness, 3),
            "feasibility_score": round(feasibility_score, 4),
            "is_feasible": is_feasible,
            "constraint_violations": violations,
            "complexity_score": round(complexity, 4),
            "material": material,
            "status": "success",
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        error_result = {
            "variant_id": variant_id,
            "status": "error",
            "error_message": str(e),
        }
        return json.dumps(error_result, indent=2)


@tool
def evaluate_structural_batch(variants_json: str) -> str:
    """Evaluate structural feasibility for a batch of design variants.

    Args:
        variants_json: JSON array of variant objects, each containing:
            variant_id, surface_area_m2, vertex_count, curvature_variation,
            surface_patch_count, max_draw_depth_mm, has_undercuts, material.

    Returns:
        JSON array of structural evaluation results.
    """
    try:
        variants = json.loads(variants_json)
        results = []
        for v in variants:
            result_str = evaluate_structural_feasibility(
                variant_id=v["variant_id"],
                surface_area_m2=v["surface_area_m2"],
                vertex_count=v["vertex_count"],
                curvature_variation=v["curvature_variation"],
                surface_patch_count=v["surface_patch_count"],
                max_draw_depth_mm=v["max_draw_depth_mm"],
                has_undercuts=v["has_undercuts"],
                material=v.get("material", "steel"),
            )
            results.append(json.loads(result_str))
        return json.dumps(results, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error_message": str(e)})


@tool
def get_material_properties(material: str = "steel") -> str:
    """Get material properties used for structural evaluation.

    Args:
        material: Material type — "steel", "aluminum", or "carbon_fiber".

    Returns:
        JSON with density, base thickness, and feasibility thresholds.
    """
    return json.dumps({
        "material": material,
        "density_kg_m3": MATERIAL_DENSITIES.get(material, MATERIAL_DENSITIES["steel"]),
        "base_thickness_mm": BASE_THICKNESS_MM.get(material, BASE_THICKNESS_MM["steel"]),
        "max_curvature_variation": MAX_CURVATURE_VARIATION,
        "max_draw_depth_mm": MAX_DRAW_DEPTH_MM,
    }, indent=2)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the Structural Agent for the Car Design Space Explorer.

You compute geometry metrics from STL files and evaluate structural feasibility of car body variants.

## Tools
1. **compute_geometry_metrics(variant_id, geometry_path="")** — Load STL from S3, return surface_area_m2, vertex_count, curvature_variation, surface_patch_count, max_draw_depth_mm, has_undercuts. Pass geometry_path when an explicit s3:// URI is provided (required for parametric/uploaded variants).
2. **evaluate_structural_feasibility(...)** — Weight, stiffness, thickness, feasibility score, constraint violations.
3. **evaluate_structural_batch(variants_json)** — Batch evaluation.
4. **get_material_properties(material)** — Density, base thickness, thresholds.

## Workflow
- When asked for geometry metrics: call compute_geometry_metrics(variant_id, geometry_path=<uri if provided>), return result directly.
- When asked for structural evaluation: call compute_geometry_metrics first (pass geometry_path if given), then evaluate_structural_feasibility with those metrics.
- Materials: steel (7850 kg/m³, 1.0mm), aluminum (2700, 1.5mm), carbon_fiber (1600, 2.0mm).

## Response Rules
Return ONLY the JSON from tools. No prose, no markdown. Under 1500 chars.
"""

# ---------------------------------------------------------------------------
# Conversation manager — truncates large tool results before Bedrock API calls
# ---------------------------------------------------------------------------
class TruncatingConversationManager(SlidingWindowConversationManager):
    """Sliding window manager that truncates large tool results.

    compute_geometry_metrics on large STL files can return verbose output.
    Truncating keeps each conversation turn within Bedrock's token limit.
    """

    def apply_management(self, agent):
        for msg in agent.messages:
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or not block.get("toolResult"):
                    continue
                tr = block["toolResult"]
                tr_content = tr.get("content", [])
                total_len = sum(
                    len(p.get("text", "")) for p in tr_content
                    if isinstance(p, dict)
                )
                if total_len > MAX_TOOL_RESULT_CHARS:
                    truncated_parts = []
                    remaining = MAX_TOOL_RESULT_CHARS
                    for part in tr_content:
                        if isinstance(part, dict) and "text" in part:
                            if remaining > 0:
                                truncated_parts.append({"text": part["text"][:remaining]})
                                remaining -= len(part["text"])
                        else:
                            truncated_parts.append(part)
                    truncated_parts.append({
                        "text": f"\n[TRUNCATED from {total_len} chars. Use the data you have.]"
                    })
                    tr["content"] = truncated_parts
                    logger.info(f"Truncated tool result from {total_len} to ~{MAX_TOOL_RESULT_CHARS} chars")
        super().apply_management(agent)


# ---------------------------------------------------------------------------
# Agent + A2A Server setup
# ---------------------------------------------------------------------------
logger.info("Creating Structural Agent...")
structural_model = BedrockModel(
    model_id=MODEL_ID,
    max_tokens=1024,
)
agent = Agent(
    name="Structural Agent",
    description="Evaluates structural feasibility of car body design variants including weight, stiffness, thickness recommendation, and manufacturing constraint checks",
    system_prompt=SYSTEM_PROMPT,
    model=structural_model,
    tools=[compute_geometry_metrics, evaluate_structural_feasibility, evaluate_structural_batch, get_material_properties],
    conversation_manager=TruncatingConversationManager(window_size=6, per_turn=True),
    concurrent_invocation_mode=ConcurrentInvocationMode.UNSAFE_REENTRANT,
)

# Clear conversation history before each A2A request so the agent is stateless
# across requests. Without this, messages accumulate and context grows unbounded.
def _clear_messages(event: BeforeInvocationEvent):
    event.agent.messages.clear()

agent.add_hook(_clear_messages)
logger.info("✅ Structural Agent created")

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
        "agent": "structural_agent",
        "version": "2.0.0",
        "features": [
            "a2a_protocol",
            "geometry_metrics_from_stl",
            "weight_estimation",
            "stiffness_scoring",
            "thickness_recommendation",
            "feasibility_assessment",
            "constraint_checking",
        ],
        "materials": list(MATERIAL_DENSITIES.keys()),
    }


_a2a_app = a2a_server.to_fastapi_app()


# ---------------------------------------------------------------------------
# ASGI middleware: wrap non-JSON-RPC payloads arriving at POST /
# ---------------------------------------------------------------------------
import uuid as _uuid
from starlette.types import ASGIApp, Receive, Scope, Send


class A2APayloadNormalizer:
    """ASGI middleware that wraps raw payloads into A2A JSON-RPC envelopes."""

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

        await self.app(scope, wrapped_receive, send)


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
    print("Structural Agent — Car Design Space Explorer")
    print(f"  Agent Card: http://{host}:{port}/.well-known/agent-card.json")
    print("=" * 60)
    uvicorn.run(app, host=host, port=port)
