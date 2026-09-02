#!/usr/bin/env python3
"""
Cost Agent — Strands A2A Server for manufacturing cost estimation.

Pure MCP-driven agent: all data access goes through MCP servers via
AgentCore Gateway with Cognito JWT auth.

Set GATEWAY_URL env var, or the agent will auto-discover the gateway.
Credentials are loaded from Secrets Manager (car-design/mcp-gateway-credentials).

MCP Servers (behind Gateway):
  1. Internal Cost Parameters — base material costs,
     stamping rates, tooling costs, assembly costs, complexity multipliers
  2. External Cost Data — market pricing, supplier
     quotes, historical benchmarks, regional factors, volume discounts

The @tool functions handle pure cost computation math. The LLM orchestrates
data retrieval from MCP servers and feeds parameters into the computation tools.

Architecture:
- Strands Agent framework with @tool decorated functions
- MCP servers accessed via AgentCore Gateway (JWT auth)
- A2A Server for agent-to-agent communication
- Deployed to Bedrock AgentCore Runtime
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback

import requests
import boto3
import uvicorn
from fastapi import FastAPI
from strands import Agent, tool
from strands.agent.agent import ConcurrentInvocationMode
from strands.hooks.events import BeforeInvocationEvent
from strands.multiagent.a2a import A2AServer

# A2A TaskStore — strip history from responses to prevent payload bloat
from a2a.server.tasks import TaskStore, InMemoryTaskStore
from a2a.server.context import ServerCallContext
from a2a.types import Task as A2ATask

# MCP imports — top-level, same pattern as SPA agent
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamablehttp_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AGENT_PORT = int(os.environ.get("COST_AGENT_PORT", "9000"))
MODEL_ID = os.environ.get("MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")

# MCP Gateway Configuration — OAuth2 client credentials flow
# Credentials are stored in Secrets Manager by the McpAuthStack CDK deployment.
# The secret contains: client_id, client_secret, token_url, scope, user_pool_id
# When MCP_GATEWAY_SECRET_NAME is set, credentials are loaded from Secrets Manager.
# When GATEWAY_URL is set, MCP servers are accessed via AgentCore Gateway with JWT auth.
# When neither is set, falls back to direct localhost URLs (local dev mode).
MCP_GATEWAY_SECRET_NAME = os.environ.get(  # nosec B105 -- resource name only
    "MCP_GATEWAY_SECRET_NAME",
    "car-design/mcp-gateway-credentials",
)
GATEWAY_URL = os.environ.get("GATEWAY_URL", "")

# These can be overridden by env vars, but are normally loaded from Secrets Manager
GATEWAY_COGNITO_CLIENT_ID = os.environ.get("GATEWAY_COGNITO_CLIENT_ID", "")
GATEWAY_COGNITO_CLIENT_SECRET = os.environ.get("GATEWAY_COGNITO_CLIENT_SECRET", "")
GATEWAY_TOKEN_URL = os.environ.get("GATEWAY_TOKEN_URL", "")
GATEWAY_SCOPE = os.environ.get("GATEWAY_SCOPE", "mcp-api/read mcp-api/write")

# Load credentials from Secrets Manager if available
if MCP_GATEWAY_SECRET_NAME and not GATEWAY_COGNITO_CLIENT_ID:
    try:
        _sm = boto3.client("secretsmanager", region_name=AWS_REGION)
        _secret_resp = _sm.get_secret_value(SecretId=MCP_GATEWAY_SECRET_NAME)
        _creds = json.loads(_secret_resp["SecretString"])
        GATEWAY_COGNITO_CLIENT_ID = _creds.get("client_id", "")
        GATEWAY_COGNITO_CLIENT_SECRET = _creds.get("client_secret", "")
        GATEWAY_TOKEN_URL = _creds.get("token_url", "")
        GATEWAY_SCOPE = _creds.get("scope", GATEWAY_SCOPE)
        logger.info(f"✅ MCP Gateway credentials loaded from Secrets Manager: {MCP_GATEWAY_SECRET_NAME}")
    except Exception as e:
        logger.warning(f"⚠️ Could not load MCP Gateway credentials from Secrets Manager: {e}")

if not GATEWAY_URL:
    logger.warning(
        "⚠️ GATEWAY_URL is not configured. MCP tools are available only in "
        "explicit local-development mode; deployed agents must receive the URL "
        "from deploy_agents.py."
    )

# Token cache
_access_token_cache: str | None = None
_token_expiry: float | None = None

# ---------------------------------------------------------------------------
# Physical constants (pure physics — not data, no MCP needed)
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

# WindsorML reference maximums for normalization
MAX_VERTEX_COUNT = 500_000
MAX_CURVATURE = 3.0
MAX_PATCHES = 200
MAX_DEPTH = 500.0

# Assembly estimation constants
BASE_STAMPING_OPS = 4
DRAW_DEPTH_THRESHOLD_MM = 200.0
MULTI_STAGE_MULTIPLIER_DEFAULT = 1.15
ASSEMBLY_COST_PER_PANEL = 85.0
PANELS_PER_PATCH_RATIO = 0.15  # ~15% of surface patches become assembly panels


# ---------------------------------------------------------------------------
# Strands @tool functions — pure cost computation math
# ---------------------------------------------------------------------------

@tool
def compute_manufacturing_cost(
    variant_id: str,
    surface_area_m2: float,
    vertex_count: int,
    curvature_variation: float,
    surface_patch_count: int,
    max_draw_depth_mm: float,
    has_undercuts: bool,
    material: str = "steel",
    material_cost_per_kg: float = 1.20,
    stamping_cost_per_op: float = 150.0,
    tooling_base_cost: float = 50000.0,
    complexity_multiplier: float = 1.3,
    welding_cost_per_meter: float = 12.0,
) -> str:
    """Compute full manufacturing cost breakdown for a car body design variant.

    This is a pure computation tool. All cost parameters should be fetched
    from the MCP servers first, then passed in as arguments.

    Args:
        variant_id: Unique identifier for the design variant (e.g. "run_15").
        surface_area_m2: Total mesh surface area in square meters.
        vertex_count: Number of mesh vertices.
        curvature_variation: Standard deviation of discrete curvature.
        surface_patch_count: Number of connected surface patches.
        max_draw_depth_mm: Maximum Z-extent of geometry in millimeters.
        has_undercuts: Whether the geometry contains negative draft angles.
        material: Material type — "steel", "aluminum", or "carbon_fiber".
        material_cost_per_kg: Cost per kilogram for the material (from MCP).
        stamping_cost_per_op: Cost per stamping operation (from MCP).
        tooling_base_cost: Base tooling/die cost in USD (from MCP).
        complexity_multiplier: Multiplier based on geometry complexity (from MCP).
        welding_cost_per_meter: Welding cost per meter of weld line (from MCP).

    Returns:
        JSON string with complete cost breakdown.
    """
    try:
        density = MATERIAL_DENSITIES.get(material, MATERIAL_DENSITIES["steel"])
        base_thickness = BASE_THICKNESS_MM.get(material, BASE_THICKNESS_MM["steel"])

        # Complexity score (0-1) — same formula as structural agent
        complexity = (
            0.3 * min(vertex_count / MAX_VERTEX_COUNT, 1.0)
            + 0.3 * min(curvature_variation / MAX_CURVATURE, 1.0)
            + 0.2 * min(surface_patch_count / MAX_PATCHES, 1.0)
            + 0.2 * min(max_draw_depth_mm / MAX_DEPTH, 1.0)
        )
        complexity = min(max(complexity, 0.0), 1.0)

        # Recommended thickness (adjusted for complexity)
        thickness_mm = base_thickness * (1.0 + 0.2 * complexity)

        # Weight
        weight_kg = surface_area_m2 * (thickness_mm / 1000.0) * density

        # --- Cost components ---

        # 1. Material cost
        mat_cost = weight_kg * material_cost_per_kg

        # 2. Stamping cost
        stamping_ops = BASE_STAMPING_OPS
        if has_undercuts:
            stamping_ops += 1
        if max_draw_depth_mm > DRAW_DEPTH_THRESHOLD_MM:
            stamping_ops += 1
        stamp_cost = stamping_ops * stamping_cost_per_op * MULTI_STAGE_MULTIPLIER_DEFAULT

        # 3. Tooling cost
        tool_cost = tooling_base_cost * complexity_multiplier

        # 4. Assembly cost
        panel_count = max(1, int(surface_patch_count * PANELS_PER_PATCH_RATIO))
        # Estimate weld line length from surface area (heuristic: sqrt(area) * patch ratio)
        weld_line_length_m = (surface_area_m2 ** 0.5) * panel_count * 0.3
        asm_cost = (panel_count * ASSEMBLY_COST_PER_PANEL
                    + weld_line_length_m * welding_cost_per_meter)

        total = mat_cost + stamp_cost + tool_cost + asm_cost

        result = {
            "variant_id": variant_id,
            "total_cost": round(total, 2),
            "material_cost": round(mat_cost, 2),
            "stamping_cost": round(stamp_cost, 2),
            "tooling_cost": round(tool_cost, 2),
            "assembly_cost": round(asm_cost, 2),
            "weight_kg": round(weight_kg, 3),
            "complexity_score": round(complexity, 4),
            "thickness_mm": round(thickness_mm, 3),
            "stamping_operations": stamping_ops,
            "panel_count": panel_count,
            "weld_line_length_m": round(weld_line_length_m, 2),
            "material": material,
            "status": "success",
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({
            "variant_id": variant_id,
            "status": "error",
            "error_message": str(e),
        }, indent=2)


@tool
def compute_cost_batch(variants_json: str) -> str:
    """Compute manufacturing cost for a batch of design variants.

    Each variant object should include geometry metrics AND cost parameters
    (fetched from MCP servers by the LLM before calling this tool).

    Args:
        variants_json: JSON array of variant objects, each containing:
            variant_id, surface_area_m2, vertex_count, curvature_variation,
            surface_patch_count, max_draw_depth_mm, has_undercuts, material,
            material_cost_per_kg, stamping_cost_per_op, tooling_base_cost,
            complexity_multiplier, welding_cost_per_meter.

    Returns:
        JSON array of cost breakdown results.
    """
    try:
        variants = json.loads(variants_json)
        results = []
        for v in variants:
            result_str = compute_manufacturing_cost(
                variant_id=v["variant_id"],
                surface_area_m2=v["surface_area_m2"],
                vertex_count=v["vertex_count"],
                curvature_variation=v["curvature_variation"],
                surface_patch_count=v["surface_patch_count"],
                max_draw_depth_mm=v["max_draw_depth_mm"],
                has_undercuts=v["has_undercuts"],
                material=v.get("material", "steel"),
                material_cost_per_kg=v.get("material_cost_per_kg", 1.20),
                stamping_cost_per_op=v.get("stamping_cost_per_op", 150.0),
                tooling_base_cost=v.get("tooling_base_cost", 50000.0),
                complexity_multiplier=v.get("complexity_multiplier", 1.3),
                welding_cost_per_meter=v.get("welding_cost_per_meter", 12.0),
            )
            results.append(json.loads(result_str))
        return json.dumps(results, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error_message": str(e)})


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the Cost Agent for the Car Design Space Explorer.

You estimate manufacturing costs for car body variants by combining geometry metrics with cost parameters.

## Tools
- **compute_manufacturing_cost** — Full cost breakdown for one variant.
- **compute_cost_batch** — Batch cost computation.

## Cost Formula
total = material_cost + stamping_cost + tooling_cost + assembly_cost
Materials: steel (7850 kg/m³), aluminum (2700), carbon_fiber (1600).
"""

# Dynamic system prompt section based on MCP availability
_MCP_AVAILABLE_PROMPT = """
## MCP Data Access
You have MCP tools for cost data via AgentCore Gateway. ALWAYS fetch parameters before computing.
Key tools: get_all_cost_parameters (most efficient — all 8 categories), get_material_cost, get_stamping_parameters, get_tooling_base_cost, get_assembly_parameters, get_complexity_multipliers.
External: get_material_market_price, get_supplier_quotes, get_historical_cost_benchmarks, get_regional_cost_factors, get_volume_discount_schedule.

Workflow: fetch get_all_cost_parameters → determine complexity → call compute_manufacturing_cost with fetched params.
"""

_MCP_UNAVAILABLE_PROMPT = """
## No MCP — Use Defaults
MCP tools are NOT available. Call compute_manufacturing_cost directly with defaults:
steel=1.20/kg, aluminum=3.50/kg, carbon_fiber=25.0/kg, stamping=150.0/op, tooling=50000.0, complexity=1.3, welding=12.0/m.
"""

_SYSTEM_PROMPT_FOOTER = """
## Response Rules
Return ONLY the JSON from compute_manufacturing_cost. No prose, no echoed parameters, no MCP data dumps. Total text under 800 chars.
"""

# ---------------------------------------------------------------------------
# OAuth2 Gateway token acquisition — same pattern as SPA agent
# ---------------------------------------------------------------------------

def get_gateway_access_token() -> str | None:
    """Get access token from Gateway Cognito using OAuth2 client credentials with caching."""
    global _access_token_cache, _token_expiry

    if _access_token_cache and _token_expiry and time.time() < _token_expiry:
        return _access_token_cache

    if not GATEWAY_TOKEN_URL or not GATEWAY_COGNITO_CLIENT_ID or not GATEWAY_COGNITO_CLIENT_SECRET:
        logger.error("Gateway OAuth2 credentials not configured")
        return None

    try:
        response = requests.post(
            GATEWAY_TOKEN_URL,
            data=(
                f"grant_type=client_credentials"
                f"&client_id={GATEWAY_COGNITO_CLIENT_ID}"
                f"&client_secret={GATEWAY_COGNITO_CLIENT_SECRET}"
                f"&scope={GATEWAY_SCOPE}"
            ),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        response.raise_for_status()
        token_data = response.json()
        _access_token_cache = token_data["access_token"]
        _token_expiry = time.time() + (50 * 60)
        logger.info("✅ Gateway OAuth2 access token obtained")
        return _access_token_cache
    except Exception as e:
        logger.error(f"Failed to get Gateway access token: {e}")
        return None


def create_streamable_http_transport(mcp_url: str, access_token: str):
    """Create HTTP transport for MCP client with Bearer token."""
    return streamablehttp_client(mcp_url, headers={"Authorization": f"Bearer {access_token}"})


# ---------------------------------------------------------------------------
# MCP Client setup — Gateway only (same pattern as SPA agent)
# ---------------------------------------------------------------------------
_mcp_client = None
_mcp_tools_cache = None


def get_mcp_tools():
    """Get tools from Gateway MCP server — called once at startup."""
    global _mcp_tools_cache, _mcp_client

    if _mcp_tools_cache is not None:
        return _mcp_tools_cache

    try:
        if not GATEWAY_URL:
            logger.warning("Gateway URL not configured — MCP tools unavailable")
            return []

        access_token = get_gateway_access_token()
        if not access_token:
            logger.warning("Failed to get access token — MCP tools unavailable")
            return []

        _mcp_client = MCPClient(lambda: create_streamable_http_transport(GATEWAY_URL, access_token))
        _mcp_client.__enter__()

        tools = []
        pagination_token = None
        while True:
            tmp_tools = _mcp_client.list_tools_sync(pagination_token=pagination_token)
            tools.extend(tmp_tools)
            if tmp_tools.pagination_token is None:
                break
            pagination_token = tmp_tools.pagination_token

        _mcp_tools_cache = tools
        logger.info(f"✅ Loaded {len(tools)} MCP tools from Gateway: {[t.tool_name for t in tools]}")
        return tools
    except Exception as e:
        logger.error(f"Failed to get MCP tools: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []


# Load MCP tools at startup
mcp_tools = get_mcp_tools()
if mcp_tools:
    logger.info(f"🔧 MCP tools ready: {[t.tool_name for t in mcp_tools]}")
else:
    logger.warning("⚠️ No MCP tools loaded — Gateway features unavailable")

# ---------------------------------------------------------------------------
# Agent + A2A Server setup
# ---------------------------------------------------------------------------
logger.info("Creating Cost Agent...")

# Build tool list: local tools + MCP tools from gateway
agent_tools = [compute_manufacturing_cost, compute_cost_batch] + mcp_tools

# Build dynamic system prompt based on MCP availability
has_mcp = bool(mcp_tools)
final_system_prompt = SYSTEM_PROMPT + (_MCP_AVAILABLE_PROMPT if has_mcp else _MCP_UNAVAILABLE_PROMPT) + _SYSTEM_PROMPT_FOOTER
if has_mcp:
    logger.info("System prompt: MCP tools AVAILABLE — agent will fetch cost parameters from MCP")
else:
    logger.warning("System prompt: MCP tools NOT available — agent will use default cost parameters")

# ---------------------------------------------------------------------------
# Conversation manager that truncates oversized MCP tool results
# ---------------------------------------------------------------------------
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.models import BedrockModel

MAX_TOOL_RESULT_CHARS = 5000  # MCP get_all_cost_parameters returns ~3KB, need full data


class TruncatingConversationManager(SlidingWindowConversationManager):
    """Sliding window manager that truncates large tool results.

    The MCP get_all_cost_parameters tool returns all 8 categories of cost data,
    easily exceeding 3KB. If the LLM echoes that in its response, the A2A
    payload overflows AgentCore's limit → -32603.
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


conversation_manager = TruncatingConversationManager(window_size=6, per_turn=True)

model = BedrockModel(
    model_id=MODEL_ID,
    max_tokens=2048,  # Cost agent needs room for MCP tool calls + cost breakdown response
)

agent = Agent(
    name="Cost Agent",
    description=(
        "Estimates manufacturing costs for car body design variants. "
        "Connects to two MCP servers for cost data: internal parameters "
        "(material costs, stamping rates, tooling costs) and external data "
        "(market pricing, supplier quotes, historical benchmarks). "
        "Computes material, stamping, tooling, and assembly cost components."
    ),
    system_prompt=final_system_prompt,
    model=model,
    tools=agent_tools,
    conversation_manager=conversation_manager,
    concurrent_invocation_mode=ConcurrentInvocationMode.UNSAFE_REENTRANT,
)

# Clear conversation history before each A2A request so the agent is stateless
# across requests. Without this, messages accumulate and context grows unbounded.
def _clear_messages(event: BeforeInvocationEvent):
    event.agent.messages.clear()

agent.add_hook(_clear_messages)
logger.info("✅ Cost Agent created")

# OTel guard — patch from_converse to handle missing 'output' key
# MaxTokensReachedException or throttling returns a response without 'output',
# causing OTel instrumentation to crash with KeyError: 'output'
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
        "agent": "cost_agent",
        "version": "1.1.0",
        "features": [
            "a2a_protocol",
            "mcp_gateway_jwt_auth",
            "material_cost_estimation",
            "stamping_cost_estimation",
            "tooling_cost_estimation",
            "assembly_cost_estimation",
        ],
        "mcp_mode": "gateway",
        "mcp_gateway": {
            "url": GATEWAY_URL or "not configured",
            "jwt_auth": bool(GATEWAY_URL and GATEWAY_TOKEN_URL),
            "credentials_secret": MCP_GATEWAY_SECRET_NAME or "not configured",
        },
        "mcp_tools_loaded": len(mcp_tools),
        "materials": list(MATERIAL_DENSITIES.keys()),
    }


_a2a_app = a2a_server.to_fastapi_app()


# ---------------------------------------------------------------------------
# ASGI middleware: wrap non-JSON-RPC payloads arriving at POST /
# ---------------------------------------------------------------------------
import uuid as _uuid
from starlette.types import ASGIApp, Receive, Scope, Send


MAX_RESPONSE_BYTES = 60_000  # Synchronous A2A — safe to allow larger payloads


def _truncate_a2a_response(body: bytes) -> bytes:
    """Truncate an A2A JSON-RPC response to fit AgentCore's payload limit.

    Strategy:
    1. Strip task.history (biggest contributor)
    2. If artifacts exist, nuke status.message (redundant)
    3. Keep only first artifact, trim text to 2000 chars
    4. Nuclear: raw byte truncation as last resort
    """
    if len(body) <= MAX_RESPONSE_BYTES:
        return body

    try:
        data = json.loads(body)
    except (json.JSONDecodeError, Exception):
        return body[:MAX_RESPONSE_BYTES]

    result = data.get("result", {})
    if not isinstance(result, dict):
        return json.dumps(data).encode()[:MAX_RESPONSE_BYTES]

    task_obj = result.get("task", result) if "task" in result else result

    # 1. Strip history
    task_obj.pop("history", None)
    result.pop("history", None)

    # 2. If artifacts exist, nuke status.message
    artifacts = task_obj.get("artifacts", [])
    if artifacts:
        status = task_obj.get("status", {})
        if isinstance(status, dict):
            status.pop("message", None)

    # 3. Merge text from ALL artifacts, then collapse to single artifact
    if artifacts:
        all_text = ""
        for artifact in artifacts:
            for part in artifact.get("parts", []):
                if isinstance(part, dict) and part.get("text"):
                    all_text += part["text"]
        trimmed = all_text.strip()[:2000]
        task_obj["artifacts"] = [{
            "artifactId": artifacts[0].get("artifactId", ""),
            "name": artifacts[0].get("name", "agent_response"),
            "parts": [{"kind": "text", "text": trimmed}],
        }]
        artifacts = task_obj["artifacts"]

    # Also trim status.message if no artifacts
    if not artifacts:
        status = task_obj.get("status", {})
        if isinstance(status, dict):
            msg = status.get("message", {})
            if isinstance(msg, dict) and "parts" in msg:
                all_text = ""
                for part in msg["parts"]:
                    if isinstance(part, dict) and "text" in part:
                        all_text += part["text"] + "\n"
                msg["parts"] = [{"kind": "text", "text": all_text.strip()[:2000]}]

    truncated = json.dumps(data, separators=(",", ":")).encode()

    # 4. Nuclear: raw byte truncation
    if len(truncated) > MAX_RESPONSE_BYTES:
        truncated = truncated[:MAX_RESPONSE_BYTES]

    logger.info(f"Truncated A2A response from {len(body)} to {len(truncated)} bytes")
    return truncated


class A2APayloadNormalizer:
    """ASGI middleware that normalizes A2A payloads in both directions.

    Inbound: wraps raw payloads into A2A JSON-RPC envelopes if needed.
    Outbound: truncates oversized response payloads to prevent -32603 errors.
    """

    def __init__(self, wrapped_app: ASGIApp):
        self.app = wrapped_app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http" or scope.get("method", "") != "POST":
            await self.app(scope, receive, send)
            return

        # --- INBOUND: collect and normalize request body ---
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


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
import atexit


def _cleanup_mcp():
    global _mcp_client
    try:
        if _mcp_client:
            _mcp_client.__exit__(None, None, None)
            logger.info("MCP client closed")
    except Exception as e:
        logger.warning(f"MCP cleanup warning: {e}")


atexit.register(_cleanup_mcp)


if __name__ == "__main__":
    # AgentCore containers require an all-interface bind; ingress is protected
    # by the Runtime JWT authorizer and AgentCore network boundary.
    host, port = "0.0.0.0", AGENT_PORT  # nosec B104
    print()
    print("=" * 60)
    print("Cost Agent — Car Design Space Explorer")
    print(f"  Agent Card: http://{host}:{port}/.well-known/agent-card.json")
    print(f"  MCP Mode: Gateway (JWT auth)")
    print(f"  Gateway URL: {GATEWAY_URL or 'NOT CONFIGURED'}")
    print("=" * 60)
    uvicorn.run(app, host=host, port=port)
