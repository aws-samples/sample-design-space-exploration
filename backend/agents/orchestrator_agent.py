#!/usr/bin/env python3
"""
Orchestrator Agent — Strands A2A Server for multi-agent coordination.

Central coordinator that receives engineer queries via natural language,
discovers specialist agents (Aero, Structural, Cost) on Bedrock AgentCore,
routes evaluation tasks via A2A protocol, synthesizes results, computes
Pareto fronts, and returns unified responses.

Architecture:
- Strands Agent framework with @tool decorated functions
- A2AClientToolProvider for dynamic agent discovery and A2A communication
- Deployed to Bedrock AgentCore Runtime
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from itertools import product as cartesian_product

import boto3
import uvicorn
from fastapi import FastAPI
from strands import Agent, tool
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.agent.agent import ConcurrentInvocationMode
from strands.hooks.events import BeforeInvocationEvent
from strands.models import BedrockModel
from strands.multiagent.a2a import A2AServer
from strands_tools.a2a_client import A2AClientToolProvider
from strands.types.tools import ToolResult, ToolUse

# A2A TaskStore — strip history from responses to prevent payload bloat
from a2a.server.tasks import TaskStore, InMemoryTaskStore
from a2a.server.context import ServerCallContext
from a2a.types import Task as A2ATask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AGENT_PORT = int(os.environ.get("ORCHESTRATOR_AGENT_PORT", "9000"))
MODEL_ID = os.environ.get("MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
S3_MODEL_BUCKET = os.environ["S3_MODEL_BUCKET"]
AERO_AGENT_URL = os.environ.get("AERO_AGENT_URL", "")
STRUCTURAL_AGENT_URL = os.environ.get("STRUCTURAL_AGENT_URL", "")
COST_AGENT_URL = os.environ.get("COST_AGENT_URL", "")
GEOMETRY_AGENT_URL = os.environ.get("GEOMETRY_AGENT_URL", "")


# ---------------------------------------------------------------------------
# Agent endpoint tool — endpoints are resolved and baked in during deployment
# ---------------------------------------------------------------------------

_CONFIGURED_AGENT_URLS = {
    name: url
    for name, url in {
        "aero": AERO_AGENT_URL,
        "structural": STRUCTURAL_AGENT_URL,
        "cost": COST_AGENT_URL,
        "geometry": GEOMETRY_AGENT_URL,
    }.items()
    if url
}


@tool
def discover_all_agents() -> str:
    """Return configured Car Design specialist A2A endpoint URLs.

    Runtime discovery is deliberately performed by the deployment principal,
    not by this runtime role. The returned URLs are ready for
    ``a2a_send_message``.
    """
    missing = sorted(
        {"aero", "structural", "cost", "geometry"} - _CONFIGURED_AGENT_URLS.keys()
    )
    if missing:
        return json.dumps({
            "status": "error",
            "error_message": "Missing configured specialist endpoints",
            "missing_agents": missing,
        }, indent=2)
    return json.dumps({
        "agents": _CONFIGURED_AGENT_URLS,
        "count": len(_CONFIGURED_AGENT_URLS),
        "source": "deployment_config",
    }, indent=2)


@tool
def compute_pareto_front(variants_json: str) -> str:
    """Compute the Pareto front from a set of evaluated design variants.

    Finds non-dominated solutions on the drag coefficient (Cd) vs total
    manufacturing cost plane. A variant is Pareto-optimal if no other
    variant is strictly better on both objectives.

    Args:
        variants_json: JSON array of objects with at least:
            variant_id, drag_coefficient (or cd), total_cost.

    Returns:
        JSON with pareto_front array and dominated variants.
    """
    try:
        variants = json.loads(variants_json)
        for v in variants:
            if "cd" in v and "drag_coefficient" not in v:
                v["drag_coefficient"] = v["cd"]

        valid = [v for v in variants if "drag_coefficient" in v and "total_cost" in v]

        pareto = []
        for candidate in valid:
            dominated = False
            for other in valid:
                if other is candidate:
                    continue
                if (other["drag_coefficient"] <= candidate["drag_coefficient"]
                        and other["total_cost"] <= candidate["total_cost"]
                        and (other["drag_coefficient"] < candidate["drag_coefficient"]
                             or other["total_cost"] < candidate["total_cost"])):
                    dominated = True
                    break
            if not dominated:
                pareto.append(candidate)

        pareto.sort(key=lambda v: v["drag_coefficient"])

        return json.dumps({
            "pareto_front": pareto,
            "pareto_count": len(pareto),
            "total_variants": len(valid),
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error_message": str(e)})


@tool
def rank_variants(variants_json: str, sort_by: str = "cd", ascending: bool = True) -> str:
    """Rank design variants by a specified metric.

    Args:
        variants_json: JSON array of variant result objects.
        sort_by: Field to sort by — "cd", "total_cost", "weight_kg",
            "feasibility_score", "stiffness_score".
        ascending: Sort ascending (True) or descending (False).

    Returns:
        JSON array of ranked variants with rank numbers.
    """
    try:
        variants = json.loads(variants_json)
        field_map = {
            "cd": "drag_coefficient",
            "drag_coefficient": "drag_coefficient",
            "total_cost": "total_cost",
            "cost": "total_cost",
            "weight_kg": "weight_kg",
            "weight": "weight_kg",
            "feasibility_score": "feasibility_score",
            "stiffness_score": "stiffness_score",
        }
        field = field_map.get(sort_by, sort_by)

        sortable = [v for v in variants if field in v]
        sortable.sort(key=lambda v: v[field], reverse=not ascending)

        for i, v in enumerate(sortable, 1):
            v["rank"] = i

        return json.dumps(sortable, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error_message": str(e)})


@tool
def generate_sweep_variants(parameter_ranges_json: str) -> str:
    """Generate design variant parameter combinations for a parameter sweep.

    Creates the Cartesian product of all parameter ranges.

    Args:
        parameter_ranges_json: JSON object mapping parameter names to
            [min, max, steps] arrays. Example:
            {"ride_height": [0.04, 0.08, 3], "diffuser_angle": [5, 15, 3]}

    Returns:
        JSON array of variant parameter combinations with generated IDs.
    """
    try:
        ranges = json.loads(parameter_ranges_json)
        param_names = list(ranges.keys())
        param_values = []
        for name in param_names:
            min_val, max_val, steps = ranges[name]
            if steps <= 1:
                param_values.append([min_val])
            else:
                step_size = (max_val - min_val) / (steps - 1)
                param_values.append([round(min_val + i * step_size, 6) for i in range(steps)])

        combinations = list(cartesian_product(*param_values))
        variants = []
        for i, combo in enumerate(combinations):
            variant = {
                "variant_id": f"sweep_{i:04d}",
                "parameters": {name: val for name, val in zip(param_names, combo)},
            }
            variants.append(variant)

        return json.dumps({
            "variants": variants,
            "count": len(variants),
            "parameter_names": param_names,
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error_message": str(e)})


@tool
def system_health_check() -> str:
    """Check the orchestrator's health and list available tools and agents.

    Use this tool when the user asks about system status, available agents,
    or when you want to verify connectivity before making A2A calls.
    This does NOT call any specialist agents — it only checks local state.

    Returns:
        JSON with system health info, available tools, and discovered agents.
    """
    try:
        import time as _time
        info = {
            "status": "healthy",
            "agent": "orchestrator",
            "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
            "aws_region": AWS_REGION,
            "model_id": MODEL_ID,
        }
        info["configured_agents"] = [
            {"name": name, "status": "configured"}
            for name in sorted(_CONFIGURED_AGENT_URLS)
        ]
        info["agent_count"] = len(_CONFIGURED_AGENT_URLS)

        return json.dumps(info, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error_message": str(e)})


def _ensure_s3_uri(path: str) -> str:
    """Convert any geometry path to an s3:// URI. Used internally by tools."""
    import re
    if path.startswith("s3://"):
        return path
    # Try presigned URL conversion
    match = re.search(r'([a-zA-Z0-9._-]+)\.s3\.[a-zA-Z0-9.-]+\.amazonaws\.com/([^\s"\'?]+\.stl)', path)
    if match:
        return f"s3://{match.group(1)}/{match.group(2)}"
    # Try s3:// extraction from mixed text
    s3_match = re.search(r's3://[a-zA-Z0-9._-]+/[^\s"\'?\]\)]+\.stl', path)
    if s3_match:
        return s3_match.group(0)
    return path


@tool
def generate_stl_viewer_tag(s3_uri: str) -> str:
    """Generate an [STL] tag with an s3:// URI for the frontend 3D viewer.

    Call this after receiving an stl_s3_uri from the Geometry Agent.
    Wraps the s3:// URI in the [STL] tag. The Lambda handler will convert
    it to a presigned URL before sending to the frontend.

    Args:
        s3_uri: S3 URI of the STL file (e.g. s3://bucket/geometries/file.stl)

    Returns:
        The [STL]s3_uri[/STL] tag to include in your response.
    """
    try:
        s3_uri = _ensure_s3_uri(s3_uri)
        return f"[STL]{s3_uri}[/STL]"
    except Exception as e:
        return json.dumps({"status": "error", "error_message": str(e)})


@tool
def generate_image_viewer_tags(s3_uris_text: str) -> str:
    """Generate [IMAGE] tags with s3:// URIs for visualization images.

    The Lambda handler will convert s3:// URIs to presigned URLs before
    sending to the frontend.

    Args:
        s3_uris_text: Comma-separated S3 URIs of PNG files.

    Returns:
        [IMAGE] tags for each image.
    """
    try:
        import re
        uris = re.findall(r's3://[a-zA-Z0-9._-]+/[^\s,\]\)]+', s3_uris_text)
        if not uris:
            return json.dumps({"status": "error", "error_message": "No s3:// URIs found in input"})

        tags = []
        for uri in uris:
            tags.append(f"[IMAGE]{uri}[/IMAGE]")
        return "\n".join(tags)
    except Exception as e:
        return json.dumps({"status": "error", "error_message": str(e)})


@tool
def extract_geometry_s3_uri(agent_response_text: str) -> str:
    """Extract the s3:// URI from a Geometry Agent response.

    Use this BEFORE sending a geometry path to any specialist agent
    (Aero, Structural, Cost). It extracts the clean s3:// URI from
    the Geometry Agent's response text, ensuring you never accidentally
    send a presigned HTTPS URL to downstream agents.

    Args:
        agent_response_text: The raw text response from the Geometry Agent.

    Returns:
        JSON with the extracted s3_uri ready for use with specialist agents.
    """
    import re
    # Try to find s3:// URI directly in the text
    s3_match = re.search(r's3://[a-zA-Z0-9._-]+/[^\s"\'\]\)]+\.stl', agent_response_text)
    if s3_match:
        s3_uri = s3_match.group(0)
        logger.info(f"Extracted s3 URI from geometry response: {s3_uri}")
        return json.dumps({"s3_uri": s3_uri, "status": "success"})

    # Try to parse as JSON and extract stl_s3_uri field
    try:
        data = json.loads(agent_response_text)
        if isinstance(data, dict):
            uri = data.get("stl_s3_uri") or data.get("s3_uri") or data.get("modified_stl_s3_uri", "")
            if uri and uri.startswith("s3://"):
                logger.info(f"Extracted s3 URI from JSON field: {uri}")
                return json.dumps({"s3_uri": uri, "status": "success"})
    except (json.JSONDecodeError, Exception):
        pass

    # Try to convert a presigned HTTPS URL to s3:// URI
    https_match = re.search(r'https://([a-zA-Z0-9._-]+)\.s3\.[a-zA-Z0-9.-]+\.amazonaws\.com/([^\s"\'?\]]+)', agent_response_text)
    if https_match:
        bucket = https_match.group(1)
        key = https_match.group(2)
        s3_uri = f"s3://{bucket}/{key}"
        logger.info(f"Converted presigned URL to s3 URI: {s3_uri}")
        return json.dumps({"s3_uri": s3_uri, "status": "success"})

    return json.dumps({
        "status": "error",
        "error_message": "Could not extract s3:// URI from geometry response. "
                         "Ensure the Geometry Agent returned an stl_s3_uri field.",
    })


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = f"""You are the Orchestrator for the Car Design Space Explorer. You coordinate specialist agents (Aero, Structural, Cost, Geometry) via A2A protocol to answer automotive engineer queries.

You have conversation memory — you can reference previous interactions. If the user says "the above variant" or "this design", check your conversation history for the variant ID or s3:// URI from the previous response.

## Agent Discovery
1. `discover_all_agents()` → configured agent name → URL map
2. `a2a_send_message(...)` → send request directly using the URL from step 1
CRITICAL: Do NOT call a2a_discover_agent — it is redundant. discover_all_agents already returns ready-to-use URLs. Go directly from discover_all_agents → a2a_send_message.

## Routing
- Aero queries (Cd, drag, compare variants, rank) → Aero Agent. For rankings tell it: "Use query_variants to find the top N by Cd"
- Cost queries → Structural Agent (geometry metrics) → Cost Agent (with those metrics)
- Structural queries (weight, stiffness, feasibility) → Structural Agent
- Geometry modifications (mirrors, spoiler, bonnet, diffuser, parametric design) → Geometry Agent
- Parametric car generation → Geometry Agent. Tell it: "Use generate_car_design with parameters: {{ride_height, rear_slant, diffuser_angle, front_overhang, boat_tail_angle}}". If user mentions a vehicle type (sedan, sports car, SUV, hatchback, mini SUV), include segment parameter (sedan/sport/suv/hatchback/mini_suv).
  CRITICAL: If the user does NOT provide specific parameter values, use these defaults by segment:
  - sedan:   ride_height=0.05, rear_slant=15, diffuser_angle=5,  front_overhang=0.85, boat_tail_angle=5
  - sport:   ride_height=0.04, rear_slant=30, diffuser_angle=12, front_overhang=0.75, boat_tail_angle=10
  - hatchback: ride_height=0.05, rear_slant=35, diffuser_angle=5, front_overhang=0.70, boat_tail_angle=5
  - suv:     ride_height=0.08, rear_slant=10, diffuser_angle=3,  front_overhang=0.90, boat_tail_angle=3
  - mini_suv: ride_height=0.07, rear_slant=18, diffuser_angle=5, front_overhang=0.75, boat_tail_angle=5
  - default (no segment mentioned): use sedan defaults above with segment=sedan
  NEVER ask the user for parameters — always use defaults if not provided. Just generate the design.
- Surface/pressure visualization (pressure, heatmap, cpavg, cfxavg) → Aero Agent get_surface_data
- Slices/flow visualization (slices, airflow, velocity, flow field, cross-section) → Aero Agent get_slices_data
- Simple queries (status, hello) → system_health_check locally, no A2A

## Chaining Rules
Chain automatically — NEVER ask the user for internal data or expose agent internals.

### Cost Estimation (most complex chain — follow exactly):
1. `discover_all_agents()` to get URLs
2. `a2a_send_message` to Structural Agent: "Compute geometry metrics for variant {{variant_id}}"
3. Parse the structural response JSON — extract surface_area_m2, vertex_count, curvature_variation, surface_patch_count, max_draw_depth_mm, has_undercuts
4. `a2a_send_message` to Cost Agent: "Estimate manufacturing cost for variant {{variant_id}} with metrics: surface_area_m2={{sa}}, vertex_count={{vc}}, curvature_variation={{cv}}, surface_patch_count={{spc}}, max_draw_depth_mm={{mdd}}, has_undercuts={{hu}}, material={{mat}}"
5. Present the cost breakdown to the user

### Other chains:
- Structural: "Compute geometry metrics and evaluate structural feasibility for {{variant_id}} using {{material}}"
- Geometry + eval: Geometry Agent → extract_geometry_s3_uri → chain to Aero/Structural/Cost with s3:// URI
- Parametric: Geometry Agent (generate_car_design) → extract_geometry_s3_uri → generate_stl_viewer_tag → eval if requested
- Surface/Slices visualization: Aero Agent → extract image_s3_uris from response → generate_image_viewer_tags → include [IMAGE] tags in response

CRITICAL: Geometry paths sent to agents MUST be s3:// URIs, never https://. Always call extract_geometry_s3_uri on Geometry Agent responses before chaining.

## Geometry Paths
WindsorML variants: s3://{S3_MODEL_BUCKET}/geometries/run_N.stl (N = 1-355)
Parametric variants: s3://{S3_MODEL_BUCKET}/geometries/parametric_XXXXXXXX.stl (same bucket/prefix pattern)
Uploaded variants: s3://{S3_MODEL_BUCKET}/geometries/uploaded_TIMESTAMP_FILENAME.stl (user-uploaded STL files)
ALL variant geometry paths follow the pattern: s3://{S3_MODEL_BUCKET}/geometries/{{variant_id}}.stl
When a user references any variant ID (run_125, parametric_84b1dd97, etc.), construct the s3:// path directly — do NOT ask the user for it.
When a user provides a full s3:// URI (e.g. for an uploaded geometry), use it DIRECTLY — do NOT ask for a variant ID. Extract a short variant_id from the filename (e.g. "uploaded_1775042243_parametric_00665b52" from the s3 key) and pass the full s3:// path as geometry_path to all agents.
For full pipeline analysis (aero + structural + cost), chain all three agents using the same s3:// geometry_path:
1. Aero Agent: "Evaluate aero KPIs for variant {{variant_id}} with geometry_path {{s3_uri}}"
2. Structural Agent: "Compute geometry metrics for variant {{variant_id}} with geometry_path {{s3_uri}}"
3. Cost Agent: use structural metrics as usual
For surface/heatmap requests, pass geometry_path directly: "Get surface pressure data for variant run_0 with geometry_path s3://{S3_MODEL_BUCKET}/geometries/run_0.stl"
For slices/flow requests, pass geometry_path directly: "Get velocity flow slices for variant run_0 with geometry_path s3://{S3_MODEL_BUCKET}/geometries/run_0.stl"

## CRITICAL: STL Geometry Availability Constraint
Only run_0 through run_9 have STL geometry files in S3. All other variants (run_10 to run_355, parametric_*, uploaded_*) do NOT have pre-seeded STL files.
Surface pressure and slice/flow visualizations require the STL file to run live inference — they will ALWAYS FAIL for run_10 and above.
Aero KPI queries (Cd, Cs, Cl, Cmy) for run_10 to run_355 are served from DynamoDB cache and work fine without an STL file.
NEVER suggest surface heatmap or slice visualization for any variant outside run_0 to run_9.

## Local Tools
- discover_all_agents, compute_pareto_front, rank_variants, generate_sweep_variants, system_health_check
- generate_stl_viewer_tag(s3_uri) → [STL]presigned_url[/STL] tag
- generate_image_viewer_tags(s3_uris_text) → [IMAGE]presigned_url[/IMAGE] tags for images
- extract_geometry_s3_uri(text) → clean s3:// URI from agent response

## A2A Tools
- a2a_discover_agent(url) — register agent before messaging
- a2a_send_message — send request to registered agent

## Response Rules
- Text under 1500 chars (excluding [STL]/[IMAGE] tags). Parametric without eval: under 500 chars.
- Max 10 tool calls per request. On failure, report error and move on — no retries.
- EFFICIENCY: Most queries need only 3-4 tool calls: discover_all_agents → a2a_send_message → format response. Skip a2a_discover_agent if you already have the URL from discover_all_agents.
- For ranking/comparison queries, ALWAYS delegate to the Aero Agent with "Use query_variants" — do NOT call evaluate_aero_kpi individually for each variant.
- Never send open-ended requests ("list all", "show everything") to agents. Always specify variant IDs, max 5 per call.
- Summarize results as: 1 sentence + compact markdown table (max 5-10 rows) + 1 short recommendation.
- No emoji, no JSON dumps, no per-variant paragraphs, no internal agent/tool names exposed to user.
- Pass through [IMAGE]url[/IMAGE] and [STL]url[/STL] tags from agents exactly as-is.
- End responses with 2-3 suggested next actions relevant to what was just done.
- IMPORTANT: Suggested next actions MUST include the specific variant ID so the user can copy-paste them directly. Example: "Add a rear spoiler to parametric_abc123 to reduce lift" NOT "Add a rear spoiler to reduce lift".
- IMPORTANT: Only suggest actions that work reliably as standalone queries. Safe suggestions:
  * Aero KPI evaluation for any specific variant (served from DynamoDB, always fast)
  * Surface pressure or slice visualization ONLY for run_0 through run_9 — never for run_10 or higher
  * Compare specific variants by drag coefficient or lift-to-drag ratio
  * Generate a new parametric design with different parameters
  * Structural feasibility evaluation for a specific variant
- Do NOT suggest: surface/slice visualization for any variant outside run_0 to run_9, cost estimation (unreliable chain), flow slices for parametric variants, geometry modifications on parametric variants, Pareto front analysis.
"""

# ---------------------------------------------------------------------------
# A2A Client Tool Provider for agent-to-agent communication
# ---------------------------------------------------------------------------
logger.info("Creating A2AClientToolProvider for agent discovery...")

import httpx
import base64

_OAUTH_SECRET_NAME = os.environ.get("OAUTH_SECRET_NAME", "")


class _RefreshingBearerAuth(httpx.Auth):
    """httpx.Auth that auto-refreshes a Cognito client_credentials token before expiry.

    Called by httpx on every outbound request, so the token is always current
    even when the container runs for longer than the 60-minute Cognito token TTL.
    """

    def __init__(self, secret_name: str, region: str) -> None:
        self._secret_name = secret_name
        self._region = region
        self._token: str = ""
        self._expiry: float = 0.0

    def _fetch_token(self) -> str:
        if self._token and time.time() < self._expiry:
            return self._token
        try:
            sm = boto3.client("secretsmanager", region_name=self._region)
            config = json.loads(
                sm.get_secret_value(SecretId=self._secret_name)["SecretString"]
            )
            auth_b64 = base64.b64encode(
                f"{config['client_id']}:{config['client_secret']}".encode()
            ).decode()
            resp = httpx.post(
                config["token_url"],
                headers={
                    "Authorization": f"Basic {auth_b64}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={"grant_type": "client_credentials", "scope": config.get("scope", "")},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            self._token = data["access_token"]
            expires_in = data.get("expires_in", 3600)
            self._expiry = time.time() + expires_in - 300  # 5-min buffer before expiry
            logger.info("✅ OAuth token refreshed for A2A communication")
        except Exception as e:
            logger.warning(f"OAuth token refresh failed: {e}")
        return self._token

    def auth_flow(self, request: httpx.Request):
        token = self._fetch_token()
        if token:
            request.headers["Authorization"] = f"Bearer {token}"
        yield request


_a2a_httpx_args: dict = {"timeout": 180.0}
if _OAUTH_SECRET_NAME:
    _a2a_httpx_args["auth"] = _RefreshingBearerAuth(_OAUTH_SECRET_NAME, AWS_REGION)
    logger.info(f"✅ Refreshing OAuth auth configured for A2A (secret: {_OAUTH_SECRET_NAME})")
else:
    logger.info("No OAUTH_SECRET_NAME set — running without OAuth (local mode)")

a2a_provider = A2AClientToolProvider(
    known_agent_urls=[],
    httpx_client_args=_a2a_httpx_args,
)
logger.info("✅ A2AClientToolProvider configured")

# Combine all tools
a2a_tools = a2a_provider.tools
all_tools = a2a_tools + [
    discover_all_agents,
    compute_pareto_front,
    rank_variants,
    generate_sweep_variants,
    system_health_check,
    generate_stl_viewer_tag,
    generate_image_viewer_tags,
    extract_geometry_s3_uri,
]
logger.info(f"Total tools available: {len(all_tools)}")

# ---------------------------------------------------------------------------
# Agent + A2A Server setup
# ---------------------------------------------------------------------------
logger.info("Creating Orchestrator Agent...")

# ---------------------------------------------------------------------------
# Custom conversation manager that truncates oversized tool results
# ---------------------------------------------------------------------------
MAX_TOOL_RESULT_CHARS = 4000  # Keep tool results lean — model only needs top N results, not all 355 variants

# Thread-local store: persists the original user query for the lifetime of one
# HTTP request across multiple converse_stream calls and any message repairs.
# Reset by _clear_messages (BeforeInvocationEvent) at the start of each invocation.
_request_context = threading.local()


def _repair_tool_mismatches(messages: list) -> None:
    """Repair toolResult/toolUse mismatches in the message history.

    Bedrock's ConverseStream API enforces strict pairing:
    1. Every toolResult must have a matching toolUse in the previous assistant turn
    2. Every toolUse must have a matching toolResult in the next user turn
    3. Messages must strictly alternate user→assistant→user

    Mismatches happen when:
    - Strands' recover_message_on_max_tokens_reached replaces toolUse with text
    - The sliding window trims messages and breaks toolUse/toolResult pairs
    - The last assistant message has toolUse blocks but no user response yet
    - _convert_prompt_to_messages injects synthetic user messages creating user→user

    Fix: merge consecutive same-role messages, then scan adjacent assistant→user
    pairs and repair both directions.
    """
    # --- FIX 0: Merge consecutive same-role messages ---
    i = 0
    while i < len(messages) - 1:
        if (isinstance(messages[i], dict) and isinstance(messages[i + 1], dict)
                and messages[i].get("role") == messages[i + 1].get("role")
                and messages[i].get("role") in ("user", "assistant")):
            content_a = messages[i].get("content", [])
            content_b = messages[i + 1].get("content", [])
            if isinstance(content_a, list) and isinstance(content_b, list):
                messages[i]["content"] = content_a + content_b
            elif isinstance(content_b, list):
                messages[i]["content"] = content_b
            messages.pop(i + 1)
            logger.warning(f"Merged consecutive {messages[i]['role']} messages at index {i}")
        else:
            i += 1

    for idx in range(len(messages) - 1):
        msg = messages[idx]
        next_msg = messages[idx + 1] if idx + 1 < len(messages) else None
        if (not isinstance(msg, dict) or msg.get("role") != "assistant"
                or not next_msg or not isinstance(next_msg, dict)
                or next_msg.get("role") != "user"):
            continue

        # Collect toolUse IDs from the assistant turn
        tool_use_ids = set()
        assistant_content = msg.get("content", [])
        if isinstance(assistant_content, list):
            for block in assistant_content:
                if isinstance(block, dict) and block.get("toolUse"):
                    tu_id = block["toolUse"].get("toolUseId", "")
                    if tu_id:
                        tool_use_ids.add(tu_id)

        # Collect toolResult IDs from the user turn
        tool_result_ids = set()
        user_content = next_msg.get("content", [])
        if isinstance(user_content, list):
            for block in user_content:
                if isinstance(block, dict) and block.get("toolResult"):
                    tr_id = block["toolResult"].get("toolUseId", "")
                    if tr_id:
                        tool_result_ids.add(tr_id)

        # --- FIX 1: Remove orphaned toolResults (no matching toolUse) ---
        if isinstance(user_content, list):
            cleaned = []
            for block in user_content:
                if isinstance(block, dict) and block.get("toolResult"):
                    tr_id = block["toolResult"].get("toolUseId", "")
                    if tr_id and tr_id not in tool_use_ids:
                        logger.warning(
                            f"Removing orphaned toolResult {tr_id} — "
                            f"no matching toolUse in previous assistant turn"
                        )
                        continue
                cleaned.append(block)
            if len(cleaned) != len(user_content):
                # Preserve any real text blocks even when all toolResults are dropped.
                # Falling back to "(tool results removed)" risks nuking the original
                # user query if the message was merged with orphaned toolResult blocks.
                has_real_text = any(
                    isinstance(b, dict) and "text" in b and b["text"] not in (
                        "(tool results removed)", "(deduplicated)",
                        "(result unavailable — conversation was trimmed)",
                        "(result unavailable — repaired)",
                    )
                    for b in cleaned
                )
                next_msg["content"] = cleaned if (cleaned and has_real_text) else (
                    cleaned if cleaned else [{"text": "(tool results removed)"}]
                )

        # --- FIX 2: Add missing toolResults for orphaned toolUse blocks ---
        missing_ids = tool_use_ids - tool_result_ids
        if missing_ids:
            current_content = next_msg.get("content", [])
            if not isinstance(current_content, list):
                current_content = []
            for missing_id in missing_ids:
                logger.warning(
                    f"Adding dummy toolResult for orphaned toolUse {missing_id} — "
                    f"no matching toolResult in next user turn"
                )
                current_content.append({
                    "toolResult": {
                        "toolUseId": missing_id,
                        "status": "error",
                        "content": [{"text": "(result unavailable — conversation was trimmed)"}],
                    }
                })
            next_msg["content"] = current_content

    # --- FIX 3: Handle trailing assistant message with toolUse but no user response ---
    if messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "assistant":
        last_content = messages[-1].get("content", [])
        trailing_tool_ids = []
        if isinstance(last_content, list):
            for block in last_content:
                if isinstance(block, dict) and block.get("toolUse"):
                    tu_id = block["toolUse"].get("toolUseId", "")
                    if tu_id:
                        trailing_tool_ids.append(tu_id)
        if trailing_tool_ids:
            # Add a synthetic user message with dummy toolResults
            dummy_results = []
            for tid in trailing_tool_ids:
                logger.warning(
                    f"Adding trailing dummy toolResult for {tid} — "
                    f"assistant message ends with toolUse but no user response"
                )
                dummy_results.append({
                    "toolResult": {
                        "toolUseId": tid,
                        "status": "error",
                        "content": [{"text": "(result unavailable — conversation was trimmed)"}],
                    }
                })
            messages.append({"role": "user", "content": dummy_results})


class TruncatingConversationManager(SlidingWindowConversationManager):
    """Sliding window manager that also truncates large tool results.

    The aero agent can return 50+ variant Cd values in a single A2A response,
    easily exceeding 10KB. If the model tries to echo that in its final answer,
    the A2A response payload overflows AgentCore's limit → -32603.

    This manager scans tool results before each model call and truncates any
    that exceed MAX_TOOL_RESULT_CHARS, replacing the tail with a summary note.
    """

    def apply_management(self, agent):
        messages = agent.messages

        # ---- STEP 0: Capture original user query ----
        # Stored in thread-local so the converse_stream fallback can use it.
        if not getattr(_request_context, "original_query", None):
            for msg in messages:
                if not isinstance(msg, dict) or msg.get("role") != "user":
                    continue
                content = msg.get("content", [])
                if not isinstance(content, list):
                    continue
                has_tool_result = any(
                    isinstance(b, dict) and "toolResult" in b for b in content
                )
                if has_tool_result:
                    continue
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        _request_context.original_query = block["text"]
                        break
                if getattr(_request_context, "original_query", None):
                    break

        # ---- STEP 1: Truncate oversized tool results ----
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for i, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                if block.get("toolResult"):
                    tr = block["toolResult"]
                    tr_content = tr.get("content", [])
                    total_len = 0
                    for part in tr_content:
                        if isinstance(part, dict) and "text" in part:
                            total_len += len(part["text"])
                    if total_len > MAX_TOOL_RESULT_CHARS:
                        truncated_parts = []
                        remaining = MAX_TOOL_RESULT_CHARS
                        for part in tr_content:
                            if isinstance(part, dict) and "text" in part:
                                text = part["text"]
                                if remaining > 0:
                                    truncated_parts.append({"text": text[:remaining]})
                                    remaining -= len(text)
                            else:
                                truncated_parts.append(part)
                        truncated_parts.append({
                            "text": f"\n\n[TRUNCATED: original was {total_len} chars. "
                                    f"Summarize the data you have — do NOT request more.]"
                        })
                        tr["content"] = truncated_parts
                        logger.info(
                            f"Truncated tool result from {total_len} to ~{MAX_TOOL_RESULT_CHARS} chars"
                        )

        # ---- STEP 2: Apply sliding window ----
        # Let the SDK handle message pairing natively — no manual repairs.
        super().apply_management(agent)


# window_size=8 keeps enough context for multi-agent chains (parametric design → aero
# eval requires 8+ tool calls). With window_size=4 the model lost the original user
# query mid-chain and returned a generic welcome instead of chaining to aero.
conversation_manager = TruncatingConversationManager(window_size=10, per_turn=True)

# Use BedrockModel with max_tokens to control output size.
# 2048 is enough for multi-agent chains (parametric design → aero eval) that require
# 7+ tool calls. 1024 was too low (caused MaxTokensReachedException), 4096 was too high
# (made responses slow). 2048 is the sweet spot.
model = BedrockModel(
    model_id=MODEL_ID,
    max_tokens=2048,
)


agent = Agent(
    name="Orchestrator Agent",
    description="Central coordinator for the Car Design Space Explorer. Receives engineer queries, discovers and routes to specialist agents (Aero, Structural, Cost, Geometry) via A2A protocol, synthesizes results, computes Pareto fronts, and returns unified responses.",
    system_prompt=SYSTEM_PROMPT,
    model=model,
    tools=all_tools,
    conversation_manager=conversation_manager,
    # session_manager removed — orchestrator is stateless per request.
    # _clear_messages hook wipes messages before each invocation.
    concurrent_invocation_mode=ConcurrentInvocationMode.UNSAFE_REENTRANT,
)

# Stateless per request: clear all messages and request context before each invocation.
def _clear_messages(event: BeforeInvocationEvent):
    event.agent.messages.clear()
    _request_context.original_query = None  # reset thread-local for this new request

agent.add_hook(_clear_messages)

# Memory compression hook removed — orchestrator is stateless per request.
# _clear_messages wipes messages before each invocation, so there's nothing to compress.

# ---------------------------------------------------------------------------
# Fix 2: OTel guard — patch from_converse to handle missing 'output' key
# ---------------------------------------------------------------------------
try:
    from opentelemetry.instrumentation.botocore.extensions import bedrock_utils
    _original_from_converse = bedrock_utils._Choice.from_converse

    @classmethod
    def _safe_from_converse(cls, response, capture_content=False):
        try:
            return _original_from_converse.__func__(cls, response, capture_content)
        except (KeyError, TypeError, Exception) as e:
            # MaxTokensReachedException or throttling — response has no 'output'
            # Newer OTel versions require 'index' arg in _Choice.__init__
            logger.warning(f"[otel_guard] ConverseStream response issue: {e} — returning empty choice")
            try:
                return cls(finish_reason="error", message=None, index=0)
            except TypeError:
                return cls(finish_reason="error", message=None)

    bedrock_utils._Choice.from_converse = _safe_from_converse
    logger.info("✅ OTel from_converse patched for KeyError guard")
except Exception as e:
    logger.warning(f"OTel patch skipped: {e}")

# ---------------------------------------------------------------------------
# Monkey-patch the boto3 converse_stream call to repair the FORMATTED request
# right before it hits the Bedrock API. Previous approach (patching _stream)
# failed because Strands' _format_request converts messages AFTER our repair,
# and _convert_prompt_to_messages can inject synthetic toolResult blocks that
# create mismatches. By patching converse_stream, we intercept the final
# wire-format request dict and repair it at the absolute last moment.
# ---------------------------------------------------------------------------

def _repair_formatted_messages(messages: list) -> bool:
    """Repair toolUse/toolResult mismatches in Bedrock wire-format messages.

    Wire format uses 'toolUse' and 'toolResult' keys inside content blocks,
    same as Strands internal format. Returns True if any repairs were made.
    """
    repaired = False

    # --- FIX 0: Merge consecutive same-role messages ---
    # Bedrock requires strict user→assistant→user alternation.
    # Strands' _convert_prompt_to_messages can inject synthetic user messages
    # (with toolResult) adjacent to existing user messages, creating
    # user→user sequences that Bedrock rejects.
    i = 0
    while i < len(messages) - 1:
        if (messages[i].get("role") == messages[i + 1].get("role")
                and messages[i].get("role") in ("user", "assistant")):
            # Merge content of messages[i+1] into messages[i]
            content_a = messages[i].get("content", [])
            content_b = messages[i + 1].get("content", [])
            if isinstance(content_a, list) and isinstance(content_b, list):
                messages[i]["content"] = content_a + content_b
            elif isinstance(content_b, list):
                messages[i]["content"] = content_b
            messages.pop(i + 1)
            repaired = True
            logger.warning(
                f"[converse_patch] Merged consecutive {messages[i]['role']} messages at index {i}/{i+1}"
            )
            # Don't increment — check if the next message also needs merging
        else:
            i += 1

    # --- FIX 0b: Deduplicate toolResult IDs within each user message ---
    # After merging consecutive user messages (FIX 0), the same toolUseId can
    # appear multiple times in a single user turn. Bedrock rejects duplicates.
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        seen_tr_ids = set()
        deduped = []
        for block in content:
            if isinstance(block, dict) and "toolResult" in block:
                tid = block["toolResult"].get("toolUseId", "")
                if tid in seen_tr_ids:
                    logger.warning(f"[converse_patch] Dropping duplicate toolResult ID: {tid}")
                    repaired = True
                    continue
                seen_tr_ids.add(tid)
            deduped.append(block)
        if len(deduped) != len(content):
            msg["content"] = deduped if deduped else [{"text": "(deduplicated)"}]

    for idx in range(len(messages) - 1):
        msg = messages[idx]
        next_msg = messages[idx + 1] if idx + 1 < len(messages) else None
        if not next_msg:
            continue
        if msg.get("role") != "assistant" or next_msg.get("role") != "user":
            continue

        # Collect toolUse IDs from assistant turn
        tu_ids = set()
        for block in msg.get("content", []):
            if isinstance(block, dict) and "toolUse" in block:
                tid = block["toolUse"].get("toolUseId", "")
                if tid:
                    tu_ids.add(tid)

        # Collect toolResult IDs from user turn
        tr_ids = set()
        for block in next_msg.get("content", []):
            if isinstance(block, dict) and "toolResult" in block:
                tid = block["toolResult"].get("toolUseId", "")
                if tid:
                    tr_ids.add(tid)

        # Remove orphaned toolResults (no matching toolUse in prev assistant)
        if tr_ids - tu_ids:
            orphaned = tr_ids - tu_ids
            cleaned = [b for b in next_msg["content"]
                       if not (isinstance(b, dict) and "toolResult" in b
                               and b["toolResult"].get("toolUseId", "") in orphaned)]
            if not cleaned:
                cleaned = [{"text": "(tool results removed)"}]
            next_msg["content"] = cleaned
            repaired = True
            logger.warning(f"[converse_patch] Removed {len(orphaned)} orphaned toolResult(s) at msg[{idx+1}]")

        # Add missing toolResults for orphaned toolUse blocks
        missing = tu_ids - tr_ids
        if missing:
            for mid in missing:
                next_msg["content"].append({
                    "toolResult": {
                        "toolUseId": mid,
                        "status": "error",
                        "content": [{"text": "(result unavailable — trimmed)"}],
                    }
                })
            repaired = True
            logger.warning(f"[converse_patch] Added {len(missing)} dummy toolResult(s) at msg[{idx+1}]")

    # Handle trailing assistant with toolUse but no following user message
    if messages and messages[-1].get("role") == "assistant":
        trailing_ids = []
        for block in messages[-1].get("content", []):
            if isinstance(block, dict) and "toolUse" in block:
                tid = block["toolUse"].get("toolUseId", "")
                if tid:
                    trailing_ids.append(tid)
        if trailing_ids:
            dummy = [{"toolResult": {"toolUseId": t, "status": "error",
                      "content": [{"text": "(result unavailable — trimmed)"}]}}
                     for t in trailing_ids]
            messages.append({"role": "user", "content": dummy})
            repaired = True
            logger.warning(f"[converse_patch] Added trailing user msg with {len(trailing_ids)} dummy toolResult(s)")

    return repaired


# Patch the boto3 client's converse_stream method
_bedrock_client = model.client
_original_converse_stream = _bedrock_client.converse_stream

def _patched_converse_stream(**kwargs):
    """Lightweight wrapper: log message structure and retry with nuked history on ValidationException.

    Previous versions had elaborate multi-pass repair logic that conflicted
    with the Strands SDK's own message management (especially after SDK updates).
    Now we just: try → if ValidationException about tool pairing → nuke to
    single user message → retry once.
    """
    messages = kwargs.get("messages", [])

    # Capture the original user query before any issues
    _original_user_query = None
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        for block in msg.get("content", []):
            if (isinstance(block, dict) and "text" in block
                    and "toolResult" not in block
                    and block["text"] not in ("(tool results removed)", "(deduplicated)",
                                               "(result unavailable — conversation was trimmed)",
                                               "(result unavailable — repaired)")):
                _original_user_query = block["text"]
                break
        if _original_user_query:
            break

    # Log message structure for debugging
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        content = msg.get("content", [])
        tu = sum(1 for b in content if isinstance(b, dict) and "toolUse" in b)
        tr = sum(1 for b in content if isinstance(b, dict) and "toolResult" in b)
        if tu or tr:
            logger.info(f"[converse_patch] msg[{i}] role={role} toolUse={tu} toolResult={tr}")

    try:
        return _original_converse_stream(**kwargs)
    except Exception as e:
        err_msg = str(e)
        if "ValidationException" not in err_msg:
            raise
        if not ("toolResult" in err_msg or "tool_result" in err_msg or "tool_use" in err_msg):
            raise

        # Single recovery: nuke all history, keep only the user query
        logger.warning(f"[converse_patch] ValidationException on tool pairing — nuking to single user message: {err_msg[:200]}")

        last_user_text = (
            getattr(_request_context, "original_query", None)
            or _original_user_query
            or "(please repeat your request)"
        )

        messages.clear()
        messages.append({"role": "user", "content": [{"text": last_user_text}]})
        logger.info(f"[converse_patch] Nuked to: {last_user_text[:100]}")
        return _original_converse_stream(**kwargs)

_bedrock_client.converse_stream = _patched_converse_stream
logger.info("✅ Monkey-patched boto3 converse_stream with message repair")
logger.info("✅ Orchestrator Agent created")

runtime_url = os.environ.get("AGENTCORE_RUNTIME_URL", f"http://127.0.0.1:{AGENT_PORT}/")
logger.info(f"Runtime URL: {runtime_url}")


# ---------------------------------------------------------------------------
# HistoryStrippingTaskStore — prevents A2A response payload bloat
# ---------------------------------------------------------------------------
class HistoryStrippingTaskStore(TaskStore):
    """TaskStore that strips conversation history from tasks on retrieval.

    The A2A protocol's Task object accumulates all messages (tool calls,
    streaming status updates, intermediate results) in task.history.
    When serialized into the JSON-RPC response, this can bloat payloads
    beyond AgentCore's size limit, causing -32603 Internal errors.

    This store delegates to InMemoryTaskStore but nulls out history on get(),
    so the response only contains artifacts and status — not the full
    conversation replay.
    """

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
    enable_a2a_compliant_streaming=False,  # Synchronous — one complete payload, no streaming events
)

# FastAPI app
app = FastAPI()


@app.get("/ping")
def ping():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agent": "orchestrator_agent",
        "version": "1.1.0",
        "features": [
            "a2a_protocol",
            "dynamic_agent_discovery",
            "multi_agent_routing",
            "geometry_modification_routing",
            "stable_diffusion_design_preview",
            "pareto_front_computation",
            "variant_ranking",
            "parameter_sweep",
        ],
    }


_a2a_app = a2a_server.to_fastapi_app()


# ---------------------------------------------------------------------------
# ASGI middleware: wrap non-JSON-RPC payloads arriving at POST /
# ---------------------------------------------------------------------------
import uuid as _uuid
from starlette.types import ASGIApp, Receive, Scope, Send


MAX_RESPONSE_BYTES = 60_000  # Synchronous A2A — no streaming history bloat, safe to allow larger payloads


def _truncate_a2a_response(body: bytes) -> bytes:
    """Reduce an A2A JSON-RPC response while always preserving valid JSON."""
    if len(body) <= MAX_RESPONSE_BYTES:
        return body

    def _oversize_error(response_id=None) -> bytes:
        return json.dumps({
            "jsonrpc": "2.0",
            "id": response_id,
            "error": {
                "code": -32603,
                "message": "Agent response exceeded the supported size limit",
            },
        }, separators=(",", ":")).encode()

    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return _oversize_error()

    import re
    TAG_PATTERN = re.compile(r'\[(STL|IMAGE)\][^\[]*\[/\1\]')

    def _extract_tags_and_trim(text: str, budget: int) -> str:
        """Extract [STL]/[IMAGE] tags, keep minimal prose, fit in budget."""
        tags = TAG_PATTERN.findall(text)
        tag_matches = list(TAG_PATTERN.finditer(text))
        if tag_matches:
            preserved = " ".join(m.group(0) for m in tag_matches)
            prose_budget = max(0, budget - len(preserved) - 20)
            # Take prose from before the first tag
            prose = text[:min(prose_budget, tag_matches[0].start())].strip()
            return (prose + "\n" + preserved).strip()[:budget]
        return text[:budget]

    # --- Dig into result (may be nested under result.task or just result) ---
    result = data.get("result", {})
    if not isinstance(result, dict):
        return _oversize_error(data.get("id"))

    # Handle both flat (result.artifacts) and nested (result.task.artifacts) shapes
    task_obj = result.get("task", result) if "task" in result else result

    # 1. ALWAYS strip history
    task_obj.pop("history", None)
    result.pop("history", None)

    # Re-check after history strip — often this alone brings it under the limit
    after_history = json.dumps(data, separators=(",", ":")).encode()
    if len(after_history) <= MAX_RESPONSE_BYTES:
        logger.info(f"Truncated A2A response from {len(body)} to {len(after_history)} bytes (history strip sufficient)")
        return after_history

    # 2. If artifacts exist, nuke status.message — it's redundant
    artifacts = task_obj.get("artifacts", [])
    if artifacts:
        status = task_obj.get("status", {})
        if isinstance(status, dict) and "message" in status:
            status.pop("message", None)

    # 3. Merge text from ALL artifacts into one, then collapse to single artifact.
    # Use a generous text budget: leave ~1 KB for JSON envelope overhead.
    TEXT_BUDGET = MAX_RESPONSE_BYTES - 1024
    if artifacts:
        all_text = ""
        non_text_parts = []
        for artifact in artifacts:
            for part in artifact.get("parts", []):
                if isinstance(part, dict):
                    text = part.get("text", "")
                    if text:
                        all_text += text
                    elif part.get("kind") == "text" and "text" in part:
                        all_text += part["text"]
                    else:
                        non_text_parts.append(part)

        # Trim the merged text — only cut if it actually exceeds the budget
        merged = all_text.strip()
        trimmed = _extract_tags_and_trim(merged, TEXT_BUDGET) if len(merged) > TEXT_BUDGET else merged
        # Collapse to single artifact
        task_obj["artifacts"] = [{
            "artifactId": artifacts[0].get("artifactId", ""),
            "name": artifacts[0].get("name", "agent_response"),
            "parts": [{"kind": "text", "text": trimmed}] + non_text_parts[:1],
        }]
        artifacts = task_obj["artifacts"]

    # Also trim status.message if it survived (no artifacts case)
    if not artifacts:
        status = task_obj.get("status", {})
        if isinstance(status, dict):
            msg = status.get("message", {})
            if isinstance(msg, dict) and "parts" in msg:
                all_text = ""
                for part in msg["parts"]:
                    if isinstance(part, dict) and "text" in part:
                        all_text += part["text"] + "\n"
                merged = all_text.strip()
                trimmed = _extract_tags_and_trim(merged, TEXT_BUDGET) if len(merged) > TEXT_BUDGET else merged
                msg["parts"] = [{"kind": "text", "text": trimmed}]

    truncated = json.dumps(data, separators=(",", ":")).encode()  # compact JSON

    # 4. If still too large, trim artifact text harder
    if len(truncated) > MAX_RESPONSE_BYTES and artifacts:
        first_parts = artifacts[0].get("parts", [])
        for part in first_parts:
            if isinstance(part, dict) and "text" in part:
                part["text"] = _extract_tags_and_trim(part["text"], MAX_RESPONSE_BYTES // 3)
        truncated = json.dumps(data, separators=(",", ":")).encode()

    # 5. Last resort: return a small, valid JSON-RPC error envelope.
    if len(truncated) > MAX_RESPONSE_BYTES:
        truncated = _oversize_error(data.get("id"))

    logger.info(f"Truncated A2A response from {len(body)} to {len(truncated)} bytes")
    return truncated


class A2APayloadNormalizer:
    """ASGI middleware that normalizes A2A payloads in both directions.

    Inbound: wraps raw payloads into A2A JSON-RPC envelopes if needed.
    Outbound: truncates oversized response payloads to prevent -32603 errors
    from AgentCore's payload size limit.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

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

        # Try to parse and check if it's already valid JSON-RPC
        needs_wrap = False
        try:
            parsed = json.loads(raw_body)
            if (isinstance(parsed, dict)
                    and parsed.get("jsonrpc") == "2.0"
                    and "method" in parsed):
                # Already valid JSON-RPC — ensure messageId exists
                msg = parsed.get("params", {}).get("message", {})
                if msg and "messageId" not in msg:
                    msg["messageId"] = str(_uuid.uuid4())
                    raw_body = json.dumps(parsed).encode()
            else:
                needs_wrap = True
        except (json.JSONDecodeError, Exception):
            needs_wrap = True

        if needs_wrap:
            # Extract prompt text from various formats
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

        # Feed the (possibly rewritten) body back to the app
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
        original_status = [200]

        async def capturing_send(message):
            nonlocal response_headers_sent
            if message["type"] == "http.response.start":
                # Capture but don't send yet — we need to check body size
                original_status[0] = message.get("status", 200)
                response_headers_sent = message
            elif message["type"] == "http.response.body":
                response_body_parts.append(message.get("body", b""))
                if not message.get("more_body", False):
                    # All body collected — truncate if needed
                    full_body = b"".join(response_body_parts)
                    truncated_body = _truncate_a2a_response(full_body)

                    # Update content-length in headers
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


# Wrap the A2A app with the normalizer middleware
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

    # If the payload is already a valid JSON-RPC message/send, pass it through
    import uuid as _uuid
    if (isinstance(body, dict)
            and body.get("jsonrpc") == "2.0"
            and body.get("method") in ("message/send", "message/stream")):
        a2a_payload = body
        # Ensure messageId is present
        msg = a2a_payload.get("params", {}).get("message", {})
        if "messageId" not in msg:
            msg["messageId"] = str(_uuid.uuid4())
    else:
        # Wrap simple payload into A2A JSON-RPC envelope
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

    # Forward to the A2A app via internal ASGI transport
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
    print("Orchestrator Agent — Car Design Space Explorer")
    print(f"  Agent Card: http://{host}:{port}/.well-known/agent-card.json")
    print("=" * 60)
    uvicorn.run(app, host=host, port=port)
