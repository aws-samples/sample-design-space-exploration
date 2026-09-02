#!/usr/bin/env python3
"""
Deploy Car Design Explorer agents to Bedrock AgentCore Runtime.

Uses the AgentCore Starter Toolkit (Runtime.configure + launch) for
direct code deployment — NO Docker required. Same pattern as SPA.

All 5 agents use A2A protocol (FastAPI + A2AServer on port 9000).

Usage (deploy all agents):
    python deploy_agents.py --deploy-all \
      --model-id us.anthropic.claude-haiku-4-5-20251001-v1:0 \
      --cognito-user-pool-id us-east-1_BnDLur150 \
      --cognito-client-id cf3qjt93tqocdhtvv5539t8mt \
      --mcp-user-pool-id us-east-1_3mhz73JyW \
      --region us-east-1

Usage (deploy single agent):
    python deploy_agents.py --agent orchestrator \
      --model-id us.anthropic.claude-haiku-4-5-20251001-v1:0 \
      --cognito-user-pool-id us-east-1_BnDLur150 \
      --cognito-client-id cf3qjt93tqocdhtvv5539t8mt \
      --region us-east-1

Usage (create MCP Gateway only):
    python deploy_agents.py --create-gateway \
      --mcp-user-pool-id us-east-1_3mhz73JyW \
      --region us-east-1

Usage (list deployed agents):
    python deploy_agents.py --list --region us-east-1

Usage (update Lambda with orchestrator ARN):
    python deploy_agents.py --wire-lambda \
      --orchestrator-arn arn:aws:bedrock-agentcore:... \
      --region us-east-1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from urllib.parse import quote

import boto3

# Import the starter toolkit at module level so it registers its bundled
# botocore service model loaders (bedrock-agentcore-control) before any
# boto3.client() calls are made further down (e.g. create_mcp_gateway).
from bedrock_agentcore_starter_toolkit import Runtime as _AgentCoreRuntime  # noqa: F401

# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------
AGENTS = {
    "aero": {
        "name": "car_design_aero",
        "display": "Aero Agent",
        "entrypoint": "agents/aero_agent.py",
        "description": "Aerodynamic KPI and surface prediction with MLSimKit",
        "needs_memory": False,
    },
    "structural": {
        "name": "car_design_structural",
        "display": "Structural Agent",
        "entrypoint": "agents/structural_agent.py",
        "description": "Structural feasibility evaluation",
        "needs_memory": False,
    },
    "cost": {
        "name": "car_design_cost",
        "display": "Cost Agent",
        "entrypoint": "agents/cost_agent.py",
        "description": "Manufacturing cost estimation with MCP servers",
        "needs_memory": False,
    },
    "geometry": {
        "name": "car_design_geometry",
        "display": "Geometry Agent",
        "entrypoint": "agents/geometry_agent.py",
        "description": "Geometry modification with Stable Diffusion 3.5 Large and trimesh",
        "needs_memory": False,
    },
    "orchestrator": {
        "name": "car_design_orchestrator",
        "display": "Orchestrator Agent",
        "entrypoint": "agents/orchestrator_agent.py",
        "description": "Central coordinator for multi-agent design exploration",
        "needs_memory": True,
    },
}

# Deploy order: specialists first, orchestrator last (discovers others)
DEPLOY_ORDER = ["aero", "structural", "cost", "geometry", "orchestrator"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(SCRIPT_DIR, "backend")
DEFAULT_IMAGE_MODEL_ID = "stability.sd3-5-large-v1:0"
DEFAULT_IMAGE_MODEL_REGION = "us-west-2"


# ---------------------------------------------------------------------------
# Per-agent requirements (what each agent needs beyond the base)
# ---------------------------------------------------------------------------
BASE_REQUIREMENTS = """\
strands-agents[a2a]
strands-agents-tools
bedrock-agentcore
uv
boto3>=1.35.0
pydantic>=2.0.0
fastapi>=0.115.0
uvicorn>=0.30.0
httpx>=0.27.0
mcp
requests>=2.31.0
"""

AERO_EXTRA = """\
torch>=2.5.0,<2.6.0
torchvision>=0.20.0,<0.21.0
torch-geometric>=2.7.0,<2.8.0
torch-summary>=1.4.5
torchmetrics>=0.11.4
numpy==1.26.4
pandas>=2.2.0
scipy>=1.15.0
scikit-learn>=1.5.0
pyvista>=0.43.0
vtk>=9.3.0
trimesh==4.10.1
matplotlib>=3.10.0
pillow>=12.0.0
PyYAML>=6.0.0
tqdm>=4.67.0
click>=8.3.0
click-didyoumean>=0.3.0
click-plugins>=1.1.0
click-repl>=0.3.0
opencv-python-headless>=4.8.0
"""

GEOMETRY_EXTRA = """\
trimesh==4.10.1
numpy==1.26.4
matplotlib==3.10.7
pillow==12.0.0
scipy==1.15.3
networkx>=3.1
manifold3d>=2.4
"""

STRUCTURAL_EXTRA = """\
trimesh==4.10.1
numpy==1.26.4
networkx>=3.1
"""


def get_requirements_for_agent(agent_key: str) -> str:
    """Return the full requirements content for an agent."""
    reqs = BASE_REQUIREMENTS
    if agent_key == "aero":
        reqs += AERO_EXTRA
    elif agent_key == "geometry":
        reqs += GEOMETRY_EXTRA
    elif agent_key == "structural":
        reqs += STRUCTURAL_EXTRA
    return reqs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_account_id() -> str:
    return boto3.client("sts").get_caller_identity()["Account"]


# ---------------------------------------------------------------------------
# CDK-owned runtime role and endpoint resolution
# ---------------------------------------------------------------------------

ROLE_OUTPUT_KEYS = {
    agent_key: f"{agent_key.title()}AgentRoleArn" for agent_key in AGENTS
}
SPECIALIST_AGENT_KEYS = ("aero", "structural", "cost", "geometry")


def resolve_execution_role_arns(
    agent_keys: list[str],
    region: str,
    requested_model_id: str,
) -> dict[str, str]:
    """Resolve per-agent runtime roles and validate their authorized model.

    Runtime roles are created and governed only by CDK. This deployment script
    intentionally has no IAM role creation or policy mutation path.
    """
    cfn = boto3.client("cloudformation", region_name=region)
    try:
        stack = cfn.describe_stacks(StackName="CarDesignAgents")["Stacks"][0]
    except Exception as exc:
        raise RuntimeError(
            "Unable to read CarDesignAgents outputs. Deploy the CDK stack before agents."
        ) from exc

    outputs = {
        item["OutputKey"]: item["OutputValue"]
        for item in stack.get("Outputs", [])
    }
    authorized_model_id = outputs.get("AgentModelId", "")
    if not authorized_model_id:
        raise RuntimeError(
            "CarDesignAgents has no AgentModelId output. Redeploy the CDK stack "
            "before deploying agents."
        )
    if authorized_model_id != requested_model_id:
        raise RuntimeError(
            f"CarDesignAgents authorizes model {authorized_model_id!r}, but "
            f"deployment requested {requested_model_id!r}. Redeploy CDK with "
            "the same AGENT_MODEL_ID first."
        )

    role_arns: dict[str, str] = {}
    missing: list[str] = []
    for agent_key in agent_keys:
        output_key = ROLE_OUTPUT_KEYS[agent_key]
        role_arn = outputs.get(output_key, "")
        if not role_arn.startswith("arn:aws:iam::"):
            missing.append(output_key)
        else:
            role_arns[agent_key] = role_arn

    if missing:
        raise RuntimeError(
            "CarDesignAgents is missing per-agent role outputs: "
            + ", ".join(missing)
            + ". Redeploy the CDK stack before deploying agents."
        )
    return role_arns


def _runtime_invocation_url(runtime_arn: str, region: str) -> str:
    encoded_arn = quote(runtime_arn, safe="")
    return (
        f"https://bedrock-agentcore.{region}.amazonaws.com/"
        f"runtimes/{encoded_arn}/invocations/"
    )


def resolve_specialist_runtime_arns(region: str, account_id: str) -> dict[str, str]:
    """Resolve READY specialist runtimes from the active AWS account.

    Local deployment JSON is intentionally not trusted because this repository
    is used across accounts and Regions.
    """
    control = boto3.client("bedrock-agentcore-control", region_name=region)
    expected_names = {AGENTS[key]["name"]: key for key in SPECIALIST_AGENT_KEYS}
    resolved: dict[str, str] = {}
    next_token = ""  # nosec B105 -- pagination sentinel, not a credential

    while True:
        request = {"maxResults": 100}
        if next_token:
            request["nextToken"] = next_token
        response = control.list_agent_runtimes(**request)
        for runtime in response.get("agentRuntimes", []):
            runtime_name = runtime.get("agentRuntimeName", "")
            runtime_arn = runtime.get("agentRuntimeArn", "")
            agent_key = expected_names.get(runtime_name)
            expected_prefix = f"arn:aws:bedrock-agentcore:{region}:{account_id}:"
            if (
                agent_key
                and runtime.get("status") == "READY"
                and runtime_arn.startswith(expected_prefix)
            ):
                resolved[agent_key] = runtime_arn
        next_token = response.get("nextToken", "")
        if not next_token:
            break

    missing = [key for key in SPECIALIST_AGENT_KEYS if key not in resolved]
    if missing:
        raise RuntimeError(
            "Cannot configure orchestrator; READY specialist runtimes are missing "
            f"from account {account_id} in {region}: " + ", ".join(missing)
        )
    return resolved


# ---------------------------------------------------------------------------
# AgentCore Memory (for orchestrator)
# ---------------------------------------------------------------------------

def create_agentcore_memory(region: str) -> str:
    """Create a new AgentCore Memory for the orchestrator agent."""
    try:
        from bedrock_agentcore.memory import MemoryClient

        print("  Creating AgentCore Memory...")
        memory_client = MemoryClient(region_name=region)
        timestamp = int(time.time())
        memory_name = f"CarDesignExplorer_{timestamp}"

        result = memory_client.create_memory_and_wait(
            name=memory_name,
            description="Car Design Space Explorer - Orchestrator conversation memory",
            strategies=[],
            event_expiry_days=30,
            max_wait=300,
            poll_interval=10,
        )
        memory_id = result["id"]
        print(f"  [OK] Memory created: {memory_id} ({memory_name})")
        return memory_id
    except Exception as e:
        print(f"  [WARNING] Memory creation failed: {e}")
        print("  Agent will create memory at startup as fallback")
        return ""


# ---------------------------------------------------------------------------
# MCP Gateway for Cost Agent
# ---------------------------------------------------------------------------

def ensure_gateway_role(account_id: str, region: str) -> str:
    """Create or reuse the MCP Gateway IAM role."""
    iam = boto3.client("iam")
    role_name = "CarDesignMcpGatewayRole"

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
            "Action": "sts:AssumeRole",
            "Condition": {
                "StringEquals": {"aws:SourceAccount": account_id},
                "ArnLike": {
                    "aws:SourceArn": f"arn:aws:bedrock-agentcore:{region}:{account_id}:*"
                },
            },
        }],
    }

    try:
        resp = iam.get_role(RoleName=role_name)
        role_arn = resp["Role"]["Arn"]
        iam.update_assume_role_policy(
            RoleName=role_name,
            PolicyDocument=json.dumps(trust_policy),
        )
        print(f"  [OK] Using existing gateway role: {role_arn}")
    except iam.exceptions.NoSuchEntityException:
        resp = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="IAM role for Car Design MCP Gateway",
        )
        role_arn = resp["Role"]["Arn"]
        print(f"  [OK] Created gateway role: {role_arn}")

    gateway_log_group = (
        f"arn:aws:logs:{region}:{account_id}:"
        "log-group:/aws/bedrock-agentcore/gateways/*"
    )
    permission_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "CreateGatewayLogs",
                "Effect": "Allow",
                "Action": ["logs:CreateLogGroup"],
                "Resource": [gateway_log_group],
            },
            {
                "Sid": "WriteGatewayLogs",
                "Effect": "Allow",
                "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
                "Resource": [f"{gateway_log_group}:log-stream:*"],
            },
        ],
    }
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName="CarDesignMcpGatewayPermissions",
        PolicyDocument=json.dumps(permission_policy),
    )
    print("  [WAIT] Waiting 10s for IAM propagation...")
    time.sleep(10)  # nosemgrep
    return role_arn


def create_mcp_gateway(
    mcp_user_pool_id: str,
    region: str,
    account_id: str,
) -> tuple[str, str]:
    """Create AgentCore MCP Gateway for Cost Agent's MCP servers.

    Returns (gateway_id, gateway_url).
    """
    client = boto3.client("bedrock-agentcore-control", region_name=region)

    gateway_name = f"CarDesignMcpGateway-{account_id[:8]}"

    # Check for existing gateway
    try:
        resp = client.list_gateways(maxResults=50)
        for gw in resp.get("items", resp.get("gatewaySummaries", [])):
            name = gw.get("name", gw.get("gatewayName", ""))
            if "CarDesign" in name:
                gw_id = gw.get("gatewayId", "")
                gw_url = f"https://{gw_id}.gateway.bedrock-agentcore.{region}.amazonaws.com"
                print(f"  [OK] Found existing MCP Gateway: {gw_id}")
                return gw_id, gw_url
    except Exception as e:
        print(f"  [INFO] list_gateways: {e}")

    discovery_url = (
        f"https://cognito-idp.{region}.amazonaws.com/"
        f"{mcp_user_pool_id}/.well-known/openid-configuration"
    )

    print(f"  Creating MCP Gateway: {gateway_name}")
    print(f"  Cognito Discovery URL: {discovery_url}")

    gateway_role_arn = ensure_gateway_role(account_id, region)

    # Get MCP Cognito client ID from Secrets Manager
    mcp_client_id = ""
    try:
        sm = boto3.client("secretsmanager", region_name=region)
        # SecretId is a Secrets Manager resource locator, not credential data.
        secret = sm.get_secret_value(  # nosec B106
            SecretId="car-design/mcp-gateway-credentials"
        )
        creds = json.loads(secret["SecretString"])
        mcp_client_id = creds.get("client_id", "")
        print(f"  [OK] MCP Client ID from Secrets Manager: {mcp_client_id[:8]}...")
    except Exception as e:
        print(f"  [WARNING] Could not read MCP credentials: {e}")

    try:
        # allowedClients is intentionally NOT used: AgentCore validates it against
        # the JWT 'aud' claim, but Cognito M2M (client_credentials) tokens do NOT
        # include 'aud' — causing 424 on every Cost Agent MCP call.
        # allowedScopes is safe: Cognito M2M tokens DO carry a 'scope' claim
        # (e.g. "mcp-api/read mcp-api/write"), so this satisfies the API's
        # requirement that at least one constraint field is present.
        create_params = dict(
            name=gateway_name,
            description="MCP Gateway for Car Design Cost Agent",
            protocolType="MCP",
            roleArn=gateway_role_arn,
            authorizerType="CUSTOM_JWT",
            authorizerConfiguration={
                "customJWTAuthorizer": {
                    "discoveryUrl": discovery_url,
                    "allowedScopes": ["mcp-api/read", "mcp-api/write"],
                }
            },
        )

        resp = client.create_gateway(**create_params)
        gw_id = resp["gatewayId"]
        gw_url = f"https://{gw_id}.gateway.bedrock-agentcore.{region}.amazonaws.com"
        print(f"  [OK] MCP Gateway created: {gw_id}")
        print(f"  Gateway URL: {gw_url}/mcp")

        print("  Waiting for gateway to become READY...")
        for i in range(30):
            try:
                status_resp = client.get_gateway(gatewayId=gw_id)
                status = status_resp.get("status", "UNKNOWN")
                print(f"    Attempt {i+1}/30 — Status: {status}")
                if status == "READY":
                    break
                if status in ("FAILED", "CREATE_FAILED"):
                    print(f"  [ERROR] Gateway creation failed: {status}")
                    return gw_id, gw_url
            except Exception:
                pass
            time.sleep(10)  # nosemgrep

        return gw_id, gw_url

    except Exception as e:
        print(f"  [ERROR] create_gateway failed: {e}")
        raise


# ---------------------------------------------------------------------------
# Wire orchestrator ARN to Lambda
# ---------------------------------------------------------------------------

def wire_lambda_orchestrator(orchestrator_arn: str, region: str) -> None:
    """Update the WebSocket Lambda function with the orchestrator ARN."""
    lambda_client = boto3.client("lambda", region_name=region)
    function_name = "CarDesignWSHandler"

    try:
        resp = lambda_client.get_function_configuration(FunctionName=function_name)
        current_env = resp.get("Environment", {}).get("Variables", {})
        current_env["ORCHESTRATOR_RUNTIME_ARN"] = orchestrator_arn
        lambda_client.update_function_configuration(
            FunctionName=function_name,
            Environment={"Variables": current_env},
        )
        print(f"  [OK] Lambda {function_name} updated with ORCHESTRATOR_RUNTIME_ARN")
    except Exception as e:
        print(f"  [WARNING] Could not update Lambda: {e}")
        print(f"  Manually set ORCHESTRATOR_RUNTIME_ARN={orchestrator_arn}")


# ---------------------------------------------------------------------------
# Prepare AgentCore runtime environment variables
# ---------------------------------------------------------------------------

def prepare_agent_config(
    agent_key: str,
    account_id: str,
    region: str,
    model_id: str,
    memory_id: str = "",
    gateway_url: str = "",
    specialist_runtime_arns: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build environment variables passed to the AgentCore runtime.

    Specialist endpoints remain explicit deployment configuration, avoiding
    runtime control-plane discovery permissions without modifying source code.
    """
    env_vars = {
        "MODEL_ID": model_id,
        "AWS_REGION": region,
        "S3_MODEL_BUCKET": f"car-design-explorer-models-{account_id}",
    }

    if agent_key == "orchestrator":
        # Secrets Manager resource name, not credential material.
        env_vars["OAUTH_SECRET_NAME"] = "car-design/agent-oauth-credentials"  # nosec B105
        specialist_runtime_arns = specialist_runtime_arns or {}
        missing = [
            key for key in SPECIALIST_AGENT_KEYS
            if not specialist_runtime_arns.get(key)
        ]
        if missing:
            raise ValueError(
                "Orchestrator requires specialist runtime ARNs: "
                + ", ".join(missing)
            )
        for key in SPECIALIST_AGENT_KEYS:
            env_vars[f"{key.upper()}_AGENT_URL"] = _runtime_invocation_url(
                specialist_runtime_arns[key], region
            )
        if memory_id:
            env_vars["AGENTCORE_MEMORY_ID"] = memory_id

    if agent_key == "cost":
        if not gateway_url:
            raise ValueError("Cost agent requires a deployed MCP Gateway URL")
        env_vars["GATEWAY_URL"] = f"{gateway_url}/mcp"
        # Secrets Manager resource name, not credential material.
        env_vars["MCP_GATEWAY_SECRET_NAME"] = "car-design/mcp-gateway-credentials"  # nosec B105

    if agent_key == "aero":
        env_vars["VARIANT_CACHE_TABLE"] = "CarDesignVariantCache"
        env_vars["KPI_MODEL_S3_URI"] = "kpi/best_model.pt"
        env_vars["SURFACE_MODEL_S3_URI"] = "surface/best_model.pt"
        env_vars["SLICES_AE_MODEL_S3_URI"] = "slices/ae_best_model.pt"
        env_vars["SLICES_MGN_MODEL_S3_URI"] = "slices/mgn_last_model.pt"
        env_vars["GEOMETRY_S3_BUCKET"] = f"car-design-explorer-models-{account_id}"

    if agent_key == "structural":
        env_vars["GEOMETRY_S3_BUCKET"] = f"car-design-explorer-models-{account_id}"

    if agent_key == "geometry":
        env_vars["GEOMETRY_S3_BUCKET"] = f"car-design-explorer-models-{account_id}"
        env_vars["IMAGE_MODEL_ID"] = os.environ.get(
            "IMAGE_MODEL_ID", DEFAULT_IMAGE_MODEL_ID
        )
        env_vars["IMAGE_MODEL_REGION"] = os.environ.get(
            "IMAGE_MODEL_REGION", DEFAULT_IMAGE_MODEL_REGION
        )
        env_vars["BYPASS_TOOL_CONSENT"] = "true"

    return env_vars


def copy_agent_entrypoint(
    agent_key: str,
    staging_dir: str,
) -> str:
    """Copy the agent entrypoint without replacing environment lookups."""
    agent_def = AGENTS[agent_key]
    src_file = os.path.join(BACKEND_DIR, agent_def["entrypoint"])
    dest_file = os.path.join(staging_dir, os.path.basename(agent_def["entrypoint"]))
    shutil.copy2(src_file, dest_file)
    print(f"  [OK] Agent entrypoint copied: {dest_file}")
    return dest_file


# ---------------------------------------------------------------------------
# Stage deployment package for an agent
# ---------------------------------------------------------------------------

def stage_agent_package(
    agent_key: str,
) -> tuple[str, str, str]:
    """Create a staging directory with the agent code and requirements.

    Returns (staging_dir, entrypoint_file, requirements_file).
    """
    staging_dir = os.path.join(SCRIPT_DIR, f".deploy_staging_{agent_key}")
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)

    # Copy all backend Python modules needed by the agent
    # 1. Copy the agents/ directory
    agents_src = os.path.join(BACKEND_DIR, "agents")
    agents_dst = os.path.join(staging_dir, "agents")
    shutil.copytree(agents_src, agents_dst, ignore=shutil.ignore_patterns("__pycache__"))

    # 2. Copy models/ directory (data_models, a2a protocol)
    models_src = os.path.join(BACKEND_DIR, "models")
    models_dst = os.path.join(staging_dir, "models")
    if agent_key == "aero":
        # Aero agent needs model weights for inference — include weights/*.pt
        shutil.copytree(
            models_src, models_dst,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        print("  [OK] Model weights included for aero agent")
    else:
        # Other agents don't need model weights
        shutil.copytree(
            models_src, models_dst,
            ignore=shutil.ignore_patterns("__pycache__", "weights", "*.pt"),
        )

    # 3. Copy geometry/ module
    geometry_src = os.path.join(BACKEND_DIR, "geometry")
    if os.path.exists(geometry_src):
        shutil.copytree(
            geometry_src, os.path.join(staging_dir, "geometry"),
            ignore=shutil.ignore_patterns("__pycache__"),
        )

    # 4. Copy only the inference module required by the Aero agent.
    if agent_key == "aero":
        training_src = os.path.join(BACKEND_DIR, "training")
        training_dst = os.path.join(staging_dir, "training")
        os.makedirs(training_dst, exist_ok=True)
        for filename in ("__init__.py", "inference.py"):
            source = os.path.join(training_src, filename)
            if os.path.exists(source):
                shutil.copy2(source, os.path.join(training_dst, filename))

    # 5. Copy mcp_servers/ module (for cost agent)
    # IMPORTANT: this directory was renamed from mcp/ to mcp_servers/ to avoid
    # shadowing the pip-installed "mcp" package (which provides mcp.types, etc.)
    mcp_src = os.path.join(BACKEND_DIR, "mcp_servers")
    if os.path.exists(mcp_src):
        shutil.copytree(
            mcp_src, os.path.join(staging_dir, "mcp_servers"),
            ignore=shutil.ignore_patterns("__pycache__"),
        )

    # 6. Copy lambda_handler/ module
    handler_src = os.path.join(BACKEND_DIR, "lambda_handler")
    if os.path.exists(handler_src):
        shutil.copytree(
            handler_src, os.path.join(staging_dir, "lambda_handler"),
            ignore=shutil.ignore_patterns("__pycache__"),
        )

    # 7. Copy __init__.py
    init_src = os.path.join(BACKEND_DIR, "__init__.py")
    if os.path.exists(init_src):
        shutil.copy2(init_src, os.path.join(staging_dir, "__init__.py"))

    # 8. Copy pyproject.toml
    pyproject_src = os.path.join(BACKEND_DIR, "pyproject.toml")
    if os.path.exists(pyproject_src):
        shutil.copy2(pyproject_src, os.path.join(staging_dir, "pyproject.toml"))

    # 9. For aero agent: copy mlsimkit_dist
    if agent_key == "aero":
        mlsimkit_src = os.path.join(BACKEND_DIR, "mlsimkit_dist")
        if os.path.exists(mlsimkit_src):
            shutil.copytree(
                mlsimkit_src, os.path.join(staging_dir, "mlsimkit_dist"),
            )
            print("  [OK] MLSimKit dist included in staging")

    # Copy the entrypoint unchanged; runtime settings are supplied through
    # AgentCore environmentVariables during launch and explicit updates.
    entrypoint_file = copy_agent_entrypoint(agent_key, staging_dir)
    entrypoint = os.path.basename(entrypoint_file)

    # Write requirements.txt
    reqs_content = get_requirements_for_agent(agent_key)
    reqs_file = os.path.join(staging_dir, "requirements.txt")
    with open(reqs_file, "w") as f:
        f.write(reqs_content)
    print(f"  [OK] Requirements written: {reqs_file}")

    return staging_dir, entrypoint, reqs_file


# ---------------------------------------------------------------------------
# Deploy a single agent using Starter Toolkit (Runtime.configure + launch)
# ---------------------------------------------------------------------------

def deploy_single_agent(
    agent_key: str,
    account_id: str,
    region: str,
    role_arn: str,
    cognito_user_pool_id: str,
    cognito_client_id: str,
    model_id: str,
    agent_m2m_client_id: str = "",
    mcp_user_pool_id: str = "",
    memory_id: str = "",
    gateway_url: str = "",
    specialist_runtime_arns: dict[str, str] | None = None,
    auto_update: bool = True,
) -> str:
    """Deploy a single agent using the AgentCore Starter Toolkit.

    Uses Runtime.configure() + runtime.launch() — same pattern as SPA.
    No Docker required. AgentCore handles container build server-side.

    Returns the agent runtime ARN.
    """
    agent_def = AGENTS[agent_key]
    agent_name = agent_def["name"]

    print(f"\n{'='*60}")
    print(f"Deploying: {agent_def['display']} ({agent_name})")
    print(f"{'='*60}")

    # 1. Prepare environment variables
    env_vars = prepare_agent_config(
        agent_key,
        account_id,
        region,
        model_id,
        memory_id,
        gateway_url,
        specialist_runtime_arns,
    )

    # 2. Stage deployment package
    print("  Staging deployment package...")
    staging_dir, entrypoint, reqs_file = stage_agent_package(agent_key)

    # 3. Build Cognito discovery URL for inbound JWT auth.
    # Must point to the MCP pool (us-east-1_rWtDSUzAQ) because:
    # - Only the MCP pool supports client_credentials (has domain + resource server)
    # - The M2M client that Lambda and orchestrator use to call agents lives on the MCP pool
    # - AgentCore validates the Bearer token's signature against this pool's JWKS
    # - Human users never call AgentCore directly; they go through the Lambda handler
    if not mcp_user_pool_id:
        raise ValueError("--mcp-user-pool-id required: agents use MCP pool for JWT auth")
    discovery_url = (
        f"https://cognito-idp.{region}.amazonaws.com/"
        f"{mcp_user_pool_id}/.well-known/openid-configuration"
    )

    # 4. Configure runtime
    # AgentCore requires at least one authorizer constraint. The deployment
    # command requires the real M2M client ID; placeholders are forbidden.
    print(f"  Configuring AgentCore Runtime: {agent_name}")
    print(f"  Entrypoint: {entrypoint}")
    print(f"  Protocol: A2A")
    print(f"  Auth: Cognito JWT (MCP pool {mcp_user_pool_id})")

    try:
        agentcore_runtime = _AgentCoreRuntime()

        # Change to staging dir so entrypoint is relative
        original_cwd = os.getcwd()
        os.chdir(staging_dir)

        agentcore_runtime.configure(
            entrypoint=entrypoint,
            execution_role=role_arn,
            auto_create_ecr=True,
            requirements_file="requirements.txt",
            region=region,
            agent_name=agent_name,
            protocol="A2A",
            authorizer_configuration={
                "customJWTAuthorizer": {
                    "discoveryUrl": discovery_url,
                    # allowedClients validates the client_id claim in Cognito M2M tokens.
                    # Every client_credentials token carries this client_id.
                    "allowedClients": [agent_m2m_client_id],
                }
            },
        )
        print(f"  [OK] Agent configured")

    except Exception as e:
        os.chdir(original_cwd)
        print(f"  [ERROR] Configuration failed: {e}")
        raise

    # 5. Launch
    print(f"  Launching {agent_name}...")
    try:
        launch_result = agentcore_runtime.launch(
            auto_update_on_conflict=auto_update,
            env_vars=env_vars,
        )
        agent_arn = launch_result.agent_arn
        agent_id = launch_result.agent_id
        print(f"  [OK] Launch initiated")
        print(f"  Agent ARN: {agent_arn}")
        print(f"  Agent ID:  {agent_id}")
    except Exception as e:
        os.chdir(original_cwd)
        print(f"  [ERROR] Launch failed: {e}")
        raise

    # 6. Monitor deployment status
    print(f"  Monitoring deployment...")
    status = "CREATING"
    max_attempts = 60
    attempt = 0

    while attempt < max_attempts:
        try:
            status_response = agentcore_runtime.status()
            status = status_response.endpoint.get("status", "UNKNOWN")
            print(f"    Attempt {attempt + 1}/{max_attempts} — Status: {status}")

            if status == "READY":
                print(f"  [SUCCESS] {agent_def['display']} is READY!")
                break
            elif status in ("CREATE_FAILED", "DELETE_FAILED", "UPDATE_FAILED", "FAILED"):
                print(f"  [ERROR] Deployment failed: {status}")
                break

            attempt += 1
            if attempt < max_attempts:
                time.sleep(30)  # nosemgrep

        except Exception as e:
            print(f"    Status check error: {e}")
            attempt += 1
            if attempt < max_attempts:
                time.sleep(30)  # nosemgrep

    # Restore working directory
    os.chdir(original_cwd)

    if status != "READY":
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise RuntimeError(
            f"{agent_def['display']} did not become READY after launch: {status}"
        )

    # 6b. The Starter Toolkit builds a new ECR image via CodeBuild but does NOT
    # call update-agent-runtime to point the runtime at the new image.
    # We must do this explicitly — otherwise the runtime keeps running the old image.
    print(f"  Updating runtime to latest ECR image...")
    try:
        ecr_client = boto3.client("ecr", region_name=region)
        ecr_repo = f"bedrock-agentcore-{agent_name}"
        images = ecr_client.describe_images(
            repositoryName=ecr_repo,
            filter={"tagStatus": "TAGGED"},
        )
        latest_image = sorted(
            images["imageDetails"],
            key=lambda x: x["imagePushedAt"],
            reverse=True,
        )[0]
        new_tag = latest_image["imageTags"][0]
        new_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{ecr_repo}:{new_tag}"
        print(f"  Latest ECR image tag: {new_tag}")

        agentcore_control = boto3.client("bedrock-agentcore-control", region_name=region)
        agentcore_control.update_agent_runtime(
            agentRuntimeId=agent_id,
            agentRuntimeArtifact={"containerConfiguration": {"containerUri": new_uri}},
            protocolConfiguration={"serverProtocol": "A2A"},
            authorizerConfiguration={
                "customJWTAuthorizer": {
                    "discoveryUrl": discovery_url,
                    "allowedClients": [agent_m2m_client_id],
                }
            },
            roleArn=role_arn,
            networkConfiguration={"networkMode": "PUBLIC"},
            environmentVariables=env_vars,
        )
        print(f"  [OK] Runtime update triggered to {new_tag}")

        # Wait for READY and verify the security-critical runtime settings.
        verified_runtime = None
        for _ in range(30):
            resp = agentcore_control.get_agent_runtime(agentRuntimeId=agent_id)
            rt_status = resp.get("status", "UNKNOWN")
            print(f"    Runtime status: {rt_status}")
            if rt_status in ("CREATE_FAILED", "UPDATE_FAILED", "FAILED"):
                raise RuntimeError(f"Runtime update failed with status {rt_status}")
            if rt_status == "READY":
                verified_runtime = resp
                break
            time.sleep(10)  # nosemgrep

        if verified_runtime is None:
            raise TimeoutError("Runtime did not return to READY after image update")
        if verified_runtime.get("roleArn") != role_arn:
            raise RuntimeError(
                "Runtime role verification failed: expected "
                f"{role_arn}, got {verified_runtime.get('roleArn', '<missing>')}"
            )
        protocol = verified_runtime.get("protocolConfiguration", {}).get("serverProtocol")
        if protocol != "A2A":
            raise RuntimeError(
                f"Runtime protocol verification failed: expected A2A, got {protocol}"
            )
        deployed_env = verified_runtime.get("environmentVariables", {})
        env_mismatches = {
            key: {"expected": value, "actual": deployed_env.get(key)}
            for key, value in env_vars.items()
            if deployed_env.get(key) != value
        }
        if env_mismatches:
            raise RuntimeError(
                f"Runtime environment verification failed: {env_mismatches}"
            )
        print("  [OK] Runtime updated, READY, and using the expected role, A2A protocol, and environment")
    except Exception as e:
        shutil.rmtree(staging_dir, ignore_errors=True)
        print(f"  [ERROR] Runtime image/role update failed: {e}")
        raise

    # 7. Save deployment info
    deployment_info = {
        "agent_name": agent_name,
        "agent_key": agent_key,
        "agent_arn": agent_arn,
        "agent_id": agent_id,
        "execution_role_arn": role_arn,
        "model_id": model_id,
        "display_name": agent_def["display"],
        "protocol": "A2A",
        "deployment_type": "direct_code_deploy",
        "inbound_auth": {
            "type": "Cognito JWT",
            "user_pool_id": cognito_user_pool_id,
            "client_id": cognito_client_id,
        },
    }
    if memory_id and agent_key == "orchestrator":
        deployment_info["memory_id"] = memory_id

    info_file = os.path.join(SCRIPT_DIR, f"{agent_name}_deployment.json")
    with open(info_file, "w") as f:
        json.dump(deployment_info, f, indent=2)
    print(f"  [OK] Deployment info saved: {info_file}")

    # Cleanup staging
    staging_dir_path = os.path.join(SCRIPT_DIR, f".deploy_staging_{agent_key}")
    if os.path.exists(staging_dir_path):
        shutil.rmtree(staging_dir_path)
        print(f"  [OK] Staging cleaned up")

    return agent_arn


# ---------------------------------------------------------------------------
# List deployed agents
# ---------------------------------------------------------------------------

def list_agents(region: str) -> None:
    """List all CarDesign agent runtimes."""
    client = boto3.client("bedrock-agentcore-control", region_name=region)
    try:
        resp = client.list_agent_runtimes(maxResults=100)
        runtimes = resp.get("agentRuntimes", [])
        car_design = [r for r in runtimes if r.get("agentRuntimeName", "").startswith("car_design")]

        if not car_design:
            print("No Car Design agents found.")
            return

        print(f"\nCar Design Agents ({len(car_design)}):")
        print("-" * 80)
        for rt in car_design:
            print(f"  {rt['agentRuntimeName']:<30} {rt.get('status', 'UNKNOWN'):<15} {rt.get('agentRuntimeArn', '')}")
        print()
    except Exception as e:
        print(f"[ERROR] Could not list agents: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Deploy Car Design Explorer agents to Bedrock AgentCore Runtime"
    )
    parser.add_argument("--deploy-all", action="store_true", help="Deploy all 5 agents")
    parser.add_argument("--agent", type=str, choices=list(AGENTS.keys()), help="Deploy a single agent")
    parser.add_argument("--create-gateway", action="store_true", help="Create MCP Gateway only")
    parser.add_argument("--list", action="store_true", help="List deployed agents")
    parser.add_argument("--wire-lambda", action="store_true", help="Wire orchestrator ARN to Lambda")

    parser.add_argument("--cognito-user-pool-id", type=str, help="Cognito User Pool ID for inbound auth")
    parser.add_argument("--cognito-client-id", type=str, help="Cognito App Client ID for inbound auth")
    parser.add_argument("--agent-m2m-client-id", type=str, default="", help="M2M Cognito client ID for agent-to-agent JWT auth")
    parser.add_argument("--mcp-user-pool-id", type=str, help="MCP Gateway Cognito User Pool ID")
    parser.add_argument("--orchestrator-arn", type=str, help="Orchestrator ARN (for --wire-lambda)")
    parser.add_argument(
        "--model-id",
        type=str,
        default=os.environ.get("AGENT_MODEL_ID", ""),
        help="Primary Bedrock model ID used by all five agents (or AGENT_MODEL_ID)",
    )
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    parser.add_argument("--no-auto-update", action="store_true", help="Don't auto-update existing runtimes")

    args = parser.parse_args()

    # --- List ---
    if args.list:
        list_agents(args.region)
        return

    # --- Wire Lambda ---
    if args.wire_lambda:
        if not args.orchestrator_arn:
            print("[ERROR] --orchestrator-arn required for --wire-lambda")
            sys.exit(1)
        wire_lambda_orchestrator(args.orchestrator_arn, args.region)
        return

    # --- Create Gateway ---
    if args.create_gateway:
        if not args.mcp_user_pool_id:
            print("[ERROR] --mcp-user-pool-id required for --create-gateway")
            sys.exit(1)
        account_id = get_account_id()
        gw_id, gw_url = create_mcp_gateway(args.mcp_user_pool_id, args.region, account_id)
        config = {"gateway_id": gw_id, "gateway_url": f"{gw_url}/mcp"}
        config_file = os.path.join(SCRIPT_DIR, "mcp_gateway_config.json")
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\n[SUCCESS] MCP Gateway ready: {gw_url}/mcp")
        print(f"  Config saved: {config_file}")
        return

    # --- Deploy ---
    if not args.deploy_all and not args.agent:
        parser.print_help()
        sys.exit(1)

    if not args.cognito_user_pool_id or not args.cognito_client_id:
        print("[ERROR] --cognito-user-pool-id and --cognito-client-id required for deployment")
        sys.exit(1)
    if not args.agent_m2m_client_id or not args.mcp_user_pool_id:
        print("[ERROR] --agent-m2m-client-id and --mcp-user-pool-id required for deployment")
        sys.exit(1)
    if not args.model_id:
        print("[ERROR] --model-id or AGENT_MODEL_ID required for deployment")
        sys.exit(1)
    if not args.model_id.startswith("us.") or not all(
        char.isalnum() or char in "._:-" for char in args.model_id
    ):
        print("[ERROR] --model-id must be a US inference profile ID beginning with 'us.'")
        sys.exit(1)

    print("=" * 60)
    print("Car Design Explorer — Agent Deployment")
    print("  (Direct Code Deploy — no Docker required)")
    print("=" * 60)

    account_id = get_account_id()
    print(f"  Account: {account_id}")
    print(f"  Region:  {args.region}")
    print(f"  Model:   {args.model_id}")

    agents_to_deploy = DEPLOY_ORDER if args.deploy_all else [args.agent]

    # Step 1: Resolve CDK-owned least-privilege runtime roles.
    print("\n[1/5] Resolving per-agent runtime roles...")
    role_arns = resolve_execution_role_arns(
        agents_to_deploy,
        args.region,
        args.model_id,
    )
    for key, role_arn in role_arns.items():
        print(f"  [OK] {key}: {role_arn}")

    # Step 2: Create MCP Gateway (if deploying cost agent or all)
    gateway_url = ""
    gateway_id = ""

    if "cost" in agents_to_deploy and args.mcp_user_pool_id:
        print("\n[2/5] Creating MCP Gateway for Cost Agent...")
        gateway_id, gateway_url = create_mcp_gateway(args.mcp_user_pool_id, args.region, account_id)
    else:
        print("\n[2/5] Skipping MCP Gateway (not deploying cost agent or no --mcp-user-pool-id)")

    # Step 3: Create AgentCore Memory (if deploying orchestrator or all)
    memory_id = ""
    if "orchestrator" in agents_to_deploy:
        print("\n[3/5] Creating AgentCore Memory for Orchestrator...")
        memory_id = create_agentcore_memory(args.region)
    else:
        print("\n[3/5] Skipping Memory (not deploying orchestrator)")

    # Step 4: Deploy agents
    print("\n[4/5] Deploying agents...")
    deployed_arns = {}

    for agent_key in agents_to_deploy:
        try:
            specialist_runtime_arns = None
            if agent_key == "orchestrator":
                specialist_runtime_arns = resolve_specialist_runtime_arns(
                    args.region, account_id
                )
            arn = deploy_single_agent(
                agent_key=agent_key,
                account_id=account_id,
                region=args.region,
                role_arn=role_arns[agent_key],
                cognito_user_pool_id=args.cognito_user_pool_id,
                cognito_client_id=args.cognito_client_id,
                model_id=args.model_id,
                agent_m2m_client_id=args.agent_m2m_client_id,
                mcp_user_pool_id=args.mcp_user_pool_id,
                memory_id=memory_id,
                gateway_url=gateway_url,
                specialist_runtime_arns=specialist_runtime_arns,
                auto_update=not args.no_auto_update,
            )
            deployed_arns[agent_key] = arn
        except Exception as e:
            shutil.rmtree(
                os.path.join(SCRIPT_DIR, f".deploy_staging_{agent_key}"),
                ignore_errors=True,
            )
            print(f"\n  [ERROR] Failed to deploy {agent_key}: {e}")
            print("  Aborting deployment to avoid stale roles or specialist endpoints.")
            sys.exit(1)

    # Step 5: Wire orchestrator to Lambda
    orchestrator_arn = deployed_arns.get("orchestrator", "")
    if orchestrator_arn:
        print("\n[5/5] Wiring orchestrator ARN to Lambda...")
        wire_lambda_orchestrator(orchestrator_arn, args.region)
    else:
        print("\n[5/5] Skipping Lambda wiring (no orchestrator ARN)")

    # Summary
    print("\n" + "=" * 60)
    print("DEPLOYMENT SUMMARY")
    print("=" * 60)
    for key, arn in deployed_arns.items():
        status = "DEPLOYED" if arn else "FAILED"
        print(f"  {AGENTS[key]['display']:<25} {status:<10} {arn}")
    if memory_id:
        print(f"\n  Memory ID: {memory_id}")
    if gateway_url:
        print(f"  MCP Gateway: {gateway_url}/mcp")
    if orchestrator_arn:
        print(f"  Orchestrator ARN: {orchestrator_arn}")

    # Preserve validated same-account/same-Region state on single-agent deploys.
    prior_config: dict = {}
    config_file = os.path.join(SCRIPT_DIR, "deployment_config.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as existing_file:
                candidate = json.load(existing_file)
            if (
                candidate.get("account_id") == account_id
                and candidate.get("region") == args.region
            ):
                prior_config = candidate
        except (OSError, ValueError, TypeError):
            pass

    merged_agents = dict(prior_config.get("agents", {}))
    merged_agents.update(deployed_arns)
    merged_roles = dict(prior_config.get("execution_roles", {}))
    merged_roles.update(role_arns)
    saved_orchestrator_arn = (
        orchestrator_arn
        or merged_agents.get("orchestrator", "")
        or prior_config.get("orchestrator_arn", "")
    )

    deploy_config = {
        "account_id": account_id,
        "region": args.region,
        "agent_model_id": args.model_id,
        "agents": merged_agents,
        "execution_roles": merged_roles,
        "memory_id": memory_id or prior_config.get("memory_id", ""),
        "gateway_id": gateway_id or prior_config.get("gateway_id", ""),
        "gateway_url": (
            f"{gateway_url}/mcp" if gateway_url
            else prior_config.get("gateway_url", "")
        ),
        "orchestrator_arn": saved_orchestrator_arn,
        "cognito": {
            "user_pool_id": args.cognito_user_pool_id,
            "client_id": args.cognito_client_id,
        },
    }
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(deploy_config, f, indent=2)
    print(f"\n  Full config saved: {config_file}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
