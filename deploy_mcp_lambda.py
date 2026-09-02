#!/usr/bin/env python3
"""
Deploy MCP Lambda target for the Car Design Cost Agent's AgentCore Gateway.

Creates:
1. Lambda function (CarDesignCostMCPHandler) that reads from DynamoDB
2. Registers it as a gateway target on the existing MCP Gateway
3. Adds Lambda invoke permission for the gateway

Usage:
    python deploy_mcp_lambda.py \
      --gateway-id cardesignmcpgateway-09135582-mztfzxwfb0 \
      --region us-east-1
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
import zipfile

import boto3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAMBDA_NAME = "CarDesignCostMCPHandler"
ROLE_NAME = "CarDesignCostMCPLambdaRole"


def get_account_id() -> str:
    return boto3.client("sts").get_caller_identity()["Account"]


def ensure_lambda_role(account_id: str, region: str) -> str:
    """Create or reuse the Lambda execution role."""
    iam = boto3.client("iam")

    trust = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    }

    try:
        resp = iam.get_role(RoleName=ROLE_NAME)
        role_arn = resp["Role"]["Arn"]
        print(f"  [OK] Using existing role: {role_arn}")
    except iam.exceptions.NoSuchEntityException:
        resp = iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description="Lambda role for Car Design Cost MCP handler",
        )
        role_arn = resp["Role"]["Arn"]
        print(f"  [OK] Created role: {role_arn}")

    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "dynamodb:GetItem", "dynamodb:Query", "dynamodb:Scan",
                ],
                "Resource": [
                    f"arn:aws:dynamodb:{region}:{account_id}:table/CarDesignCostParameters",
                    f"arn:aws:dynamodb:{region}:{account_id}:table/CarDesignExternalCostData",
                ],
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents",
                ],
                "Resource": ["*"],
            },
        ],
    }
    iam.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName="CarDesignCostMCPPermissions",
        PolicyDocument=json.dumps(policy),
    )

    # Attach basic execution role
    try:
        iam.attach_role_policy(
            RoleName=ROLE_NAME,
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
        )
    except Exception:
        pass

    print("  [WAIT] Waiting 10s for IAM propagation...")
    time.sleep(10)  # nosemgrep
    return role_arn


def create_lambda_zip() -> bytes:
    """Create a zip with the Lambda handler code."""
    handler_path = os.path.join(SCRIPT_DIR, "backend", "mcp_servers", "lambda_mcp_handler.py")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(handler_path, "lambda_mcp_handler.py")
    return buf.getvalue()


def deploy_lambda(region: str, role_arn: str) -> str:
    """Create or update the MCP Lambda function. Returns the Lambda ARN."""
    client = boto3.client("lambda", region_name=region)
    zip_bytes = create_lambda_zip()

    try:
        resp = client.get_function(FunctionName=LAMBDA_NAME)
        lambda_arn = resp["Configuration"]["FunctionArn"]
        print(f"  [OK] Updating existing Lambda: {LAMBDA_NAME}")
        client.update_function_code(
            FunctionName=LAMBDA_NAME,
            ZipFile=zip_bytes,
        )
        # Wait for update
        time.sleep(5)  # nosemgrep
        client.update_function_configuration(
            FunctionName=LAMBDA_NAME,
            Timeout=30,
            MemorySize=256,
            Environment={"Variables": {"AWS_REGION_OVERRIDE": region}},
        )
    except client.exceptions.ResourceNotFoundException:
        print(f"  Creating Lambda: {LAMBDA_NAME}")
        resp = client.create_function(
            FunctionName=LAMBDA_NAME,
            Runtime="python3.12",
            Role=role_arn,
            Handler="lambda_mcp_handler.handler",
            Code={"ZipFile": zip_bytes},
            Timeout=30,
            MemorySize=256,
            Environment={"Variables": {"AWS_REGION_OVERRIDE": region}},
            Description="MCP target Lambda for Car Design Cost data (DynamoDB)",
        )
        lambda_arn = resp["FunctionArn"]
        print(f"  [OK] Lambda created: {lambda_arn}")
        # Wait for active
        print("  Waiting for Lambda to become Active...")
        waiter = client.get_waiter("function_active_v2")
        waiter.wait(FunctionName=LAMBDA_NAME)

    print(f"  [OK] Lambda ARN: {lambda_arn}")
    return lambda_arn


def update_gateway_role_for_lambda(account_id: str, region: str, lambda_arn: str):
    """Add lambda:InvokeFunction permission to the existing CarDesignMcpGatewayRole."""
    iam = boto3.client("iam")
    role_name = "CarDesignMcpGatewayRole"

    # Read existing policy, add Lambda invoke permission
    try:
        existing = iam.get_role_policy(
            RoleName=role_name, PolicyName="CarDesignMcpGatewayPermissions"
        )
        policy = json.loads(existing["PolicyDocument"]) if isinstance(
            existing["PolicyDocument"], str
        ) else existing["PolicyDocument"]
    except Exception:
        policy = {"Version": "2012-10-17", "Statement": []}

    # Check if Lambda invoke already present
    has_lambda = any(
        "lambda:InvokeFunction" in str(s.get("Action", ""))
        for s in policy.get("Statement", [])
    )
    if not has_lambda:
        policy["Statement"].append({
            "Sid": "LambdaInvoke",
            "Effect": "Allow",
            "Action": "lambda:InvokeFunction",
            "Resource": lambda_arn,
        })
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName="CarDesignMcpGatewayPermissions",
            PolicyDocument=json.dumps(policy),
        )
        print(f"  [OK] Added lambda:InvokeFunction to {role_name}")
        print("  [WAIT] Waiting 30s for IAM policy propagation...")
        time.sleep(30)  # nosemgrep
    else:
        print(f"  [OK] {role_name} already has lambda:InvokeFunction")


def register_gateway_target(gateway_id: str, lambda_arn: str, region: str):
    """Register the Lambda as a gateway target with full tool schema."""
    client = boto3.client("bedrock-agentcore-control", region_name=region)

    # Check for existing targets
    deleted_any = False
    try:
        resp = client.list_gateway_targets(gatewayIdentifier=gateway_id, maxResults=50)
        targets = resp.get("items", resp.get("gatewayTargetSummaries", []))
        if targets:
            print(f"  [INFO] Gateway already has {len(targets)} target(s):")
            for t in targets:
                print(f"    - {t.get('name', t.get('targetId', 'unknown'))}")
            # Delete existing targets to re-register with updated schema
            for t in targets:
                tid = t.get("targetId", "")
                if tid:
                    try:
                        client.delete_gateway_target(
                            gatewayIdentifier=gateway_id, targetId=tid
                        )
                        print(f"    [OK] Deleted old target: {tid}")
                        deleted_any = True
                    except Exception as e:
                        print(f"    [WARN] Could not delete target {tid}: {e}")
    except Exception as e:
        print(f"  [WARN] Could not list existing targets: {e}")

    # Wait for deletes to propagate before creating with the same name
    if deleted_any:
        print("  [WAIT] Waiting 20s for target deletion to propagate...")
        time.sleep(20)  # nosemgrep

    # Build the tool schema — all 15 tools from both MCP servers
    tool_schema = [
        # --- Internal Cost Parameters (10 tools) ---
        {
            "name": "get_material_cost",
            "description": "Get the base manufacturing cost per kilogram for a material.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "material": {"type": "string", "description": "Material type: steel, aluminum, or carbon_fiber"}
                },
                "required": ["material"]
            }
        },
        {
            "name": "get_all_material_costs",
            "description": "Get base manufacturing costs for all available materials.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "get_stamping_parameters",
            "description": "Get stamping cost parameters (cost per operation, multi-stage multiplier).",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "get_tooling_base_cost",
            "description": "Get all tooling cost parameters (base die cost, trim die, flange die, fixtures, maintenance).",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "get_assembly_parameters",
            "description": "Get assembly cost parameters (welding cost per meter, etc.).",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "get_complexity_multipliers",
            "description": "Get complexity-based cost multipliers (low, medium, high).",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "get_surface_treatment_costs",
            "description": "Get surface treatment cost parameters (e-coat, primer, paint, galvanizing, anodizing).",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "get_quality_parameters",
            "description": "Get quality inspection and rework cost parameters.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "get_logistics_parameters",
            "description": "Get logistics and overhead cost parameters (transport, packaging, warehousing, energy).",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "get_all_cost_parameters",
            "description": "Get all cost parameters in a single call — materials, stamping, tooling, assembly, multipliers, surface treatment, quality, logistics.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        # --- External Cost Data (5 tools) ---
        {
            "name": "get_material_market_price",
            "description": "Retrieve current commodity market pricing for a material with regional adjustments.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "material": {"type": "string", "description": "Material type: steel, aluminum, carbon_fiber, high_strength_steel, magnesium, titanium"},
                    "region": {"type": "string", "description": "Manufacturing region: north_america, europe, asia_pacific"}
                },
                "required": ["material"]
            }
        },
        {
            "name": "get_supplier_quotes",
            "description": "Query supplier pricing for manufacturing components and tooling with lead times and volume adjustments.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "component_type": {"type": "string", "description": "Component: stamping_die, welding_fixture, paint_booth_time, assembly_jig, trim_die, inspection_fixture"},
                    "complexity": {"type": "string", "description": "Geometry complexity: low, medium, high"},
                    "quantity": {"type": "integer", "description": "Number of units"}
                },
                "required": ["component_type"]
            }
        },
        {
            "name": "get_historical_cost_benchmarks",
            "description": "Retrieve historical manufacturing cost benchmarks with statistical distributions for similar designs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "material": {"type": "string", "description": "Material type: steel, aluminum, carbon_fiber"},
                    "complexity": {"type": "string", "description": "Complexity level: low, medium, high"},
                    "body_style": {"type": "string", "description": "Vehicle body style: sedan, suv, coupe, hatchback, truck"}
                }
            }
        },
        {
            "name": "get_regional_cost_factors",
            "description": "Get regional manufacturing cost multipliers for labor, energy, logistics, and overhead.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "Manufacturing region: north_america, europe, asia_pacific, south_america, middle_east"}
                }
            }
        },
        {
            "name": "get_volume_discount_schedule",
            "description": "Get volume-based pricing tiers for material procurement with discount percentages.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "material": {"type": "string", "description": "Material type: steel, aluminum, carbon_fiber"}
                }
            }
        },
    ]

    target_config = {
        "mcp": {
            "lambda": {
                "lambdaArn": lambda_arn,
                "toolSchema": {
                    "inlinePayload": tool_schema,
                },
            }
        }
    }

    credential_config = [{"credentialProviderType": "GATEWAY_IAM_ROLE"}]

    print(f"  Registering gateway target with {len(tool_schema)} tools...")
    # Retry with backoff — IAM policy propagation can take up to 60s
    last_error = None
    for attempt in range(4):
        try:
            resp = client.create_gateway_target(
                gatewayIdentifier=gateway_id,
                name="CarDesignCostMCPTarget",
                description="Lambda target serving 15 cost MCP tools from DynamoDB",
                targetConfiguration=target_config,
                credentialProviderConfigurations=credential_config,
            )
            target_id = resp.get("targetId", "unknown")
            print(f"  [OK] Gateway target created: {target_id}")
            last_error = None
            break
        except Exception as e:
            last_error = e
            err_str = str(e)
            if "lacks permission" in err_str and attempt < 3:
                wait = 15 * (attempt + 1)
                print(f"  [RETRY] IAM not propagated yet, waiting {wait}s... (attempt {attempt+1}/4)")
                time.sleep(wait)
            elif "ConflictException" in err_str or "already exists" in err_str:
                # Target with same name still being deleted — wait and retry
                if attempt < 3:
                    wait = 15 * (attempt + 1)
                    print(f"  [RETRY] Target still deleting, waiting {wait}s... (attempt {attempt+1}/4)")
                    time.sleep(wait)
                else:
                    raise
            else:
                raise

    if last_error:
        raise last_error

    # Wait for target to be ready
    print("  Waiting for target to become READY...")
    for i in range(20):
        try:
            status_resp = client.get_gateway_target(
                gatewayIdentifier=gateway_id, targetId=target_id
            )
            status = status_resp.get("status", "UNKNOWN")
            print(f"    Attempt {i+1}/20 — {status}")
            if status == "READY":
                break
            if status in ("FAILED", "CREATE_FAILED"):
                print(f"    [ERROR] Target creation failed")
                break
        except Exception:
            pass
        time.sleep(5)  # nosemgrep

    return target_id


def add_lambda_permission(lambda_arn: str, gateway_id: str, region: str):
    """Add permission for AgentCore Gateway to invoke the Lambda."""
    account_id = get_account_id()
    gateway_arn = f"arn:aws:bedrock-agentcore:{region}:{account_id}:gateway/{gateway_id}"

    client = boto3.client("lambda", region_name=region)
    statement_id = f"AllowBedrockAgentCore-{gateway_id.replace('-', '')[:20]}"

    try:
        client.add_permission(
            FunctionName=lambda_arn,
            StatementId=statement_id,
            Action="lambda:InvokeFunction",
            Principal="bedrock-agentcore.amazonaws.com",
            SourceArn=gateway_arn,
        )
        print(f"  [OK] Lambda permission added for gateway")
    except client.exceptions.ResourceConflictException:
        print(f"  [OK] Lambda permission already exists")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy MCP Lambda target for Car Design Cost Agent Gateway"
    )
    parser.add_argument(
        "--gateway-id", required=True,
        help="Existing MCP Gateway ID (e.g. cardesignmcpgateway-09135582-mztfzxwfb0)",
    )
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    print("=" * 60)
    print("Car Design Explorer — MCP Lambda Target Deployment")
    print("=" * 60)

    account_id = get_account_id()
    print(f"Account: {account_id}, Region: {args.region}")

    # Step 1: IAM role for Lambda
    print("\n[1/5] Ensuring Lambda execution role...")
    role_arn = ensure_lambda_role(account_id, args.region)

    # Step 2: Deploy Lambda
    print("\n[2/5] Deploying Lambda function...")
    lambda_arn = deploy_lambda(args.region, role_arn)

    # Step 3: Update gateway role to allow Lambda invoke
    print("\n[3/5] Updating gateway role with Lambda invoke permission...")
    update_gateway_role_for_lambda(account_id, args.region, lambda_arn)

    # Step 4: Add Lambda resource policy for gateway
    print("\n[4/5] Adding Lambda permission for gateway...")
    add_lambda_permission(lambda_arn, args.gateway_id, args.region)

    # Step 5: Register as gateway target
    print("\n[5/5] Registering Lambda as gateway target...")
    target_id = register_gateway_target(args.gateway_id, lambda_arn, args.region)

    print("\n" + "=" * 60)
    print("[SUCCESS] MCP Lambda target deployed and registered")
    print(f"  Lambda: {LAMBDA_NAME} ({lambda_arn})")
    print(f"  Gateway: {args.gateway_id}")
    print(f"  Target: {target_id}")
    print(f"  Tools: 15 (10 internal + 5 external)")
    print("=" * 60)
    print("\nNext steps:")
    print(f"  1. Redeploy cost agent:")
    print(f"     python3 deploy_agents.py --agent cost \\")
    print(f"       --cognito-user-pool-id us-east-1_BnDLur150 \\")
    print(f"       --cognito-client-id cf3qjt93tqocdhtvv5539t8mt \\")
    print(f"       --mcp-user-pool-id us-east-1_3mhz73JyW \\")
    print(f"       --region {args.region}")
    print(f"  2. Test: 'Estimate the manufacturing cost for run_125'")


if __name__ == "__main__":
    main()
