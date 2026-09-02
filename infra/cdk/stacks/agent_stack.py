"""Agent Stack — Deploy agents to Bedrock AgentCore Runtime.

Uses CfnResource for AgentCore Runtime resources since L2 constructs
may not yet exist. Each agent is deployed as a separate runtime with
its own container image built from the backend code.
"""

import os

import aws_cdk as cdk
from aws_cdk import (
    aws_dynamodb as dynamodb,
    aws_iam as iam,
    aws_kms as kms,
    aws_s3 as s3,
    aws_secretsmanager as secretsmanager,
)
from constructs import Construct

# Agent definitions
AGENTS = [
    {
        "name": "orchestrator",
        "display_name": "Car Design Orchestrator Agent",
        "description": "Central coordinator for multi-agent design exploration",
        "entry_point": "backend/agents/orchestrator_agent.py",
    },
    {
        "name": "aero",
        "display_name": "Car Design Aero Agent",
        "description": "Aerodynamic KPI and surface variable prediction using MLSimKit",
        "entry_point": "backend/agents/aero_agent.py",
    },
    {
        "name": "structural",
        "display_name": "Car Design Structural Agent",
        "description": "Structural feasibility evaluation for car body variants",
        "entry_point": "backend/agents/structural_agent.py",
    },
    {
        "name": "cost",
        "display_name": "Car Design Cost Agent",
        "description": "Manufacturing cost estimation with MCP server integration",
        "entry_point": "backend/agents/cost_agent.py",
    },
    {
        "name": "geometry",
        "display_name": "Car Design Geometry Agent",
        "description": "Geometry modification with Stable Diffusion 3.5 Large preview and trimesh 3D mesh operations",
        "entry_point": "backend/agents/geometry_agent.py",
    },
]

DEFAULT_AGENT_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_IMAGE_MODEL_ID = "stability.sd3-5-large-v1:0"
DEFAULT_IMAGE_MODEL_REGION = "us-west-2"

AGENT_MODEL_ID = os.environ.get("AGENT_MODEL_ID", DEFAULT_AGENT_MODEL_ID)
IMAGE_MODEL_ID = os.environ.get("IMAGE_MODEL_ID", DEFAULT_IMAGE_MODEL_ID)
IMAGE_MODEL_REGION = os.environ.get("IMAGE_MODEL_REGION", DEFAULT_IMAGE_MODEL_REGION)

if not AGENT_MODEL_ID.startswith("us.") or not all(
    char.isalnum() or char in "._:-" for char in AGENT_MODEL_ID
):
    raise ValueError(
        "AGENT_MODEL_ID must be a US geographic inference profile ID beginning with 'us.'"
    )
if not IMAGE_MODEL_ID.startswith("stability.") or not all(
    char.isalnum() or char in "._:-" for char in IMAGE_MODEL_ID
):
    raise ValueError("IMAGE_MODEL_ID must be a Stability AI foundation model ID")
if not IMAGE_MODEL_REGION.startswith("us-") or not all(
    char.isalnum() or char == "-" for char in IMAGE_MODEL_REGION
):
    raise ValueError("IMAGE_MODEL_REGION must be a valid US AWS region")

MODEL_IDS = {agent["name"]: AGENT_MODEL_ID for agent in AGENTS}
US_INFERENCE_REGIONS = ("us-east-1", "us-east-2", "us-west-2")


class AgentStack(cdk.Stack):
    def __init__(
        self,
        scope: Construct,
        id: str,
        model_bucket: s3.IBucket,
        mcp_secret: secretsmanager.ISecret,
        mcp_secret_key: kms.IKey,
        variant_cache_table: dynamodb.ITable,
        dynamo_key: kms.IKey,
        **kwargs,
    ):
        super().__init__(scope, id, **kwargs)

        # CDK is the sole owner of runtime roles. Deployment/build operations run
        # under the invoking human or CI principal and are never granted here.
        self.agent_roles: dict[str, iam.Role] = {}
        for agent_def in AGENTS:
            agent_name = agent_def["name"]
            runtime_name = f"car_design_{agent_name}"
            role = iam.Role(
                self,
                f"{agent_name.title()}RuntimeRole",
                role_name=f"CarDesign{agent_name.title()}RuntimeRole",
                assumed_by=iam.ServicePrincipal(
                    "bedrock-agentcore.amazonaws.com",
                    conditions={
                        "StringEquals": {"aws:SourceAccount": cdk.Aws.ACCOUNT_ID},
                        "ArnLike": {
                            "aws:SourceArn": (
                                f"arn:aws:bedrock-agentcore:{cdk.Aws.REGION}:"
                                f"{cdk.Aws.ACCOUNT_ID}:*"
                            )
                        },
                    },
                ),
            )
            self._grant_runtime_baseline(role, runtime_name)
            self._grant_cross_region_model(role, MODEL_IDS[agent_name])
            self.agent_roles[agent_name] = role

        # Orchestrator: outbound A2A OAuth credentials only. Specialist runtime
        # endpoints are baked in at deploy time, so no AgentCore control-plane
        # permissions are required at runtime.
        self.agent_roles["orchestrator"].add_to_policy(iam.PolicyStatement(
            actions=["secretsmanager:GetSecretValue"],
            resources=[
                f"arn:aws:secretsmanager:{cdk.Aws.REGION}:{cdk.Aws.ACCOUNT_ID}:"
                "secret:car-design/agent-oauth-credentials-*"
            ],
        ))

        # Aero: exact model/geometry reads, visualization writes, and variant
        # cache access. No other application table or bucket is accessible.
        aero_role = self.agent_roles["aero"]
        aero_role.add_to_policy(iam.PolicyStatement(
            actions=["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:Query", "dynamodb:Scan"],
            resources=[variant_cache_table.table_arn],
        ))
        aero_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "kms:Encrypt", "kms:Decrypt", "kms:ReEncrypt*",
                "kms:GenerateDataKey*", "kms:DescribeKey",
            ],
            resources=[dynamo_key.key_arn],
            conditions={
                "StringEquals": {
                    "kms:ViaService": f"dynamodb.{cdk.Aws.REGION}.amazonaws.com",
                    "kms:CallerAccount": cdk.Aws.ACCOUNT_ID,
                },
            },
        ))
        aero_role.add_to_policy(iam.PolicyStatement(
            actions=["s3:GetObject"],
            resources=[
                model_bucket.arn_for_objects("kpi/best_model.pt"),
                model_bucket.arn_for_objects("surface/best_model.pt"),
                model_bucket.arn_for_objects("slices/ae_best_model.pt"),
                model_bucket.arn_for_objects("slices/mgn_last_model.pt"),
                model_bucket.arn_for_objects("geometries/*"),
            ],
        ))
        aero_role.add_to_policy(iam.PolicyStatement(
            actions=["s3:PutObject"],
            resources=[model_bucket.arn_for_objects("visualizations/*")],
        ))

        # Structural: read-only access to source geometries.
        self.agent_roles["structural"].add_to_policy(iam.PolicyStatement(
            actions=["s3:GetObject"],
            resources=[model_bucket.arn_for_objects("geometries/*")],
        ))

        # Cost: only the MCP Gateway credential secret and its KMS key.
        cost_role = self.agent_roles["cost"]
        cost_role.add_to_policy(iam.PolicyStatement(
            actions=["secretsmanager:GetSecretValue"],
            resources=[mcp_secret.secret_arn],
        ))
        cost_role.add_to_policy(iam.PolicyStatement(
            actions=["kms:Decrypt"],
            resources=[mcp_secret_key.key_arn],
            conditions={
                "StringEquals": {
                    "kms:ViaService": f"secretsmanager.{cdk.Aws.REGION}.amazonaws.com",
                    "kms:CallerAccount": cdk.Aws.ACCOUNT_ID,
                },
            },
        ))

        # Geometry: read/write only its geometry prefix, invoke the configured
        # cross-region image model, and use the AWS-managed Code Interpreter resource.
        geometry_role = self.agent_roles["geometry"]
        geometry_role.add_to_policy(iam.PolicyStatement(
            actions=["s3:GetObject", "s3:PutObject"],
            resources=[model_bucket.arn_for_objects("geometries/*")],
        ))
        geometry_role.add_to_policy(iam.PolicyStatement(
            actions=["s3:ListBucket"],
            resources=[model_bucket.bucket_arn],
            conditions={"StringLike": {"s3:prefix": ["geometries/*"]}},
        ))
        geometry_role.add_to_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModel"],
            resources=[
                f"arn:aws:bedrock:{IMAGE_MODEL_REGION}::"
                f"foundation-model/{IMAGE_MODEL_ID}"
            ],
        ))
        geometry_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "bedrock-agentcore:StartCodeInterpreterSession",
                "bedrock-agentcore:InvokeCodeInterpreter",
                "bedrock-agentcore:GetCodeInterpreterSession",
                "bedrock-agentcore:StopCodeInterpreterSession",
            ],
            resources=[
                f"arn:aws:bedrock-agentcore:{cdk.Aws.REGION}:aws:"
                "code-interpreter/aws.codeinterpreter.v1"
            ],
        ))

        # Create AgentCore Runtime for each agent
        # Note: AgentCore Runtime resources are created via CLI/SDK since
        # CloudFormation support may be limited. This stack outputs the
        # configuration needed for deployment scripts.
        for agent_def in AGENTS:
            cdk.CfnOutput(
                self,
                f"{agent_def['name'].title()}AgentConfig",
                value=cdk.Fn.join(",", [
                    agent_def["display_name"],
                    agent_def["description"],
                    agent_def["entry_point"],
                ]),
                description=f"Configuration for {agent_def['display_name']}",
            )

        # Per-agent role ARNs are consumed by deploy_agents.py. The deployment
        # script resolves these outputs and never creates or mutates IAM roles.
        for agent_name, role in self.agent_roles.items():
            cdk.CfnOutput(
                self,
                f"{agent_name.title()}AgentRoleArn",
                value=role.role_arn,
            )
        cdk.CfnOutput(
            self,
            "AgentModelId",
            value=AGENT_MODEL_ID,
            description="Primary Bedrock model authorized for all five agents",
        )
        cdk.CfnOutput(
            self,
            "McpGatewaySecretName",
            value=mcp_secret.secret_name,
        )
        cdk.CfnOutput(
            self,
            "DeploymentNote",
            value="Use deploy_agents.py; runtime roles are managed by this stack",
        )

    def _grant_runtime_baseline(self, role: iam.Role, runtime_name: str) -> None:
        """Grant only infrastructure permissions required by AgentCore Runtime."""
        log_group_arn = (
            f"arn:aws:logs:{cdk.Aws.REGION}:{cdk.Aws.ACCOUNT_ID}:"
            f"log-group:/aws/bedrock-agentcore/runtimes/{runtime_name}-*"
        )
        role.add_to_policy(iam.PolicyStatement(
            actions=["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
            resources=[
                f"arn:aws:ecr:{cdk.Aws.REGION}:{cdk.Aws.ACCOUNT_ID}:"
                f"repository/bedrock-agentcore-{runtime_name}"
            ],
        ))
        # ECR authorization tokens and telemetry APIs do not support
        # resource-level permissions; actions and metric namespace are bounded.
        role.add_to_policy(iam.PolicyStatement(
            actions=["ecr:GetAuthorizationToken"],
            resources=["*"],
        ))
        role.add_to_policy(iam.PolicyStatement(
            actions=["logs:CreateLogGroup", "logs:DescribeLogStreams", "logs:PutResourcePolicy"],
            resources=[log_group_arn],
        ))
        role.add_to_policy(iam.PolicyStatement(
            actions=["logs:CreateLogStream", "logs:PutLogEvents"],
            resources=[f"{log_group_arn}:log-stream:*"],
        ))
        role.add_to_policy(iam.PolicyStatement(
            actions=["logs:DescribeLogGroups"],
            resources=[
                f"arn:aws:logs:{cdk.Aws.REGION}:{cdk.Aws.ACCOUNT_ID}:log-group:*"
            ],
        ))
        role.add_to_policy(iam.PolicyStatement(
            actions=[
                "xray:PutTraceSegments", "xray:PutTelemetryRecords",
                "xray:GetSamplingRules", "xray:GetSamplingTargets",
            ],
            resources=["*"],
        ))
        role.add_to_policy(iam.PolicyStatement(
            actions=["cloudwatch:PutMetricData"],
            resources=["*"],
            conditions={
                "StringEquals": {"cloudwatch:namespace": "bedrock-agentcore"}
            },
        ))

    def _grant_cross_region_model(self, role: iam.Role, profile_id: str) -> None:
        """Grant one geographic inference profile and its exact US models."""
        model_id = profile_id.removeprefix("us.")
        profile_arn = (
            f"arn:aws:bedrock:{cdk.Aws.REGION}:{cdk.Aws.ACCOUNT_ID}:"
            f"inference-profile/{profile_id}"
        )
        actions = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
        role.add_to_policy(iam.PolicyStatement(
            actions=actions,
            resources=[profile_arn],
        ))
        role.add_to_policy(iam.PolicyStatement(
            actions=actions,
            resources=[
                f"arn:aws:bedrock:{region}::foundation-model/{model_id}"
                for region in US_INFERENCE_REGIONS
            ],
            conditions={
                "StringEquals": {"bedrock:InferenceProfileArn": profile_arn}
            },
        ))
