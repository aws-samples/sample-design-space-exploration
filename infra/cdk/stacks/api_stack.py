"""API Stack — WebSocket API Gateway and Lambda handler."""

import aws_cdk as cdk
from aws_cdk import (
    aws_apigatewayv2 as apigwv2,
    aws_cognito as cognito,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_kms as kms,
    aws_lambda as lambda_,
    aws_logs as logs,
    aws_sqs as sqs,
)
from constructs import Construct


class ApiStack(cdk.Stack):
    def __init__(
        self,
        scope: Construct,
        id: str,
        user_pool: cognito.UserPool,
        vpc: ec2.IVpc,
        lambda_security_group: ec2.ISecurityGroup,
        dynamo_key: kms.IKey,
        **kwargs,
    ):
        super().__init__(scope, id, **kwargs)

        # --- KMS key for the Lambda dead-letter queue (mirrors SQSKMSKey) ---
        dlq_key = kms.Key(
            self, "LambdaDLQKMSKey",
            description="KMS key for Car Design Lambda DLQ encryption",
            enable_key_rotation=True,
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )

        # Dead-letter queue for failed async invocations
        self.dlq = sqs.Queue(
            self, "WSHandlerDLQ",
            queue_name="CarDesignWSHandler-DLQ",
            encryption=sqs.QueueEncryption.KMS,
            encryption_master_key=dlq_key,
            retention_period=cdk.Duration.days(14),
            enforce_ssl=True,
        )

        # --- KMS key for CloudWatch Logs encryption (mirrors CloudWatchLogsKMSKey) ---
        logs_key = kms.Key(
            self, "LogsKMSKey",
            description="KMS key for Car Design CloudWatch Logs encryption",
            enable_key_rotation=True,
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )
        # CloudWatch Logs service must be allowed to use the key.
        logs_key.add_to_resource_policy(iam.PolicyStatement(
            sid="AllowCloudWatchLogs",
            effect=iam.Effect.ALLOW,
            principals=[iam.ServicePrincipal(f"logs.{cdk.Aws.REGION}.amazonaws.com")],
            actions=[
                "kms:Encrypt", "kms:Decrypt", "kms:ReEncrypt*",
                "kms:GenerateDataKey*", "kms:DescribeKey", "kms:CreateGrant",
            ],
            resources=["*"],
            conditions={
                "ArnLike": {
                    "kms:EncryptionContext:aws:logs:arn":
                        f"arn:aws:logs:{cdk.Aws.REGION}:{cdk.Aws.ACCOUNT_ID}:*",
                },
            },
        ))

        # Explicit, KMS-encrypted log group for the WebSocket handler.
        handler_log_group = logs.LogGroup(
            self, "WSHandlerLogGroup",
            log_group_name="/aws/lambda/CarDesignWSHandler",
            encryption_key=logs_key,
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )

        # Lambda handler for WebSocket — deployed into the VPC private subnets
        # with egress via NAT so it can reach AWS service endpoints.
        self.handler_fn = lambda_.Function(
            self, "WSHandler",
            function_name="CarDesignWSHandler",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handler.handler",
            code=lambda_.Code.from_asset("../../backend/lambda_handler"),
            timeout=cdk.Duration.seconds(600),
            memory_size=256,
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            security_groups=[lambda_security_group],
            dead_letter_queue=self.dlq,
            log_group=handler_log_group,
            environment={
                "CONNECTIONS_TABLE": "CarDesignWSConnections",
                "ORCHESTRATOR_RUNTIME_ARN": "",
                # Secrets Manager resource name; credential material is fetched at runtime.
                "OAUTH_SECRET_NAME": "car-design/agent-oauth-credentials",  # nosec B105
                "GEOMETRY_S3_BUCKET": f"car-design-explorer-models-{cdk.Aws.ACCOUNT_ID}",
            },
        )

        # Allow the handler to publish failed async invocations to the DLQ
        # (grants sqs:SendMessage + kms:GenerateDataKey/Decrypt on the DLQ key).
        self.dlq.grant_send_messages(self.handler_fn)

        # Disable async retry — Phase 2 is invoked as Event (async). AWS retries
        # failed async invocations up to 2 times by default. If an agent is down,
        # this causes 3 × timeout = up to 30 min of dead attempts. Set retries=0
        # so a dead agent fails fast with a single attempt.
        self.handler_fn.configure_async_invoke(
            retry_attempts=0,
        )

        # Grant Lambda permissions
        self.handler_fn.add_to_role_policy(iam.PolicyStatement(
            actions=["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:DeleteItem"],
            resources=[f"arn:aws:dynamodb:{cdk.Aws.REGION}:{cdk.Aws.ACCOUNT_ID}:table/CarDesignWSConnections"],
        ))
        # The connections table uses the Data stack's customer-managed KMS key.
        # DynamoDB requires the caller's identity policy to allow use of that key.
        self.handler_fn.add_to_role_policy(iam.PolicyStatement(
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
        # OAuth is mandatory for AgentCore invocation, so the handler needs no
        # bedrock-agentcore InvokeAgentRuntime IAM permission. Scope secret read
        # to the one deployment-managed OAuth secret (Secrets Manager appends a
        # six-character suffix to its ARN).
        self.handler_fn.add_to_role_policy(iam.PolicyStatement(
            actions=["secretsmanager:GetSecretValue"],
            resources=[
                f"arn:aws:secretsmanager:{cdk.Aws.REGION}:{cdk.Aws.ACCOUNT_ID}:"
                "secret:car-design/agent-oauth-credentials-*"
            ],
        ))
        self.handler_fn.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:PutObject", "s3:GetObject"],
            resources=[
                f"arn:aws:s3:::car-design-explorer-models-{cdk.Aws.ACCOUNT_ID}/geometries/*",
                f"arn:aws:s3:::car-design-explorer-models-{cdk.Aws.ACCOUNT_ID}/visualizations/*",
                f"arn:aws:s3:::car-design-explorer-models-{cdk.Aws.ACCOUNT_ID}/predictions/*",
            ],
        ))
        # Allow the Lambda to invoke itself asynchronously (async self-invoke pattern:
        # Phase 1 returns 200 to API Gateway immediately; Phase 2 calls AgentCore).
        # Uses a constructed ARN string rather than self.handler_fn.function_arn to
        # avoid a CloudFormation circular dependency: Lambda → Role Policy → Lambda.Arn → Lambda.
        self.handler_fn.add_to_role_policy(iam.PolicyStatement(
            actions=["lambda:InvokeFunction"],
            resources=[f"arn:aws:lambda:{cdk.Aws.REGION}:{cdk.Aws.ACCOUNT_ID}:function:CarDesignWSHandler"],
        ))

        # WebSocket API
        self.ws_api = apigwv2.CfnApi(
            self, "WebSocketApi",
            name="CarDesignExplorerWS",
            protocol_type="WEBSOCKET",
            route_selection_expression="$request.body.action",
        )

        self.handler_fn.add_to_role_policy(iam.PolicyStatement(
            actions=["execute-api:ManageConnections"],
            resources=[
                f"arn:aws:execute-api:{cdk.Aws.REGION}:{cdk.Aws.ACCOUNT_ID}:"
                f"{self.ws_api.ref}/prod/POST/@connections/*"
            ],
        ))

        # Lambda integration
        integration = apigwv2.CfnIntegration(
            self, "WSIntegration",
            api_id=self.ws_api.ref,
            integration_type="AWS_PROXY",
            integration_uri=(
                f"arn:aws:apigateway:{cdk.Aws.REGION}:lambda:path"
                f"/2015-03-31/functions/{self.handler_fn.function_arn}/invocations"
            ),
        )

        # Routes
        for route_key in ["$connect", "$disconnect", "$default", "sendMessage"]:
            safe_name = route_key.replace("$", "Dollar")
            apigwv2.CfnRoute(
                self, f"Route{safe_name}",
                api_id=self.ws_api.ref,
                route_key=route_key,
                target=f"integrations/{integration.ref}",
            )

        # Stage
        apigwv2.CfnStage(
            self, "WSStage",
            api_id=self.ws_api.ref,
            stage_name="prod",
            auto_deploy=True,
        )

        # Grant API Gateway permission to invoke Lambda
        self.handler_fn.add_permission(
            "WSApiInvoke",
            principal=iam.ServicePrincipal("apigateway.amazonaws.com"),
            source_arn=f"arn:aws:execute-api:{cdk.Aws.REGION}:{cdk.Aws.ACCOUNT_ID}:{self.ws_api.ref}/*",
        )

        # Outputs
        ws_url = f"wss://{self.ws_api.ref}.execute-api.{cdk.Aws.REGION}.amazonaws.com/prod"
        cdk.CfnOutput(self, "WebSocketUrl", value=ws_url)
        cdk.CfnOutput(self, "LambdaFunctionName",
                       value=self.handler_fn.function_name)
