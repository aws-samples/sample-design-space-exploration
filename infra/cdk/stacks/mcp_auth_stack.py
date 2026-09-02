"""MCP Auth Stack — Dedicated Cognito User Pool for MCP Gateway JWT auth.

Provides machine-to-machine (M2M) OAuth2 client credentials flow for the
Cost Agent to authenticate with MCP servers behind AgentCore Gateway.

Resources created:
- Cognito User Pool (no human sign-up, M2M only)
- Cognito Domain (for token endpoint)
- Resource Server with custom scopes (mcp-api/read, mcp-api/write)
- App Client configured for client_credentials grant (with secret)
- Secrets Manager secret storing client_id, client_secret, token_url, scope
"""

import aws_cdk as cdk
from aws_cdk import (
    aws_cognito as cognito,
    aws_secretsmanager as secretsmanager,
    aws_kms as kms,
    custom_resources as cr,
    aws_iam as iam,
)
from constructs import Construct


class McpAuthStack(cdk.Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # --- Cognito User Pool (M2M only, no human users) ---
        self.user_pool = cognito.UserPool(
            self, "McpGatewayUserPool",
            user_pool_name="CarDesignMcpGateway",
            self_sign_up_enabled=False,
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )

        # --- Cognito Domain (required for OAuth2 token endpoint) ---
        domain_prefix = f"car-design-mcp-{cdk.Aws.ACCOUNT_ID}"
        self.domain = self.user_pool.add_domain(
            "McpDomain",
            cognito_domain=cognito.CognitoDomainOptions(
                domain_prefix=domain_prefix,
            ),
        )

        # --- Resource Server with custom scopes ---
        read_scope = cognito.ResourceServerScope(
            scope_name="read", scope_description="Read MCP cost data"
        )
        write_scope = cognito.ResourceServerScope(
            scope_name="write", scope_description="Write MCP cost data"
        )

        self.resource_server = self.user_pool.add_resource_server(
            "McpResourceServer",
            identifier="mcp-api",
            scopes=[read_scope, write_scope],
        )

        # --- App Client (L2) for client_credentials grant ---
        self.app_client = self.user_pool.add_client(
            "McpGatewayClient",
            user_pool_client_name="car-design-mcp-gateway-client",
            generate_secret=True,
            auth_flows=cognito.AuthFlow(
                custom=False,
                user_password=False,
                user_srp=False,
            ),
            o_auth=cognito.OAuthSettings(
                flows=cognito.OAuthFlows(
                    client_credentials=True,
                ),
                scopes=[
                    cognito.OAuthScope.resource_server(self.resource_server, read_scope),
                    cognito.OAuthScope.resource_server(self.resource_server, write_scope),
                ],
            ),
        )

        # --- Fetch client secret via AwsSdkCall Custom Resource ---
        # Cognito L2 doesn't expose client_secret as a token, so we use
        # AwsCustomResource to call DescribeUserPoolClient and extract it.
        describe_client = cr.AwsCustomResource(
            self, "DescribeUserPoolClient",
            on_create=cr.AwsSdkCall(
                service="CognitoIdentityServiceProvider",
                action="describeUserPoolClient",
                parameters={
                    "UserPoolId": self.user_pool.user_pool_id,
                    "ClientId": self.app_client.user_pool_client_id,
                },
                physical_resource_id=cr.PhysicalResourceId.of("McpClientSecret"),
            ),
            on_update=cr.AwsSdkCall(
                service="CognitoIdentityServiceProvider",
                action="describeUserPoolClient",
                parameters={
                    "UserPoolId": self.user_pool.user_pool_id,
                    "ClientId": self.app_client.user_pool_client_id,
                },
                physical_resource_id=cr.PhysicalResourceId.of("McpClientSecret"),
            ),
            policy=cr.AwsCustomResourcePolicy.from_statements([
                iam.PolicyStatement(
                    actions=["cognito-idp:DescribeUserPoolClient"],
                    resources=[self.user_pool.user_pool_arn],
                ),
            ]),
        )
        describe_client.node.add_dependency(self.app_client)

        client_secret = describe_client.get_response_field(
            "UserPoolClient.ClientSecret"
        )

        # Token URL
        token_url = (
            f"https://{domain_prefix}.auth.{cdk.Aws.REGION}.amazoncognito.com/oauth2/token"
        )

        # --- Customer-managed KMS key for Secrets Manager encryption ---
        # Mirrors the SecretsKMSKey in the reference infrastructure.yaml.
        # The ViaService grant lets the externally-managed Cost agent role read
        # the secret (Secrets Manager decrypts on its behalf) without needing
        # explicit KMS permissions on its IAM policy.
        self.secret_key = kms.Key(
            self, "SecretsKMSKey",
            description="KMS key for Car Design Secrets Manager encryption",
            enable_key_rotation=True,
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )
        self.secret_key.add_to_resource_policy(iam.PolicyStatement(
            sid="AllowAccessViaSecretsManager",
            effect=iam.Effect.ALLOW,
            principals=[iam.AccountRootPrincipal()],
            actions=[
                "kms:Encrypt", "kms:Decrypt", "kms:ReEncrypt*",
                "kms:GenerateDataKey*", "kms:DescribeKey", "kms:CreateGrant",
            ],
            resources=["*"],
            conditions={
                "StringEquals": {
                    "kms:ViaService": f"secretsmanager.{cdk.Aws.REGION}.amazonaws.com",
                    "kms:CallerAccount": cdk.Aws.ACCOUNT_ID,
                },
            },
        ))

        # --- Secrets Manager: store full credentials for Cost Agent ---
        self.mcp_secret = secretsmanager.Secret(
            self, "McpGatewayCredentials",
            # This is a Secrets Manager resource name, never a credential value.
            secret_name="car-design/mcp-gateway-credentials",  # nosec B106
            description="OAuth2 client credentials for MCP Gateway JWT auth",
            encryption_key=self.secret_key,
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )

        # Populate the secret with actual credentials using AwsSdkCall
        # We write the secret value after creation so we can include the
        # real Cognito client_secret fetched by the Custom Resource above.
        cr.AwsCustomResource(
            self, "PopulateMcpSecret",
            on_create=cr.AwsSdkCall(
                service="SecretsManager",
                action="putSecretValue",
                parameters={
                    "SecretId": self.mcp_secret.secret_name,
                    "SecretString": cdk.Fn.join("", [
                        '{"client_id":"', self.app_client.user_pool_client_id,
                        '","client_secret":"', client_secret,
                        '","token_url":"', token_url,
                        '","scope":"mcp-api/read mcp-api/write',
                        '","user_pool_id":"', self.user_pool.user_pool_id,
                        '"}',
                    ]),
                },
                physical_resource_id=cr.PhysicalResourceId.of("PopulateMcpSecret"),
            ),
            on_update=cr.AwsSdkCall(
                service="SecretsManager",
                action="putSecretValue",
                parameters={
                    "SecretId": self.mcp_secret.secret_name,
                    "SecretString": cdk.Fn.join("", [
                        '{"client_id":"', self.app_client.user_pool_client_id,
                        '","client_secret":"', client_secret,
                        '","token_url":"', token_url,
                        '","scope":"mcp-api/read mcp-api/write',
                        '","user_pool_id":"', self.user_pool.user_pool_id,
                        '"}',
                    ]),
                },
                physical_resource_id=cr.PhysicalResourceId.of("PopulateMcpSecret"),
            ),
            policy=cr.AwsCustomResourcePolicy.from_statements([
                iam.PolicyStatement(
                    actions=["secretsmanager:PutSecretValue"],
                    resources=[self.mcp_secret.secret_arn],
                ),
                # Secrets Manager uses this customer-managed key on behalf of
                # the custom-resource role when writing the secret value.
                iam.PolicyStatement(
                    actions=[
                        "kms:Encrypt",
                        "kms:Decrypt",
                        "kms:ReEncrypt*",
                        "kms:GenerateDataKey*",
                        "kms:DescribeKey",
                    ],
                    resources=[self.secret_key.key_arn],
                ),
            ]),
        )

        # --- Outputs ---
        cdk.CfnOutput(self, "McpUserPoolId", value=self.user_pool.user_pool_id)
        cdk.CfnOutput(self, "McpUserPoolArn", value=self.user_pool.user_pool_arn)
        cdk.CfnOutput(self, "McpUserPoolClientId",
                       value=self.app_client.user_pool_client_id)
        cdk.CfnOutput(self, "McpTokenUrl", value=token_url)
        cdk.CfnOutput(self, "McpDomainPrefix", value=domain_prefix)
        cdk.CfnOutput(self, "McpSecretName",
                       value=self.mcp_secret.secret_name)
        cdk.CfnOutput(self, "McpSecretArn", value=self.mcp_secret.secret_arn)
        cdk.CfnOutput(self, "McpResourceServerId", value="mcp-api")
        cdk.CfnOutput(self, "McpScopes", value="mcp-api/read mcp-api/write")
