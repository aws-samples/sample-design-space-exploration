"""Data Stack — DynamoDB tables for cost parameters and variant cache.

Tables are encrypted with a customer-managed KMS key (rotation enabled),
mirroring the ``DynamoDBKMSKey`` in the reference ``infrastructure.yaml``.

The key policy grants the account root full control (delegates to IAM) AND
allows any same-account principal that accesses the key *through DynamoDB*
(``kms:ViaService``). The ViaService grant means the externally-managed
AgentCore agent roles (e.g. the aero agent reading CarDesignVariantCache) and
the MCP Lambda continue to work without needing explicit KMS permissions on
their IAM policies.
"""

import aws_cdk as cdk
from aws_cdk import (
    aws_dynamodb as dynamodb,
    aws_iam as iam,
    aws_kms as kms,
)
from constructs import Construct


class DataStack(cdk.Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # --- Customer-managed KMS key for DynamoDB encryption ---
        self.table_key = kms.Key(
            self, "DynamoDBKMSKey",
            description="KMS key for Car Design DynamoDB table encryption",
            enable_key_rotation=True,
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )
        # Allow any same-account principal to use the key *via DynamoDB*.
        # This keeps externally-managed agent/MCP roles working without
        # modifying their IAM policies.
        self.table_key.add_to_resource_policy(iam.PolicyStatement(
            sid="AllowAccessViaDynamoDB",
            effect=iam.Effect.ALLOW,
            principals=[iam.AccountRootPrincipal()],
            actions=[
                "kms:Encrypt", "kms:Decrypt", "kms:ReEncrypt*",
                "kms:GenerateDataKey*", "kms:DescribeKey", "kms:CreateGrant",
            ],
            resources=["*"],
            conditions={
                "StringEquals": {
                    "kms:ViaService": f"dynamodb.{cdk.Aws.REGION}.amazonaws.com",
                    "kms:CallerAccount": cdk.Aws.ACCOUNT_ID,
                },
            },
        ))

        _enc = dict(
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=cdk.RemovalPolicy.DESTROY,
            encryption=dynamodb.TableEncryption.CUSTOMER_MANAGED,
            encryption_key=self.table_key,
            point_in_time_recovery=True,
        )

        # Cost Parameters Table
        self.cost_params_table = dynamodb.Table(
            self, "CostParametersTable",
            table_name="CarDesignCostParameters",
            partition_key=dynamodb.Attribute(name="pk", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="sk", type=dynamodb.AttributeType.STRING),
            **_enc,
        )

        # External Cost Data Table (for MCP server)
        self.external_cost_table = dynamodb.Table(
            self, "ExternalCostDataTable",
            table_name="CarDesignExternalCostData",
            partition_key=dynamodb.Attribute(name="pk", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="sk", type=dynamodb.AttributeType.STRING),
            **_enc,
        )

        # Variant Cache Table with GSI on cd
        self.variant_cache_table = dynamodb.Table(
            self, "VariantCacheTable",
            table_name="CarDesignVariantCache",
            partition_key=dynamodb.Attribute(name="pk", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="sk", type=dynamodb.AttributeType.STRING),
            **_enc,
        )

        self.variant_cache_table.add_global_secondary_index(
            index_name="cd-index",
            partition_key=dynamodb.Attribute(name="pk", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="cd", type=dynamodb.AttributeType.NUMBER),
            projection_type=dynamodb.ProjectionType.ALL,
        )

        # WebSocket Connections Table
        self.connections_table = dynamodb.Table(
            self, "WSConnectionsTable",
            table_name="CarDesignWSConnections",
            partition_key=dynamodb.Attribute(name="connectionId", type=dynamodb.AttributeType.STRING),
            time_to_live_attribute="ttl",
            **_enc,
        )

        # Outputs
        cdk.CfnOutput(self, "DynamoDBKmsKeyArn", value=self.table_key.key_arn)
        cdk.CfnOutput(self, "CostParamsTableName", value=self.cost_params_table.table_name)
        cdk.CfnOutput(self, "VariantCacheTableName", value=self.variant_cache_table.table_name)
        cdk.CfnOutput(self, "ConnectionsTableName", value=self.connections_table.table_name)
