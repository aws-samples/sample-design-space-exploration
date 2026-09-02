"""DynamoDB Seed Stack — Populate DynamoDB tables with seed data via Custom Resource.

Uses a Lambda-backed Custom Resource to seed all three DynamoDB tables:
  1. CarDesignCostParameters (~50 rows of internal cost parameters)
  2. CarDesignExternalCostData (~55 rows of market/supplier/historical data)
  3. CarDesignVariantCache (pre-computed KPI + cost data from S3 CSV)

The Lambda reads the KPI predictions CSV from S3 to populate the variant cache,
and has all cost parameter data embedded inline (no external dependencies).

This runs automatically during `cdk deploy` — no manual seeding needed.
"""

import json
import os
import textwrap

import aws_cdk as cdk
from aws_cdk import (
    aws_dynamodb as dynamodb,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_s3 as s3,
    custom_resources as cr,
)
from constructs import Construct


class DynamoSeedStack(cdk.Stack):
    def __init__(
        self,
        scope: Construct,
        id: str,
        cost_params_table: dynamodb.ITable,
        external_cost_table: dynamodb.ITable,
        variant_cache_table: dynamodb.ITable,
        model_bucket: s3.IBucket,
        **kwargs,
    ):
        super().__init__(scope, id, **kwargs)

        # Lambda function that seeds all three tables
        seeder_fn = lambda_.Function(
            self,
            "DynamoSeederFn",
            function_name="CarDesignDynamoSeeder",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="index.handler",
            code=lambda_.Code.from_asset(
                os.path.join(os.path.dirname(__file__), "..", "lambda", "dynamo_seeder")
            ),
            timeout=cdk.Duration.minutes(5),
            memory_size=512,
            environment={
                "COST_PARAMS_TABLE": cost_params_table.table_name,
                "EXTERNAL_COST_TABLE": external_cost_table.table_name,
                "VARIANT_CACHE_TABLE": variant_cache_table.table_name,
                "MODEL_BUCKET": model_bucket.bucket_name,
                "KPI_CSV_KEY": "predictions/kpi/variant_kpis_all4.csv",
            },
        )

        # Grant permissions
        cost_params_table.grant_write_data(seeder_fn)
        external_cost_table.grant_write_data(seeder_fn)
        variant_cache_table.grant_write_data(seeder_fn)
        model_bucket.grant_read(seeder_fn)

        # Custom Resource that triggers the Lambda on deploy
        provider = cr.Provider(
            self,
            "SeederProvider",
            on_event_handler=seeder_fn,
        )

        cdk.CustomResource(
            self,
            "DynamoSeedTrigger",
            service_token=provider.service_token,
            properties={
                # Change this value to force re-seeding on next deploy
                "SeedVersion": "1.0.3",
            },
        )

        cdk.CfnOutput(self, "SeedStatus", value="DynamoDB tables seeded automatically")
