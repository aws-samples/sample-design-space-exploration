"""Seed Stack — Upload ML model weights, KPI CSV, and sample STL geometries to S3.

Uploads model weights, the KPI predictions CSV, and 10 sample WindsorML STL
files to the models S3 bucket as part of CDK deployment.

Assets uploaded:
- kpi/best_model.pt                      — KPI surrogate model (Cd, Cs, Cl, Cmy)
- surface/best_model.pt                  — Surface variable model (cpavg, cfxavg)
- slices/ae_best_model.pt                — Slices autoencoder model
- slices/mgn_last_model.pt               — Slices MeshGraphNet model
- predictions/kpi/variant_kpis_all4.csv  — Pre-computed KPIs for 355 variants (→ DynamoDB)
- geometries/run_0.stl … run_9.stl       — 10 sample STL geometries

Pre-computed VTP/PNG surface/slice predictions are NOT stored in the repo.
Surface and slice visualisations are generated via live inference at query time.
"""

import os

import aws_cdk as cdk
from aws_cdk import aws_s3 as s3, aws_s3_deployment as s3deploy
from constructs import Construct

# Resolve paths relative to the CDK app root (infra/cdk/)
CDK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CDK_DIR))


class SeedStack(cdk.Stack):
    def __init__(
        self,
        scope: Construct,
        id: str,
        model_bucket: s3.IBucket,
        **kwargs,
    ):
        super().__init__(scope, id, **kwargs)

        weights_dir = os.path.join(PROJECT_ROOT, "backend", "models", "weights")
        kpi_csv_dir = os.path.join(PROJECT_ROOT, "predictions", "kpi")
        stl_samples_dir = os.path.join(PROJECT_ROOT, "geometries")

        # --- ML Model Weights ---
        # Upload kpi/best_model.pt, surface/best_model.pt,
        # slices/ae_best_model.pt, slices/mgn_last_model.pt
        if os.path.isdir(weights_dir):
            s3deploy.BucketDeployment(
                self, "ModelWeights",
                sources=[s3deploy.Source.asset(weights_dir)],
                destination_bucket=model_bucket,
                memory_limit=1024,
                ephemeral_storage_size=cdk.Size.mebibytes(1024),
                prune=False,
            )

        # --- KPI Predictions CSV (used by DynamoSeedStack to populate CarDesignVariantCache) ---
        # Contains pre-computed Cd/Cs/Cl/Cmy for all 355 WindsorML variants.
        # Uploaded to s3://…/predictions/kpi/ so the seeder Lambda can find it.
        if os.path.isdir(kpi_csv_dir):
            s3deploy.BucketDeployment(
                self, "KpiCsv",
                sources=[s3deploy.Source.asset(kpi_csv_dir)],
                destination_bucket=model_bucket,
                destination_key_prefix="predictions/kpi",
                memory_limit=512,
                ephemeral_storage_size=cdk.Size.mebibytes(512),
                prune=False,
            )

        # --- Sample STL Geometries (10 WindsorML variants) ---
        # These are the variants supported out-of-the-box for surface/slices
        # inference. Any other variant requires the user to upload an STL.
        if os.path.isdir(stl_samples_dir):
            s3deploy.BucketDeployment(
                self, "SampleGeometries",
                sources=[s3deploy.Source.asset(stl_samples_dir)],
                destination_bucket=model_bucket,
                destination_key_prefix="geometries",
                memory_limit=512,
                ephemeral_storage_size=cdk.Size.mebibytes(512),
                prune=False,
            )

        # --- Outputs ---
        cdk.CfnOutput(
            self, "SeedStatus",
            value="Model weights and sample geometries uploaded to S3",
        )
