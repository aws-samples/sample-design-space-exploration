"""Storage Stack — S3 buckets, CloudFront distribution for frontend hosting.

S3 hardening mirrors the reference standard in the parent repo's
``infrastructure.yaml``:
  * A dedicated server-access-logging bucket (cannot log to itself).
  * Every data bucket: block-all-public-access, versioning, SSE-S3 (AES256)
    encryption, enforced TLS (deny insecure transport), and server access
    logging delivered to the logging bucket.

Note: the standard uses SSE-S3 (AES256) for S3 rather than a customer-managed
KMS key. This keeps object reads transparent for the AgentCore agent roles
(geometry / aero) that access these buckets directly, while still encrypting
all objects at rest.
"""

import aws_cdk as cdk
from aws_cdk import (
    aws_s3 as s3,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
)
from constructs import Construct


class StorageStack(cdk.Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # ------------------------------------------------------------------
        # Server access logging bucket (a logging bucket cannot log to itself)
        # ------------------------------------------------------------------
        self.logging_bucket = s3.Bucket(
            self, "AccessLogsBucket",
            bucket_name=f"car-design-explorer-logs-{cdk.Aws.ACCOUNT_ID}",
            removal_policy=cdk.RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=True,
            # ACLs enabled (BucketOwnerPreferred) so the S3 log delivery group
            # can write access logs into this bucket.
            object_ownership=s3.ObjectOwnership.BUCKET_OWNER_PREFERRED,
        )

        _secure = dict(
            removal_policy=cdk.RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=True,
        )

        # Model artifacts bucket (ML models, predictions, geometries)
        self.model_bucket = s3.Bucket(
            self, "ModelBucket",
            bucket_name=f"car-design-explorer-models-{cdk.Aws.ACCOUNT_ID}",
            cors=[s3.CorsRule(
                allowed_methods=[s3.HttpMethods.GET],
                allowed_origins=["*"],
                allowed_headers=["*"],
            )],
            server_access_logs_bucket=self.logging_bucket,
            server_access_logs_prefix="model-bucket/",
            **_secure,
        )

        # Geometry files bucket
        self.geometry_bucket = s3.Bucket(
            self, "GeometryBucket",
            bucket_name=f"car-design-explorer-geometries-{cdk.Aws.ACCOUNT_ID}",
            cors=[s3.CorsRule(
                allowed_methods=[s3.HttpMethods.GET],
                allowed_origins=["*"],
                allowed_headers=["*"],
            )],
            server_access_logs_bucket=self.logging_bucket,
            server_access_logs_prefix="geometry-bucket/",
            **_secure,
        )

        # Frontend hosting bucket
        self.frontend_bucket = s3.Bucket(
            self, "FrontendBucket",
            bucket_name=f"car-design-explorer-frontend-{cdk.Aws.ACCOUNT_ID}",
            server_access_logs_bucket=self.logging_bucket,
            server_access_logs_prefix="frontend-bucket/",
            **_secure,
        )

        # CloudFront distribution (same stack as bucket to avoid cyclic deps)
        self.distribution = cloudfront.Distribution(
            self, "FrontendDistribution",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3BucketOrigin.with_origin_access_control(self.frontend_bucket),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
            ),
            default_root_object="index.html",
            error_responses=[
                cloudfront.ErrorResponse(
                    http_status=404,
                    response_http_status=200,
                    response_page_path="/index.html",
                ),
            ],
        )

        # Outputs
        cdk.CfnOutput(self, "ModelBucketName", value=self.model_bucket.bucket_name)
        cdk.CfnOutput(self, "GeometryBucketName", value=self.geometry_bucket.bucket_name)
        cdk.CfnOutput(self, "FrontendBucketName", value=self.frontend_bucket.bucket_name)
        cdk.CfnOutput(self, "AccessLogsBucketName", value=self.logging_bucket.bucket_name)
        cdk.CfnOutput(self, "CloudFrontUrl",
                       value=f"https://{self.distribution.distribution_domain_name}")
        cdk.CfnOutput(self, "CloudFrontDistributionId",
                       value=self.distribution.distribution_id)
