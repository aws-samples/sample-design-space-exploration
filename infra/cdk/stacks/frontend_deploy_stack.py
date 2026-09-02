"""Frontend Deploy Stack — Build React app and deploy to S3 + CloudFront.

Injects Cognito and WebSocket configuration at build time via environment
variables, then deploys the built assets to the frontend S3 bucket with
CloudFront cache invalidation.

This makes the frontend deployment fully automated — no manual npm build
or S3 upload needed.
"""

import os

import aws_cdk as cdk
from aws_cdk import (
    aws_cloudfront as cloudfront,
    aws_cognito as cognito,
    aws_s3 as s3,
    aws_s3_deployment as s3deploy,
)
from constructs import Construct

CDK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CDK_DIR))


class FrontendDeployStack(cdk.Stack):
    def __init__(
        self,
        scope: Construct,
        id: str,
        frontend_bucket: s3.IBucket,
        distribution: cloudfront.IDistribution,
        user_pool: cognito.IUserPool,
        user_pool_client: cognito.IUserPoolClient,
        ws_api_id: str,
        **kwargs,
    ):
        super().__init__(scope, id, **kwargs)

        frontend_dir = os.path.join(PROJECT_ROOT, "frontend")
        dist_dir = os.path.join(frontend_dir, "build")

        # Construct WebSocket URL from API Gateway ID (deploy-time token)
        websocket_url = cdk.Fn.join("", [
            "wss://", ws_api_id,
            ".execute-api.", cdk.Aws.REGION, ".amazonaws.com/prod",
        ])

        # Runtime config JSON — Cognito + WebSocket values injected by CDK.
        # The frontend fetches /config.json at startup so it doesn't need
        # to be rebuilt when these values change.
        runtime_config_source = s3deploy.Source.json_data(
            "config.json",
            {
                "cognito": {
                    "userPoolId": user_pool.user_pool_id,
                    "userPoolClientId": user_pool_client.user_pool_client_id,
                    "region": cdk.Aws.REGION,
                },
                "websocket": {
                    "url": websocket_url,
                },
            },
        )

        if os.path.isdir(dist_dir):
            # Pre-built frontend exists — deploy it together with config.json
            # in a single BucketDeployment so prune doesn't delete config.json
            s3deploy.BucketDeployment(
                self,
                "FrontendAssets",
                sources=[
                    s3deploy.Source.asset(dist_dir),
                    runtime_config_source,
                ],
                destination_bucket=frontend_bucket,
                distribution=distribution,
                distribution_paths=["/*"],
                memory_limit=512,
                prune=True,
            )
        else:
            # No pre-built frontend — deploy placeholder + config.json together
            s3deploy.BucketDeployment(
                self,
                "FrontendPlaceholder",
                sources=[
                    s3deploy.Source.data(
                        "index.html",
                        _placeholder_html(),
                    ),
                    runtime_config_source,
                ],
                destination_bucket=frontend_bucket,
                distribution=distribution,
                distribution_paths=["/*"],
                prune=True,
            )

        cdk.CfnOutput(
            self,
            "FrontendUrl",
            value=f"https://{distribution.distribution_domain_name}",
        )
        cdk.CfnOutput(
            self,
            "RuntimeConfigUrl",
            value=f"https://{distribution.distribution_domain_name}/config.json",
        )


def _placeholder_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Car Design Space Explorer</title>
  <style>
    body { font-family: system-ui, sans-serif; display: flex; justify-content: center;
           align-items: center; min-height: 100vh; margin: 0; background: #0f172a; color: #e2e8f0; }
    .container { text-align: center; max-width: 600px; padding: 2rem; }
    h1 { color: #38bdf8; margin-bottom: 0.5rem; }
    p { color: #94a3b8; line-height: 1.6; }
    code { background: #1e293b; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.9rem; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Car Design Space Explorer</h1>
    <p>Infrastructure deployed successfully.</p>
    <p>To deploy the full frontend, run:</p>
    <p><code>cd frontend && npm install && npm run build</code></p>
    <p>Then redeploy this stack or upload the <code>dist/</code> folder to S3.</p>
  </div>
</body>
</html>"""
