#!/usr/bin/env python3
"""CDK App entry point for Car Design Space Explorer."""

import os

import aws_cdk as cdk
from stacks.auth_stack import AuthStack
from stacks.mcp_auth_stack import McpAuthStack
from stacks.data_stack import DataStack
from stacks.storage_stack import StorageStack
from stacks.network_stack import NetworkStack
from stacks.api_stack import ApiStack
from stacks.agent_stack import AgentStack
from stacks.seed_stack import SeedStack
from stacks.dynamo_seed_stack import DynamoSeedStack
from stacks.frontend_deploy_stack import FrontendDeployStack

app = cdk.App()

env = cdk.Environment(
    account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
    region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1"),
)

auth = AuthStack(app, "CarDesignAuth", env=env)
mcp_auth = McpAuthStack(app, "CarDesignMcpAuth", env=env)
data = DataStack(app, "CarDesignData", env=env)
storage = StorageStack(app, "CarDesignStorage", env=env)
network = NetworkStack(app, "CarDesignNetwork", env=env)
api = ApiStack(app, "CarDesignApi", env=env,
               user_pool=auth.user_pool,
               vpc=network.vpc,
               lambda_security_group=network.lambda_security_group,
               dynamo_key=data.table_key)
api.add_dependency(network)
agents = AgentStack(app, "CarDesignAgents", env=env,
                    model_bucket=storage.model_bucket,
                    mcp_secret=mcp_auth.mcp_secret,
                    mcp_secret_key=mcp_auth.secret_key,
                    variant_cache_table=data.variant_cache_table,
                    dynamo_key=data.table_key)
agents.add_dependency(mcp_auth)
agents.add_dependency(data)
seed = SeedStack(app, "CarDesignSeed", env=env,
                 model_bucket=storage.model_bucket)
seed.add_dependency(storage)

# DynamoDB seeding — depends on Data (tables) and Seed (S3 KPI CSV)
dynamo_seed = DynamoSeedStack(
    app, "CarDesignDynamoSeed", env=env,
    cost_params_table=data.cost_params_table,
    external_cost_table=data.external_cost_table,
    variant_cache_table=data.variant_cache_table,
    model_bucket=storage.model_bucket,
)
dynamo_seed.add_dependency(data)
dynamo_seed.add_dependency(seed)

# Frontend deploy — depends on Auth (Cognito IDs) and Api (WebSocket URL)
frontend = FrontendDeployStack(
    app, "CarDesignFrontend", env=env,
    frontend_bucket=storage.frontend_bucket,
    distribution=storage.distribution,
    user_pool=auth.user_pool,
    user_pool_client=auth.app_client,
    ws_api_id=api.ws_api.ref,
)
frontend.add_dependency(auth)
frontend.add_dependency(api)
frontend.add_dependency(storage)

app.synth()
