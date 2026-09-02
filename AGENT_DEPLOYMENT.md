# Car Design Explorer — Agent Deployment Guide

## Architecture

5 Strands agents deployed to Bedrock AgentCore Runtime using A2A protocol:

| Agent | Role | Special Dependencies |
|-------|------|---------------------|
| Orchestrator | Central coordinator, routes to specialists | AgentCore Memory, A2AClientToolProvider |
| Aero | Aerodynamic KPI prediction (Cd, Cs, Cl, Cmy) | MLSimKit, PyTorch, DynamoDB cache |
| Structural | Structural feasibility evaluation | Pure computation |
| Cost | Manufacturing cost estimation | MCP Gateway (Cognito JWT) |
| Geometry | 3D mesh modification + Stable Diffusion preview | trimesh, matplotlib, Bedrock Runtime |

## Prerequisites

1. CDK stacks deployed (`cdk deploy --all` from `infra/cdk/`)
2. AWS CLI configured for `us-east-1`, with Bedrock access to Stable Diffusion 3.5 Large in `us-west-2`
3. Python 3.10+ with required packages:
   ```bash
   pip install boto3 botocore bedrock-agentcore bedrock-agentcore-starter-toolkit \
     strands-agents strands-agents-tools
   ```

## Deployment (Single Command)

From the Jupyter terminal (or any machine with AWS credentials):

```bash
cd car-design-explorer
export AGENT_MODEL_ID="us.anthropic.claude-haiku-4-5-20251001-v1:0"
export IMAGE_MODEL_ID="stability.sd3-5-large-v1:0"
export IMAGE_MODEL_REGION="us-west-2"
python3 deploy_agents.py --deploy-all \
  --cognito-user-pool-id <USER_POOL_ID> \
  --cognito-client-id <CLIENT_ID> \
  --mcp-user-pool-id <MCP_USER_POOL_ID> \
  --region us-east-1
```

This uses **Direct Code Deploy** (same pattern as SPA) — no Docker required.
AgentCore handles container build server-side via CodeBuild.

The script:
1. Creates MCP Gateway for Cost Agent (Cognito JWT auth)
2. Deploys all 5 agents using `Runtime.configure() + launch()`:
   - Resolves five least-privilege runtime role ARNs from the `CarDesignAgents` CDK stack
   - Leaves ECR/CodeBuild/AgentCore control-plane permissions on the invoking deployment identity, never on runtime roles
   - Creates AgentCore Memory for Orchestrator
   - For each agent: stages code → configures Runtime (A2A protocol) → launches
   - Waits for READY status
3. Wires orchestrator ARN to Lambda WebSocket handler
4. Verifies all agents are deployed

## Deploy Individual Agent

```bash
python3 deploy_agents.py --agent orchestrator \
  --model-id us.anthropic.claude-haiku-4-5-20251001-v1:0 \
  --cognito-user-pool-id <USER_POOL_ID> \
  --cognito-client-id <CLIENT_ID> \
  --agent-m2m-client-id <AGENT_M2M_CLIENT_ID> \
  --mcp-user-pool-id <MCP_USER_POOL_ID> \
  --region <REGION>
```

## List Deployed Agents

```bash
python3 deploy_agents.py --list --region <REGION>
```

## Wire Lambda Manually

```bash
python3 deploy_agents.py --wire-lambda \
  --orchestrator-arn <ORCHESTRATOR_ARN> \
  --region <REGION>
```

## How It Works

Uses the AgentCore Starter Toolkit (`Runtime.configure() + launch()`) for
direct code deployment — same approach as the SPA agent. No Docker needed.

Each agent:
- Runs as a FastAPI app with A2AServer mounted at root (`serve_at_root=True`)
- Listens on port 9000 (AgentCore A2A requirement)
- Uses Cognito JWT for inbound authentication
- Is packaged and deployed via CodeBuild (ARM64 container built server-side)

The orchestrator discovers other agents via `list_agentcore_runtimes()` and
communicates using `A2AClientToolProvider` from `strands-agents-tools`.

## Agent Communication Flow

```
Frontend → WebSocket API → Lambda → AgentCore Runtime (Orchestrator)
                                         ↓ A2A protocol
                              ┌──────────┼──────────┐
                              ↓          ↓          ↓
                          Aero Agent  Structural  Cost Agent
                                      Agent       ↓ MCP Gateway
                                               MCP Servers
                              ↓
                          Geometry Agent
                          (Stable Diffusion + trimesh)
```

## CDK Outputs Reference

After deploying CDK stacks, note these outputs (used in agent deployment):

- `UserPoolId` — Cognito User Pool ID for frontend auth
- `UserPoolClientId` — Cognito App Client ID
- `WebSocketUrl` — WebSocket API Gateway endpoint
- `CloudFrontUrl` — Frontend distribution URL
- `MCPUserPoolId` — MCP Gateway Cognito User Pool ID

## Troubleshooting

- **Agent not READY**: Run `python3 deploy_agents.py --list --region <REGION>`
- **Memory creation fails**: The orchestrator creates memory at startup as fallback
- **MCP Gateway issues**: Verify MCP gateway credentials in Secrets Manager
- **Lambda not updated**: Use `--wire-lambda` with the orchestrator ARN
- **CodeBuild fails**: Check CloudWatch logs for the CodeBuild project
- **boto3 too old**: Run `pip install --upgrade boto3 botocore`
