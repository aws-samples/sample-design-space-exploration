#!/usr/bin/env bash
# =============================================================================
# Car Design Space Explorer — Full Deployment Script
# Run from the car-design-explorer/ directory.
#
# Usage:
#   ./deploy.sh                  # Full deployment
#   ./deploy.sh --region us-west-2
#   ./deploy.sh --skip-infra     # Skip CDK (re-deploy agents + frontend only)
#   ./deploy.sh --skip-agents    # Skip agent deployment (infra + frontend only)
#   ./deploy.sh --frontend-only  # Build and sync frontend only
# =============================================================================
set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}  $*"; }
info() { echo -e "${CYAN}[--]${NC}  $*"; }
warn() { echo -e "${YELLOW}[!!]${NC}  $*"; }
die()  { echo -e "${RED}[ERR]${NC} $*" >&2; exit 1; }
header() { echo -e "\n${BOLD}${CYAN}═══ $* ═══${NC}"; }

# ── Flags ─────────────────────────────────────────────────────────────────────
SKIP_INFRA=false
SKIP_AGENTS=false
FRONTEND_ONLY=false
REGION=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-infra)    SKIP_INFRA=true;    shift ;;
    --skip-agents)   SKIP_AGENTS=true;   shift ;;
    --frontend-only) FRONTEND_ONLY=true; shift ;;
    --region)        REGION="$2";        shift 2 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

# Frontend credentials are mandatory for every deployment mode. The password is
# consumed from the environment and is never printed or passed as a CLI argument.
[[ -n "${FRONTEND_USERNAME_EMAIL:-}" ]] \
  || die "Set mandatory environment variable FRONTEND_USERNAME_EMAIL."
[[ -n "${FRONTEND_USERNAME_PASSWORD:-}" ]] \
  || die "Set mandatory environment variable FRONTEND_USERNAME_PASSWORD."

# One primary reasoning model is used by all five agents. Require it whenever
# agents are deployed so CDK IAM grants and AgentCore runtime configuration use
# exactly the same model ID.
if [[ "$SKIP_AGENTS" == "false" && "$FRONTEND_ONLY" == "false" ]]; then
  [[ -n "${AGENT_MODEL_ID:-}" ]] \
    || die "Set mandatory environment variable AGENT_MODEL_ID (for example: us.anthropic.claude-haiku-4-5-20251001-v1:0)."
  if ! [[ "$AGENT_MODEL_ID" =~ ^us\.[A-Za-z0-9._:-]+$ ]]; then
    die "AGENT_MODEL_ID must be a US geographic inference profile ID beginning with 'us.'."
  fi
  export AGENT_MODEL_ID
fi

if ! [[ "$FRONTEND_USERNAME_EMAIL" =~ ^[^[:space:]@]+@[^[:space:]@]+\.[^[:space:]@]+$ ]]; then
  die "FRONTEND_USERNAME_EMAIL must be a valid email address."
fi
if [[ ${#FRONTEND_USERNAME_PASSWORD} -lt 8 ]] \
  || ! [[ "$FRONTEND_USERNAME_PASSWORD" =~ [[:lower:]] ]] \
  || ! [[ "$FRONTEND_USERNAME_PASSWORD" =~ [[:upper:]] ]] \
  || ! [[ "$FRONTEND_USERNAME_PASSWORD" =~ [[:digit:]] ]]; then
  die "FRONTEND_USERNAME_PASSWORD must be at least 8 characters and include lowercase, uppercase, and numeric characters."
fi

# ── Resolve region ────────────────────────────────────────────────────────────
if [[ -z "$REGION" ]]; then
  REGION=$(aws configure get region 2>/dev/null || true)
  REGION=${REGION:-us-east-1}
fi
export CDK_DEFAULT_REGION="$REGION"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export CDK_DEFAULT_ACCOUNT="$ACCOUNT_ID"

echo -e "\n${BOLD}Car Design Space Explorer — Deployment${NC}"
echo "  Account : $ACCOUNT_ID"
echo "  Region  : $REGION"
if [[ "$SKIP_AGENTS" == "false" && "$FRONTEND_ONLY" == "false" ]]; then
  echo "  Agent model: $AGENT_MODEL_ID"
fi
echo ""

# Verify we are in the right directory
[[ -f "deploy_agents.py" && -d "frontend" && -d "infra" ]] \
  || die "Run this script from the car-design-explorer/ directory."

# ── Helper: read a single CloudFormation output ───────────────────────────────
cfn_output() {
  local stack="$1" key="$2"
  aws cloudformation describe-stacks \
    --stack-name "$stack" \
    --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='${key}'].OutputValue" \
    --output text 2>/dev/null || true
}

# =============================================================================
# Step 1 — Prerequisites
# =============================================================================
if [[ "$FRONTEND_ONLY" == "false" ]]; then
  header "Step 1 — Prerequisites"

  command -v python3 >/dev/null 2>&1 || die "python3 not found. Install Python 3.10+."
  command -v pip3    >/dev/null 2>&1 || command -v pip >/dev/null 2>&1 || die "pip not found."
  command -v aws     >/dev/null 2>&1 || die "AWS CLI not found."
  command -v git     >/dev/null 2>&1 || die "git not found."

  # ── Node.js 18+ ───────────────────────────────────────────────────────────
  NODE_OK=false
  if command -v node >/dev/null 2>&1; then
    NODE_VER=$(node -e "process.stdout.write(process.versions.node.split('.')[0])")
    [[ "$NODE_VER" -ge 18 ]] && NODE_OK=true
  fi

  if [[ "$NODE_OK" == "false" ]]; then
    info "Node.js 18+ not found — installing via nvm..."
    export NVM_DIR="${HOME}/.nvm"
    if [[ ! -f "${NVM_DIR}/nvm.sh" ]]; then
      curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
    fi
    # shellcheck source=/dev/null
    source "${NVM_DIR}/nvm.sh"
    nvm install 18 --silent
    nvm use 18
    ok "Node.js $(node --version) installed via nvm"
  else
    # Ensure nvm shims are on PATH if nvm is present
    [[ -f "${HOME}/.nvm/nvm.sh" ]] && source "${HOME}/.nvm/nvm.sh"
    ok "Node.js $(node --version) present"
  fi

  # ── AWS CDK ───────────────────────────────────────────────────────────────
  if ! command -v cdk >/dev/null 2>&1; then
    info "AWS CDK not found — installing globally..."
    npm install -g aws-cdk --silent
    ok "CDK $(cdk --version) installed"
  else
    ok "CDK $(cdk --version) present"
  fi

  ok "All prerequisites present"

  # Install Python agent deps
  info "Installing/upgrading Python dependencies..."
  # --upgrade is required: SageMaker pre-installs an older boto3/botocore that
  # does not include the bedrock-agentcore-control service model. Without
  # upgrading, pip skips already-installed packages and the service stays unknown.
  python3 -m pip install --upgrade --quiet \
    boto3 botocore \
    bedrock-agentcore \
    bedrock-agentcore-starter-toolkit \
    strands-agents \
    strands-agents-tools \
    trimesh \
    manifold3d
  ok "Python dependencies installed/upgraded"
fi

# =============================================================================
# Step 2 — Git LFS (model weights)
# =============================================================================
if [[ "$FRONTEND_ONLY" == "false" && "$SKIP_INFRA" == "false" ]]; then
  header "Step 2 — Model Weights (Git LFS)"

  if ! command -v git-lfs >/dev/null 2>&1 && ! git lfs version >/dev/null 2>&1; then
    warn "git-lfs not found — skipping LFS pull. Install Git LFS if weights are not present."
  else
    info "Pulling LFS objects..."
    git lfs pull
  fi

  # Verify weights exist
  WEIGHTS_MISSING=false
  for f in \
    "backend/models/weights/kpi/best_model.pt" \
    "backend/models/weights/surface/best_model.pt" \
    "backend/models/weights/slices/ae_best_model.pt" \
    "backend/models/weights/slices/mgn_last_model.pt"
  do
    if [[ ! -f "$f" ]]; then
      warn "Missing weight file: $f"
      WEIGHTS_MISSING=true
    fi
  done

  if [[ "$WEIGHTS_MISSING" == "true" ]]; then
    warn "Some model weights are missing. The CDK Seed stack uploads these to S3."
    warn "If this is a fresh clone, ensure git lfs pull completed successfully."
    read -rp "Continue anyway? [y/N] " CONT
    [[ "${CONT,,}" == "y" ]] || die "Aborted — resolve missing weights first."
  else
    ok "All model weights present"
  fi
fi

# =============================================================================
# Step 3 — CDK Bootstrap
# =============================================================================
if [[ "$SKIP_INFRA" == "false" && "$FRONTEND_ONLY" == "false" ]]; then
  header "Step 3 — CDK Bootstrap"

  # Check if already bootstrapped by looking for the CDKToolkit stack
  BOOTSTRAP_STATUS=$(aws cloudformation describe-stacks \
    --stack-name CDKToolkit --region "$REGION" \
    --query "Stacks[0].StackStatus" --output text 2>/dev/null || echo "NOT_FOUND")

  if [[ "$BOOTSTRAP_STATUS" == "NOT_FOUND" || "$BOOTSTRAP_STATUS" == "ROLLBACK_COMPLETE" ]]; then
    info "Bootstrapping CDK for $ACCOUNT_ID / $REGION ..."
    cdk bootstrap "aws://${ACCOUNT_ID}/${REGION}"
    ok "CDK bootstrapped"
  else
    ok "CDK already bootstrapped (CDKToolkit: $BOOTSTRAP_STATUS)"
  fi
fi

# =============================================================================
# Step 4 — CDK Deploy
# =============================================================================
if [[ "$SKIP_INFRA" == "false" && "$FRONTEND_ONLY" == "false" ]]; then
  header "Step 4 — Deploy Infrastructure (CDK)"

  info "Installing CDK Python dependencies..."
  pip3 install --quiet -r infra/cdk/requirements.txt

  info "Deploying all CDK stacks (this takes ~5–10 min)..."
  (
    cd infra/cdk
    cdk deploy --all \
      --require-approval never \
      --outputs-file ../../cdk-outputs.json \
      --region "$REGION"
  )
  ok "CDK stacks deployed"

  # Re-wire orchestrator ARN to Lambda if agents were previously deployed.
  # CDK always resets ORCHESTRATOR_RUNTIME_ARN to "" on every deploy.
  if [[ -f "deployment_config.json" ]]; then
    PREV_ORCH_ARN=$(python3 -c "import json; print(json.load(open('deployment_config.json')).get('orchestrator_arn',''))" 2>/dev/null || true)
    if [[ -n "$PREV_ORCH_ARN" ]]; then
      info "Re-wiring orchestrator ARN to Lambda after CDK deploy..."
      python3 deploy_agents.py --wire-lambda \
        --orchestrator-arn "$PREV_ORCH_ARN" \
        --region "$REGION"
      ok "Orchestrator ARN re-wired: $PREV_ORCH_ARN"
    fi
  fi
fi

# =============================================================================
# Step 5 — Capture CDK Outputs
# =============================================================================
header "Step 5 — Capture CDK Outputs"

USER_POOL_ID=$(cfn_output "CarDesignAuth" "UserPoolId")
CLIENT_ID=$(cfn_output "CarDesignAuth" "UserPoolClientId")
MCP_USER_POOL_ID=$(cfn_output "CarDesignMcpAuth" "McpUserPoolId")
WS_URL=$(cfn_output "CarDesignApi" "WebSocketUrl")
FRONTEND_BUCKET=$(cfn_output "CarDesignStorage" "FrontendBucketName")
DIST_ID=$(cfn_output "CarDesignStorage" "CloudFrontDistributionId")
CF_URL=$(cfn_output "CarDesignStorage" "CloudFrontUrl")

# Validate required outputs
for var_name in USER_POOL_ID CLIENT_ID MCP_USER_POOL_ID WS_URL FRONTEND_BUCKET DIST_ID CF_URL; do
  val="${!var_name}"
  if [[ -z "$val" || "$val" == "None" ]]; then
    die "CDK output '$var_name' is empty. Check that all stacks deployed successfully."
  fi
done

echo "  UserPoolId        : $USER_POOL_ID"
echo "  UserPoolClientId  : $CLIENT_ID"
echo "  McpUserPoolId     : $MCP_USER_POOL_ID"
echo "  WebSocketUrl      : $WS_URL"
echo "  FrontendBucket    : $FRONTEND_BUCKET"
echo "  CloudFrontId      : $DIST_ID"
echo "  CloudFrontUrl     : $CF_URL"
ok "All CDK outputs captured"

# Export so subprocesses (Python heredocs) can read them via os.environ
export USER_POOL_ID MCP_USER_POOL_ID CLIENT_ID WS_URL FRONTEND_BUCKET DIST_ID CF_URL

# =============================================================================
# Step 5a — Provision Frontend Cognito User
# =============================================================================
header "Step 5a — Frontend Cognito User"

python3 - <<'PYEOF'
import hashlib
import os

import boto3

region = os.environ["CDK_DEFAULT_REGION"]
pool_id = os.environ["USER_POOL_ID"]
email = os.environ["FRONTEND_USERNAME_EMAIL"].strip().lower()
password = os.environ["FRONTEND_USERNAME_PASSWORD"]
legacy_username = "demo-user"

cognito = boto3.client("cognito-idp", region_name=region)
attributes = [
    {"Name": "email", "Value": email},
    {"Name": "email_verified", "Value": "true"},
]

# Email is configured as a sign-in alias, so Cognito requires a separate,
# non-email internal Username. Reuse an existing user by email when present;
# otherwise derive a stable, non-sensitive username from the email hash.
users = cognito.list_users(
    UserPoolId=pool_id,
    Filter=f'email = "{email}"',
    Limit=2,
).get("Users", [])
matching_users = [
    user for user in users
    if any(
        attribute.get("Name") == "email"
        and attribute.get("Value", "").lower() == email
        for attribute in user.get("Attributes", [])
    )
]

if matching_users:
    username = matching_users[0]["Username"]
    cognito.admin_update_user_attributes(
        UserPoolId=pool_id,
        Username=username,
        UserAttributes=attributes,
    )
    action = "Updated"
else:
    username = f"frontend-{hashlib.sha256(email.encode()).hexdigest()[:20]}"
    created = cognito.admin_create_user(
        UserPoolId=pool_id,
        Username=username,
        UserAttributes=attributes,
        MessageAction="SUPPRESS",
    )
    username = created["User"]["Username"]
    action = "Created"

cognito.admin_set_user_password(
    UserPoolId=pool_id,
    Username=username,
    Password=password,
    Permanent=True,
)
print(f"  [OK] {action} frontend user: {email}")

if username != legacy_username:
    try:
        legacy = cognito.admin_get_user(
            UserPoolId=pool_id,
            Username=legacy_username,
        )
        if legacy.get("Enabled", False):
            cognito.admin_disable_user(
                UserPoolId=pool_id,
                Username=legacy_username,
            )
            print("  [OK] Disabled legacy demo-user account")
    except cognito.exceptions.UserNotFoundException:
        pass
PYEOF

ok "Frontend Cognito user provisioned; password not displayed"

# =============================================================================
# Step 5b — Provision Agent M2M OAuth credentials
# =============================================================================
# The Lambda handler and orchestrator agent authenticate inter-agent (A2A) calls
# using a dedicated M2M Cognito client with client_credentials grant.
# This secret must exist before agents are deployed so the JWT authorizer
# on each AgentCore runtime can validate tokens from it.
# =============================================================================
if [[ "$SKIP_AGENTS" == "false" && "$FRONTEND_ONLY" == "false" ]]; then
  header "Step 5b — Agent M2M OAuth Credentials"

  AGENT_M2M_CLIENT_ID=$(python3 - <<'PYEOF'
import boto3, json, sys, os

region     = os.environ["CDK_DEFAULT_REGION"]
pool_id    = os.environ.get("MCP_USER_POOL_ID", "")
secret_name = "car-design/agent-oauth-credentials"

if not pool_id:
    print("[ERROR] MCP_USER_POOL_ID not set", file=sys.stderr)
    sys.exit(1)

cognito = boto3.client("cognito-idp", region_name=region)
sm      = boto3.client("secretsmanager", region_name=region)

# --- Check / create M2M client ---
existing_id = None
paginator = cognito.get_paginator("list_user_pool_clients")
for page in paginator.paginate(UserPoolId=pool_id):
    for c in page["UserPoolClients"]:
        if c["ClientName"] == "car-design-agent-m2m":
            existing_id = c["ClientId"]
            break
    if existing_id:
        break

if existing_id:
    # Fetch the secret so we can get/update it
    resp = cognito.describe_user_pool_client(UserPoolId=pool_id, ClientId=existing_id)
    client_secret = resp["UserPoolClient"].get("ClientSecret", "")
    client_id = existing_id
    print(f"  [OK] Reusing existing M2M client {client_id}", file=sys.stderr)
else:
    resp = cognito.create_user_pool_client(
        UserPoolId=pool_id,
        ClientName="car-design-agent-m2m",
        GenerateSecret=True,
        AllowedOAuthFlows=["client_credentials"],
        AllowedOAuthScopes=["mcp-api/read", "mcp-api/write"],
        AllowedOAuthFlowsUserPoolClient=True,
        ExplicitAuthFlows=[],
    )
    client = resp["UserPoolClient"]
    client_id     = client["ClientId"]
    client_secret = client["ClientSecret"]
    print(f"  [OK] Created M2M client {client_id}", file=sys.stderr)

# Derive discovery URL from pool ARN pattern
pool_resp   = cognito.describe_user_pool(UserPoolId=pool_id)
pool_domain = pool_resp["UserPool"].get("Domain", "")
discovery_url = (
    f"https://cognito-idp.{region}.amazonaws.com/{pool_id}/.well-known/openid-configuration"
)
token_url = f"https://{pool_domain}.auth.{region}.amazoncognito.com/oauth2/token"

secret_val = json.dumps({
    "client_id":      client_id,
    "client_secret":  client_secret,
    "token_url":      token_url,
    "discovery_url":  discovery_url,
    "scope":          "mcp-api/read mcp-api/write",
})

# Upsert secret
try:
    sm.create_secret(Name=secret_name, SecretString=secret_val)
    print(f"  [OK] Created secret {secret_name}", file=sys.stderr)
except sm.exceptions.ResourceExistsException:
    sm.put_secret_value(SecretId=secret_name, SecretString=secret_val)
    print(f"  [OK] Updated secret {secret_name}", file=sys.stderr)

# Print ONLY the client_id to stdout so the shell can capture it
print(client_id)
PYEOF
)
  if [[ -z "$AGENT_M2M_CLIENT_ID" ]]; then
    die "Failed to provision agent M2M client — check Python output above."
  fi

  info "Agent M2M Client ID: $AGENT_M2M_CLIENT_ID"
  ok "Agent OAuth credentials provisioned → car-design/agent-oauth-credentials"
fi

# =============================================================================
# Step 6 — Deploy Agents
# =============================================================================
if [[ "$SKIP_AGENTS" == "false" && "$FRONTEND_ONLY" == "false" ]]; then
  header "Step 6 — Deploy Agents (AgentCore Runtime)"

  AUTHORIZED_AGENT_MODEL=$(cfn_output "CarDesignAgents" "AgentModelId")
  if [[ -n "$AUTHORIZED_AGENT_MODEL" && "$AUTHORIZED_AGENT_MODEL" != "$AGENT_MODEL_ID" ]]; then
    die "CarDesignAgents IAM roles authorize '$AUTHORIZED_AGENT_MODEL', but AGENT_MODEL_ID is '$AGENT_MODEL_ID'. Run without --skip-infra so CDK can update the model permissions."
  fi
  if [[ -z "$AUTHORIZED_AGENT_MODEL" ]]; then
    die "CarDesignAgents has no AgentModelId output. Run without --skip-infra so CDK can configure model permissions."
  fi

  info "Deploying 5 agents with model $AGENT_MODEL_ID to Bedrock AgentCore (~20–30 min, parallel CodeBuild)..."

  python3 deploy_agents.py --deploy-all \
    --model-id              "$AGENT_MODEL_ID" \
    --cognito-user-pool-id  "$USER_POOL_ID" \
    --cognito-client-id     "$CLIENT_ID" \
    --agent-m2m-client-id   "$AGENT_M2M_CLIENT_ID" \
    --mcp-user-pool-id      "$MCP_USER_POOL_ID" \
    --region                "$REGION"

  ok "Agents deployed"
fi

# =============================================================================
# Step 7 — Deploy MCP Lambda
# =============================================================================
if [[ "$FRONTEND_ONLY" == "false" ]]; then
  header "Step 7 — Deploy MCP Lambda (Cost Agent tools)"

  if [[ ! -f "deployment_config.json" ]]; then
    die "deployment_config.json not found — run agent deployment first (without --skip-agents)."
  fi

  GATEWAY_ID=$(python3 -c "import json; print(json.load(open('deployment_config.json'))['gateway_id'])")
  info "MCP Gateway ID: $GATEWAY_ID"

  python3 deploy_mcp_lambda.py \
    --gateway-id "$GATEWAY_ID" \
    --region     "$REGION"

  ok "MCP Lambda deployed"
fi

# =============================================================================
# Step 8 — Build and Deploy Frontend
# =============================================================================
header "Step 8 — Build and Deploy Frontend"

info "Writing frontend/public/config.json..."
python3 - <<PYEOF
import json, sys

config = {
    "cognito": {
        "userPoolId":       "$USER_POOL_ID",
        "userPoolClientId": "$CLIENT_ID",
        "region":           "$REGION"
    },
    "websocket": {
        "url": "$WS_URL"
    }
}

with open("frontend/public/config.json", "w") as f:
    json.dump(config, f, indent=2)

print("  config.json written")
PYEOF

info "Installing npm dependencies..."
(cd frontend && npm install --silent)

info "Building React app..."
(cd frontend && npm run build)

info "Syncing to S3 (excluding config.json — already deployed)..."
aws s3 sync frontend/build/ "s3://${FRONTEND_BUCKET}/" \
  --delete \
  --exclude "config.json" \
  --region "$REGION"

info "Invalidating CloudFront cache..."
aws cloudfront create-invalidation \
  --distribution-id "$DIST_ID" \
  --paths "/*" \
  --region "$REGION" \
  --output text --query "Invalidation.Id" | xargs -I{} echo "  Invalidation ID: {}"

ok "Frontend deployed"

# =============================================================================
# Step 9 — Verify Agents
# =============================================================================
if [[ "$SKIP_AGENTS" == "false" && "$FRONTEND_ONLY" == "false" ]]; then
  header "Step 9 — Verify Agent Status"
  python3 deploy_agents.py --list --region "$REGION"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${BOLD}${GREEN}═══ Deployment Complete ═══${NC}"
echo ""
echo -e "  Frontend URL     : ${CYAN}${CF_URL}${NC}"
echo -e "  WebSocket        : ${CYAN}${WS_URL}${NC}"
echo -e "  Demo login email : ${CYAN}${FRONTEND_USERNAME_EMAIL}${NC}"
echo "  Demo password    : configured from FRONTEND_USERNAME_PASSWORD (not displayed)"
echo ""
