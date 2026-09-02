#!/usr/bin/env bash
# =============================================================================
# Car Design Space Explorer — Full Teardown Script
# Run from the car-design-explorer/ directory.
#
# Usage:
#   ./destroy.sh                   # Full teardown (prompts for confirmation)
#   ./destroy.sh --force           # Skip confirmation prompts
#   ./destroy.sh --region us-west-2
#   ./destroy.sh --skip-agents     # Skip AgentCore agent deletion
#   ./destroy.sh --skip-cdk        # Skip CDK destroy (manual teardown only)
# =============================================================================
set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
ok()     { echo -e "${GREEN}[OK]${NC}  $*"; }
info()   { echo -e "${CYAN}[--]${NC}  $*"; }
warn()   { echo -e "${YELLOW}[!!]${NC}  $*"; }
die()    { echo -e "${RED}[ERR]${NC} $*" >&2; exit 1; }
header() { echo -e "\n${BOLD}${CYAN}═══ $* ═══${NC}"; }
skip()   { echo -e "${YELLOW}[SK]${NC}  $*"; }

# ── Flags ─────────────────────────────────────────────────────────────────────
FORCE=false
SKIP_AGENTS=false
SKIP_CDK=false
REGION=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --force)        FORCE=true;        shift ;;
    --skip-agents)  SKIP_AGENTS=true;  shift ;;
    --skip-cdk)     SKIP_CDK=true;     shift ;;
    --region)       REGION="$2";       shift 2 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

# ── Resolve region & account ──────────────────────────────────────────────────
if [[ -z "$REGION" ]]; then
  REGION=$(aws configure get region 2>/dev/null || true)
  REGION=${REGION:-us-east-1}
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo -e "\n${BOLD}${RED}Car Design Space Explorer — Teardown${NC}"
echo "  Account : $ACCOUNT_ID"
echo "  Region  : $REGION"
echo ""

# Verify we are in the right directory
[[ -f "deploy_agents.py" && -d "frontend" && -d "infra" ]] \
  || die "Run this script from the car-design-explorer/ directory."

# ── Confirmation gate ─────────────────────────────────────────────────────────
if [[ "$FORCE" == "false" ]]; then
  echo -e "${RED}${BOLD}WARNING: This will permanently delete ALL Car Design Explorer resources${NC}"
  echo "  - All 5 AgentCore agent runtimes"
  echo "  - MCP Gateway"
  echo "  - Orphaned AgentCore Memory resources"
  echo "  - All S3 bucket contents (models, geometries, frontend)"
  echo "  - All CDK stacks (DynamoDB, Lambda, API GW, Cognito, CloudFront, ...)"
  echo "  - Secrets Manager secrets"
  echo ""
  read -rp "Type 'destroy' to confirm: " CONFIRM
  [[ "$CONFIRM" == "destroy" ]] || die "Aborted."
fi

# =============================================================================
# Helper: empty an S3 bucket including all versions and delete markers
# =============================================================================
empty_bucket() {
  local bucket="$1"

  # Check if bucket exists
  if ! aws s3api head-bucket --bucket "$bucket" --region "$REGION" 2>/dev/null; then
    skip "Bucket $bucket does not exist — skipping"
    return 0
  fi

  info "Emptying $bucket ..."

  # Delete current objects
  aws s3 rm "s3://${bucket}" --recursive --region "$REGION" --quiet || true

  # Buckets are now versioned — also purge all object versions and delete markers
  # so the bucket can be deleted by CDK. (auto_delete_objects also covers this,
  # but doing it here avoids surprises if CDK destroy is skipped.)
  while true; do
    VERSIONS=$(aws s3api list-object-versions \
      --bucket "$bucket" --region "$REGION" --max-items 500 \
      --query '{Objects: Versions[].{Key:Key,VersionId:VersionId}, DeleteMarkers: DeleteMarkers[].{Key:Key,VersionId:VersionId}}' \
      --output json 2>/dev/null || echo '{}')

    DELETE_PAYLOAD=$(python3 - "$VERSIONS" <<'PYEOF' 2>/dev/null || true
import json, sys
data = json.loads(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else {}
items = (data.get("Objects") or []) + (data.get("DeleteMarkers") or [])
items = [i for i in items if i and i.get("Key") and i.get("VersionId")]
print(json.dumps({"Objects": items, "Quiet": True}) if items else "")
PYEOF
)
    [[ -z "$DELETE_PAYLOAD" ]] && break
    aws s3api delete-objects --bucket "$bucket" --region "$REGION" \
      --delete "$DELETE_PAYLOAD" >/dev/null 2>&1 || break
  done

  ok "Bucket $bucket emptied (incl. versions)"
}

# =============================================================================
# Step 1 — Delete AgentCore Agent Runtimes
# =============================================================================
if [[ "$SKIP_AGENTS" == "false" ]]; then

  # Read IDs directly from deployment_config.json written by deploy_agents.py
  CONFIG_FILE="$(pwd)/deployment_config.json"

  python3 - "$CONFIG_FILE" "$REGION" <<'PYEOF'
import boto3, json, os, sys, time
import glob as _glob

config_file = sys.argv[1]
region      = sys.argv[2] if len(sys.argv) > 2 else "us-east-1"
script_dir  = os.path.dirname(os.path.abspath(config_file))

agents     = {}   # {agent_key: arn_or_id}
memory_id  = ""
gateway_id = ""

if os.path.exists(config_file):
    with open(config_file) as f:
        cfg = json.load(f)
    region     = cfg.get("region", region)
    agents     = cfg.get("agents", {})
    memory_id  = cfg.get("memory_id", "")
    gateway_id = cfg.get("gateway_id", "")
    print(f"  Loaded deployment_config.json ({len(agents)} agents)")
else:
    # Fall back to individual per-agent deployment JSON files written by deploy_agents.py
    per_agent_files = sorted(_glob.glob(os.path.join(script_dir, "car_design_*_deployment.json")))
    if not per_agent_files:
        print("  [SK] No deployment config found — nothing to delete")
        sys.exit(0)
    print(f"  Loaded {len(per_agent_files)} individual deployment JSON files (fallback)")
    for f_path in per_agent_files:
        with open(f_path) as f:
            info = json.load(f)
        key      = info.get("agent_key", "")
        agent_id = info.get("agent_id", "")
        if key and agent_id:
            agents[key] = agent_id  # raw ID — split("/")[-1] below handles ARNs too
        if key == "orchestrator" and info.get("memory_id"):
            memory_id = info["memory_id"]
    gw_config = os.path.join(script_dir, "mcp_gateway_config.json")
    if os.path.exists(gw_config):
        with open(gw_config) as f:
            gw = json.load(f)
        gateway_id = gw.get("gateway_id", "")

client = boto3.client("bedrock-agentcore-control", region_name=region)

# ── 1. Delete agent runtimes ──────────────────────────────────────────────
print("\n═══ Step 1 — Delete AgentCore Agent Runtimes ═══")
if not agents:
    print("  [SK] No agents in deployment_config.json")
else:
    for key, arn in agents.items():
        # ARN format: arn:aws:bedrock-agentcore:region:account:agent-runtime/ID
        runtime_id = arn.split("/")[-1] if "/" in arn else arn
        print(f"  Deleting {key}: {runtime_id}")
        try:
            client.delete_agent_runtime(agentRuntimeId=runtime_id)
            print(f"  [OK] Deleted {key}")
        except Exception as e:
            print(f"  [WARN] {key}: {e}")
    print("  Waiting 10s for deletions to propagate...")
    time.sleep(10)

# ── 2. Delete memory ──────────────────────────────────────────────────────
print("\n═══ Step 1b — Delete AgentCore Memory ═══")
if not memory_id:
    print("  [SK] No memory_id in deployment_config.json")
else:
    print(f"  Deleting memory: {memory_id}")
    try:
        client.delete_memory(memoryId=memory_id)
        print(f"  [OK] Deleted memory {memory_id}")
    except Exception as e:
        print(f"  [WARN] {e}")

# ── 3. Delete MCP Gateway (targets first) ────────────────────────────────
print("\n═══ Step 1c — Delete MCP Gateway ═══")
if not gateway_id:
    print("  [SK] No gateway_id in deployment_config.json")
else:
    print(f"  Deleting gateway: {gateway_id}")
    # Delete all targets first
    targets_deleted = 0
    try:
        resp = client.list_gateway_targets(gatewayIdentifier=gateway_id, maxResults=50)
        targets = resp.get("items", resp.get("gatewayTargetSummaries", []))
        for t in targets:
            t_id = t.get("targetId", t.get("id", ""))
            if not t_id:
                continue
            print(f"    Deleting target: {t_id}")
            try:
                client.delete_gateway_target(gatewayIdentifier=gateway_id, targetId=t_id)
                print(f"    [OK] Deleted target {t_id}")
                targets_deleted += 1
            except Exception as e:
                print(f"    [WARN] target {t_id}: {e}")
    except Exception as e:
        print(f"  [WARN] list targets: {e}")

    if targets_deleted > 0:
        print("  Waiting 10s for target deletions to propagate...")
        time.sleep(10)

    for attempt in range(3):
        try:
            client.delete_gateway(gatewayIdentifier=gateway_id)
            print(f"  [OK] Deleted gateway {gateway_id}")
            break
        except Exception as e:
            if "targets associated" in str(e) and attempt < 2:
                print(f"  [WARN] Still has targets, retrying in 10s ({attempt+1}/3)...")
                time.sleep(10)
            else:
                print(f"  [WARN] {e}")
                break

PYEOF

  ok "AgentCore resources deleted (or already absent)"
else
  skip "AgentCore resources — skipped (--skip-agents)"
fi

# =============================================================================
# Step 2 — Empty S3 Buckets
# =============================================================================
header "Step 2 — Empty S3 Buckets"

empty_bucket "car-design-explorer-models-${ACCOUNT_ID}"
empty_bucket "car-design-explorer-geometries-${ACCOUNT_ID}"
empty_bucket "car-design-explorer-frontend-${ACCOUNT_ID}"
empty_bucket "car-design-explorer-logs-${ACCOUNT_ID}"

ok "All S3 buckets emptied"

# =============================================================================
# Step 3 — CDK Destroy
# =============================================================================
if [[ "$SKIP_CDK" == "false" ]]; then
  header "Step 3 — CDK Destroy (all stacks)"

  if ! command -v cdk >/dev/null 2>&1; then
    die "AWS CDK (cdk) not found. Install with: npm install -g aws-cdk"
  fi

  if ! python3 -c "import aws_cdk" 2>/dev/null; then
    info "Installing CDK Python dependencies..."
    pip3 install --quiet -r infra/cdk/requirements.txt
  fi

  info "Destroying all CDK stacks (this may take 5–15 min)..."
  (
    cd infra/cdk
    CDK_DEFAULT_ACCOUNT="$ACCOUNT_ID" CDK_DEFAULT_REGION="$REGION" \
    cdk destroy --all --force --region "$REGION"
  )

  ok "CDK stacks destroyed"
else
  skip "CDK destroy — skipped (--skip-cdk)"
fi

# =============================================================================
# Step 3a — Ensure the VPC / network stack tears down (Lambda ENI cleanup)
# =============================================================================
# VPC-attached Lambdas leave Hyperplane ENIs that can take 20-40 min to detach,
# which blocks deletion of the security group / subnets / VPC. This step deletes
# any detached (status=available) ENIs in the VPC and retries the network stack
# destroy a few times so teardown completes in one run where possible.
if [[ "$SKIP_CDK" == "false" ]]; then
  header "Step 3a — VPC Teardown Assurance"

  NET_STATUS=$(aws cloudformation describe-stacks \
    --stack-name CarDesignNetwork --region "$REGION" \
    --query "Stacks[0].StackStatus" --output text 2>/dev/null || echo "NOT_FOUND")

  if [[ "$NET_STATUS" == "NOT_FOUND" ]]; then
    ok "CarDesignNetwork already deleted"
  else
    VPC_ID=$(aws ec2 describe-vpcs --region "$REGION" \
      --filters "Name=tag:Name,Values=car-design-explorer-vpc" \
      --query "Vpcs[0].VpcId" --output text 2>/dev/null || echo "None")

    for attempt in 1 2 3 4; do
      # Delete any detached ENIs sitting in our VPC
      if [[ "$VPC_ID" != "None" && -n "$VPC_ID" ]]; then
        AVAIL_ENIS=$(aws ec2 describe-network-interfaces --region "$REGION" \
          --filters "Name=vpc-id,Values=${VPC_ID}" "Name=status,Values=available" \
          --query "NetworkInterfaces[].NetworkInterfaceId" --output text 2>/dev/null || true)
        for eni in $AVAIL_ENIS; do
          info "Deleting detached ENI: $eni"
          aws ec2 delete-network-interface --network-interface-id "$eni" \
            --region "$REGION" 2>/dev/null || true
        done
      fi

      # Retry destroying just the network stack
      info "Destroying CarDesignNetwork (attempt ${attempt}/4)..."
      (
        cd infra/cdk
        CDK_DEFAULT_ACCOUNT="$ACCOUNT_ID" CDK_DEFAULT_REGION="$REGION" \
        cdk destroy CarDesignNetwork --force --region "$REGION"
      ) || true

      NET_STATUS=$(aws cloudformation describe-stacks \
        --stack-name CarDesignNetwork --region "$REGION" \
        --query "Stacks[0].StackStatus" --output text 2>/dev/null || echo "NOT_FOUND")
      if [[ "$NET_STATUS" == "NOT_FOUND" ]]; then
        ok "CarDesignNetwork deleted"
        break
      fi

      if [[ "$attempt" -lt 4 ]]; then
        warn "Network stack still present (ENIs likely still detaching). Waiting 120s..."
        sleep 120
      else
        warn "CarDesignNetwork did not delete after retries. Lambda ENIs can take"
        warn "up to ~40 min to detach — re-run ./destroy.sh later to finish."
      fi
    done
  fi
fi

# =============================================================================
# Step 3b — Delete orphaned CloudWatch Log Groups
# =============================================================================
# Lambda auto-creates /aws/lambda/<fn> log groups that are NOT owned by
# CloudFormation, so they survive `cdk destroy`. The API stack now creates an
# explicit, KMS-encrypted log group for the WS handler — if an orphaned one is
# left behind, the next `deploy.sh` fails with "log group already exists".
# Delete the known Lambda log groups so redeploy is clean.
header "Step 3b — Delete orphaned CloudWatch Log Groups"

for lg in \
  "/aws/lambda/CarDesignWSHandler" \
  "/aws/lambda/CarDesignDynamoSeeder" \
  "/aws/lambda/CarDesignCostMCPHandler"
do
  if aws logs describe-log-groups \
       --log-group-name-prefix "$lg" --region "$REGION" \
       --query "logGroups[?logGroupName=='${lg}'].logGroupName" \
       --output text 2>/dev/null | grep -q "$lg"; then
    info "Deleting log group: $lg"
    aws logs delete-log-group --log-group-name "$lg" --region "$REGION" 2>/dev/null \
      && ok "Deleted log group: $lg" \
      || warn "Could not delete log group: $lg"
  else
    skip "Log group not found: $lg"
  fi
done

# =============================================================================
# Step 4 — Clean up Secrets Manager
# =============================================================================
header "Step 4 — Delete Secrets Manager Secrets"

for secret_name in \
  "car-design/mcp-gateway-credentials" \
  "car-design/agent-oauth-credentials"
do
  if aws secretsmanager describe-secret \
      --secret-id "$secret_name" \
      --region "$REGION" \
      --output text > /dev/null 2>&1; then
    info "Deleting secret: $secret_name"
    aws secretsmanager delete-secret \
      --secret-id "$secret_name" \
      --force-delete-without-recovery \
      --region "$REGION" \
      --output text > /dev/null
    ok "Deleted secret: $secret_name"
  else
    skip "Secret not found: $secret_name"
  fi
done

# =============================================================================
# Step 5 — Verification
# =============================================================================
header "Step 5 — Verification"

ISSUES=0

# Check CDK stacks are gone
if [[ "$SKIP_CDK" == "false" ]]; then
  for stack in CarDesignAuth CarDesignMcpAuth CarDesignData CarDesignStorage CarDesignNetwork \
               CarDesignApi CarDesignAgents CarDesignSeed CarDesignDynamoSeed CarDesignFrontend; do
    STATUS=$(aws cloudformation describe-stacks \
      --stack-name "$stack" \
      --region "$REGION" \
      --query "Stacks[0].StackStatus" \
      --output text 2>/dev/null || echo "NOT_FOUND")
    if [[ "$STATUS" == "NOT_FOUND" ]]; then
      ok "Stack $stack: deleted"
    else
      warn "Stack $stack still exists (status: $STATUS)"
      ISSUES=$((ISSUES + 1))
      if [[ "$stack" == "CarDesignNetwork" ]]; then
        warn "  CarDesignNetwork often fails first-pass teardown because the VPC"
        warn "  Lambda's elastic network interfaces (ENIs) can take 20-40 min to"
        warn "  detach. Wait a few minutes and re-run ./destroy.sh — it will clear."
      fi
    fi
  done
fi

# Check S3 buckets are gone
for bucket in \
  "car-design-explorer-models-${ACCOUNT_ID}" \
  "car-design-explorer-geometries-${ACCOUNT_ID}" \
  "car-design-explorer-frontend-${ACCOUNT_ID}" \
  "car-design-explorer-logs-${ACCOUNT_ID}"
do
  if aws s3api head-bucket --bucket "$bucket" --region "$REGION" 2>/dev/null; then
    warn "Bucket still exists: $bucket"
    ISSUES=$((ISSUES + 1))
  else
    ok "Bucket $bucket: gone"
  fi
done

# Check AgentCore agents
if [[ "$SKIP_AGENTS" == "false" ]]; then
  REMAINING=$(python3 - 2>/dev/null <<PYEOF || true
import boto3
try:
    client = boto3.client("bedrock-agentcore-control", region_name="$REGION")
    resp = client.list_agent_runtimes(maxResults=100)
    runtimes = resp.get("agentRuntimeSummaries", resp.get("items", []))
    remaining = [r for r in runtimes if "car_design" in r.get("agentRuntimeName", r.get("name", "")).lower()]
    print(len(remaining))
except Exception:
    print(0)
PYEOF
)
  if [[ "${REMAINING:-0}" == "0" ]]; then
    ok "AgentCore agents: all deleted"
  else
    warn "AgentCore: $REMAINING agent runtime(s) still present"
    ISSUES=$((ISSUES + 1))
  fi
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
if [[ "$ISSUES" -eq 0 ]]; then
  echo -e "${BOLD}${GREEN}═══ Teardown Complete — All resources removed ═══${NC}"
else
  echo -e "${BOLD}${YELLOW}═══ Teardown Complete — $ISSUES issue(s) noted above ═══${NC}"
  echo "  Some resources may still exist. Review warnings above."
fi
echo ""

# Clean up local deployment config files
if [[ -f "deployment_config.json" ]]; then
  info "Removing deployment_config.json..."
  rm -f deployment_config.json
  ok "deployment_config.json removed"
fi
if [[ -f "mcp_gateway_config.json" ]]; then
  info "Removing mcp_gateway_config.json..."
  rm -f mcp_gateway_config.json
  ok "mcp_gateway_config.json removed"
fi
if [[ -f "cdk-outputs.json" ]]; then
  info "Removing cdk-outputs.json..."
  rm -f cdk-outputs.json
  ok "cdk-outputs.json removed"
fi
# Remove individual per-agent deployment JSON files (fallback config)
for f in car_design_*_deployment.json; do
  [[ -f "$f" ]] || continue
  info "Removing $f..."
  rm -f "$f"
  ok "$f removed"
done
