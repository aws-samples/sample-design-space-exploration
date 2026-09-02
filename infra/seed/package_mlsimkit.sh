#!/bin/bash
# =============================================================================
# Package MLSimKit from EC2 for Docker container inclusion
#
# MLSimKit is NOT a public pip package — it's installed from the AWS tutorial
# repo. This script discovers the full installation (Python packages + CLI
# binary) and packages it into a tarball that can be included in the agent
# Docker image.
#
# Run on EC2 (MyMLTrain instance):
#   chmod +x package_mlsimkit.sh
#   ./package_mlsimkit.sh
# =============================================================================

set -e

S3_BUCKET="car-design-explorer-models-${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID}"
PACKAGE_DIR="/home/ubuntu/mlsimkit_package"
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
SITE_PACKAGES=$(python3 -c "import site; print(site.getusersitepackages())")
SYSTEM_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")

echo "============================================================"
echo "Packaging MLSimKit for Docker container"
echo "============================================================"
echo "Python version: ${PYTHON_VERSION}"
echo "User site-packages: ${SITE_PACKAGES}"
echo "System site-packages: ${SYSTEM_SITE}"
echo ""

rm -rf "${PACKAGE_DIR}"
mkdir -p "${PACKAGE_DIR}/site-packages"
mkdir -p "${PACKAGE_DIR}/bin"

# Step 1: Find mlsimkit Python package
echo "Step 1: Locating mlsimkit Python package..."

MLSIMKIT_PKG=""
for candidate in "${SITE_PACKAGES}/mlsimkit" "${SYSTEM_SITE}/mlsimkit" "/home/ubuntu/.local/lib/python${PYTHON_VERSION}/site-packages/mlsimkit"; do
    if [ -d "$candidate" ]; then
        MLSIMKIT_PKG="$candidate"
        break
    fi
done

if [ -z "$MLSIMKIT_PKG" ]; then
    echo "Searching more broadly..."
    MLSIMKIT_PKG=$(python3 -c "import mlsimkit; print(mlsimkit.__path__[0])" 2>/dev/null || echo "")
fi

if [ -z "$MLSIMKIT_PKG" ]; then
    echo "ERROR: Cannot find mlsimkit Python package"
    exit 1
fi

echo "  Found mlsimkit at: ${MLSIMKIT_PKG}"
MLSIMKIT_PARENT=$(dirname "${MLSIMKIT_PKG}")
echo "  Parent site-packages: ${MLSIMKIT_PARENT}"

# Step 2: Copy mlsimkit package and its dependencies
echo ""
echo "Step 2: Copying mlsimkit package and metadata..."

# Copy the mlsimkit package itself
cp -r "${MLSIMKIT_PKG}" "${PACKAGE_DIR}/site-packages/"

# Copy dist-info/egg-info for mlsimkit
for meta in "${MLSIMKIT_PARENT}"/mlsimkit*.dist-info "${MLSIMKIT_PARENT}"/mlsimkit*.egg-info; do
    if [ -d "$meta" ]; then
        cp -r "$meta" "${PACKAGE_DIR}/site-packages/"
        echo "  Copied metadata: $(basename $meta)"
    fi
done

# AgentCore uses MLSimKit for inference only. Keep the acceleration symbol
# importable by mlsimkit.learn.cli, but remove its subprocess-based training
# launcher from the production package.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCELERATE_STUB="${SCRIPT_DIR}/mlsimkit_accelerate_inference_stub.py"
if [ ! -f "${ACCELERATE_STUB}" ]; then
    echo "ERROR: Missing inference-only acceleration stub: ${ACCELERATE_STUB}"
    exit 1
fi
cp "${ACCELERATE_STUB}" "${PACKAGE_DIR}/site-packages/mlsimkit/learn/accelerate.py"
rm -f "${PACKAGE_DIR}"/site-packages/mlsimkit/learn/__pycache__/accelerate*.pyc

# Step 3: Find and copy mlsimkit-learn CLI binary
echo ""
echo "Step 3: Copying mlsimkit-learn CLI binary..."

MLSIMKIT_BIN=$(which mlsimkit-learn 2>/dev/null || echo "")
if [ -z "$MLSIMKIT_BIN" ]; then
    MLSIMKIT_BIN="/home/ubuntu/.local/bin/mlsimkit-learn"
fi

if [ -f "$MLSIMKIT_BIN" ]; then
    cp "$MLSIMKIT_BIN" "${PACKAGE_DIR}/bin/"
    echo "  Copied: ${MLSIMKIT_BIN}"
    echo "  Content (first 5 lines):"
    head -5 "$MLSIMKIT_BIN"
else
    echo "  WARNING: mlsimkit-learn binary not found"
fi

# Step 4: Identify mlsimkit's dependencies that aren't standard
echo ""
echo "Step 4: Identifying mlsimkit dependencies..."

# Get the list of packages mlsimkit imports
python3 << 'PYEOF'
import importlib
import os
import sys

# Find what mlsimkit needs
try:
    import mlsimkit
    pkg_dir = os.path.dirname(mlsimkit.__path__[0])
    
    # Check for requirements in dist-info
    for item in os.listdir(pkg_dir):
        if item.startswith('mlsimkit') and item.endswith('.dist-info'):
            req_file = os.path.join(pkg_dir, item, 'REQUIRES')
            meta_file = os.path.join(pkg_dir, item, 'METADATA')
            
            if os.path.exists(req_file):
                print(f"Requirements from {item}/REQUIRES:")
                with open(req_file) as f:
                    print(f.read())
            
            if os.path.exists(meta_file):
                print(f"\nDependencies from {item}/METADATA:")
                with open(meta_file) as f:
                    for line in f:
                        if line.startswith('Requires-Dist:'):
                            print(f"  {line.strip()}")
except Exception as e:
    print(f"Error: {e}")

# Also check what's importable
print("\nKey packages check:")
for pkg in ['torch', 'torch_geometric', 'pyvista', 'trimesh', 'numpy', 'pandas', 'scipy']:
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, '__version__', 'unknown')
        loc = getattr(mod, '__path__', [getattr(mod, '__file__', 'unknown')])[0]
        print(f"  {pkg}: {ver} at {loc}")
    except ImportError:
        print(f"  {pkg}: NOT INSTALLED")
PYEOF

# Step 5: Create a requirements.txt from the EC2 environment
echo ""
echo "Step 5: Generating requirements.txt from EC2 pip freeze..."

pip3 freeze 2>/dev/null | grep -iE "mlsimkit|torch|geometric|pyvista|trimesh|vtk|numpy|pandas|scipy|click|pydantic|boto3|pyyaml|tqdm|pillow|matplotlib" > "${PACKAGE_DIR}/requirements_mlsimkit.txt"
echo "  Saved to: ${PACKAGE_DIR}/requirements_mlsimkit.txt"
cat "${PACKAGE_DIR}/requirements_mlsimkit.txt"

# Step 6: Also get the full pip list for reference
pip3 list --format=freeze 2>/dev/null > "${PACKAGE_DIR}/pip_freeze_full.txt"

# Step 7: Package everything into a tarball
echo ""
echo "Step 6: Creating tarball..."

cd /home/ubuntu
tar czf mlsimkit_package.tar.gz -C "${PACKAGE_DIR}" .
TARBALL_SIZE=$(du -h mlsimkit_package.tar.gz | cut -f1)
echo "  Tarball: /home/ubuntu/mlsimkit_package.tar.gz (${TARBALL_SIZE})"

# Step 8: Upload to S3
echo ""
echo "Step 7: Uploading to S3..."

aws s3 cp /home/ubuntu/mlsimkit_package.tar.gz "s3://${S3_BUCKET}/mlsimkit/mlsimkit_package.tar.gz"
aws s3 cp "${PACKAGE_DIR}/requirements_mlsimkit.txt" "s3://${S3_BUCKET}/mlsimkit/requirements_mlsimkit.txt"
aws s3 cp "${PACKAGE_DIR}/pip_freeze_full.txt" "s3://${S3_BUCKET}/mlsimkit/pip_freeze_full.txt"

echo ""
echo "============================================================"
echo "DONE"
echo "============================================================"
echo ""
echo "Package contents:"
ls -la "${PACKAGE_DIR}/site-packages/"
ls -la "${PACKAGE_DIR}/bin/"
echo ""
echo "S3 artifacts:"
aws s3 ls "s3://${S3_BUCKET}/mlsimkit/"
echo ""
echo "Next steps:"
echo "  1. Download requirements_mlsimkit.txt locally"
echo "  2. Use it in the Aero Agent Dockerfile to pip install dependencies"
echo "  3. Copy the mlsimkit package into the Docker image's site-packages"
echo "  4. Copy the mlsimkit-learn binary into /usr/local/bin/"
