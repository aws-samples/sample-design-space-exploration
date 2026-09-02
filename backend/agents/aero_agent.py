#!/usr/bin/env python3
"""
Aero Agent — Strands A2A Server for aerodynamic evaluation.

Evaluates car body design variants for aerodynamic KPIs (Cd, Cs, Cl, Cmy)
and surface variable distributions (cpavg, cfxavg) using MLSimKit surrogate
models. Implements a two-tier data access pattern: DynamoDB cache for known
variants, live model inference for new geometries.

Architecture:
- Strands Agent framework with @tool decorated functions
- A2A Server for agent-to-agent communication
- DynamoDB variant cache for fast lookups
- MLSimKit inference for cache misses and surface data
- Deployed to Bedrock AgentCore Runtime
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Bootstrap: Install MLSimKit from bundled tarball if not already available
# ---------------------------------------------------------------------------
import os
import sys

_TRACKING_STUB = '''\
"""No-op tracking stub for inference-only environments."""

class TrackerBase:
    def __init__(self, *a, **kw): pass
    def log_param(self, *a, **kw): pass
    def log_params(self, *a, **kw): pass
    def log_metric(self, *a, **kw): pass
    def log_metrics(self, *a, **kw): pass
    def log_artifact(self, *a, **kw): pass
    def log_artifacts(self, *a, **kw): pass
    def start_run(self, *a, **kw): return self
    def end_run(self, *a, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass

class Tracker(TrackerBase): pass

def get_tracker(*a, **kw): return Tracker()
def init_tracking(*a, **kw): pass
def log_param(*a, **kw): pass
def log_metric(*a, **kw): pass
def log_artifacts(*a, **kw): pass
def log_artifact(*a, **kw): pass

class _NullContext:
    def __enter__(self): return self
    def __exit__(self, *a): pass

def context(*a, **kw): return _NullContext()
'''

# Training-only packages that mlsimkit imports but we don't need for inference.
# These get stubbed both in-process (sys.modules) and on disk (for CLI subprocess).
_TRAINING_ONLY_PACKAGES = {
    "accelerate": {
        "__init__": (
            "class PartialState:\n"
            "    _shared_state = {}\n"
            "    def __init__(self, *a, **kw):\n"
            "        self.process_index = 0\n"
            "        self.local_process_index = 0\n"
            "        self.num_processes = 1\n"
            "        self.is_main_process = True\n"
            "        self.is_local_main_process = True\n"
            "class Accelerator:\n"
            "    def __init__(self, *a, **kw):\n"
            "        self.process_index = 0\n"
            "        self.local_process_index = 0\n"
            "        self.num_processes = 1\n"
            "        self.is_main_process = True\n"
            "        self.is_local_main_process = True\n"
            "        self.device = 'cpu'\n"
            "    def wait_for_everyone(self): pass\n"
            "    def print(self, *a, **kw): pass\n"
            "    def gather(self, x): return x\n"
        ),
        "submodules": ["state"],
        "state": (
            "class PartialState:\n"
            "    _shared_state = {}\n"
            "    def __init__(self, *a, **kw):\n"
            "        self.process_index = 0\n"
            "        self.local_process_index = 0\n"
            "        self.num_processes = 1\n"
            "        self.is_main_process = True\n"
            "        self.is_local_main_process = True\n"
            "class Accelerator:\n"
            "    def __init__(self, *a, **kw):\n"
            "        self.process_index = 0\n"
            "        self.local_process_index = 0\n"
            "        self.num_processes = 1\n"
            "        self.is_main_process = True\n"
            "        self.is_local_main_process = True\n"
            "        self.device = 'cpu'\n"
            "    def wait_for_everyone(self): pass\n"
            "    def print(self, *a, **kw): pass\n"
            "    def gather(self, x): return x\n"
        ),
    },
    "mlflow": {
        "__init__": (
            "__version__ = '0.0.0-stub'\n"
            "def log_param(*a, **kw): pass\n"
            "def log_params(*a, **kw): pass\n"
            "def log_metric(*a, **kw): pass\n"
            "def log_metrics(*a, **kw): pass\n"
            "def log_artifact(*a, **kw): pass\n"
            "def log_artifacts(*a, **kw): pass\n"
            "def start_run(*a, **kw): return type('R',(),{'__enter__':lambda s:s,'__exit__':lambda s,*a:None})()\n"
            "def end_run(*a, **kw): pass\n"
            "def set_tracking_uri(*a, **kw): pass\n"
            "def set_experiment(*a, **kw): pass\n"
            "def active_run(): return None\n"
        ),
        "submodules": ["tracking", "entities", "utils"],
    },
    "sklearn": {
        "__init__": "",
        "submodules": ["metrics", "preprocessing", "model_selection"],
        "metrics": (
            "def mean_squared_error(*a, **kw): return 0.0\n"
            "def mean_absolute_error(*a, **kw): return 0.0\n"
            "def r2_score(*a, **kw): return 1.0\n"
        ),
        "preprocessing": "",
        "model_selection": "",
    },
}


def _install_training_stubs():
    """Inject explicit no-op modules for training-only dependencies.

    The in-process modules are populated through normal attribute assignment;
    static source strings are written only for the isolated CLI subprocess.
    No dynamic source execution is used.
    """
    import importlib.machinery
    import types

    class PartialState:
        _shared_state = {}

        def __init__(self, *args, **kwargs):
            self.process_index = 0
            self.local_process_index = 0
            self.num_processes = 1
            self.is_main_process = True
            self.is_local_main_process = True

    class Accelerator(PartialState):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.device = "cpu"

        def wait_for_everyone(self):
            return None

        def print(self, *args, **kwargs):
            return None

        def gather(self, value):
            return value

    class NullRun:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    def no_op(*args, **kwargs):
        return None

    def populate_module(module, package_name, submodule_name=None):
        if package_name == "accelerate":
            module.PartialState = PartialState
            module.Accelerator = Accelerator
        elif package_name == "mlflow" and submodule_name is None:
            module.__version__ = "0.0.0-stub"
            module.log_param = no_op
            module.log_params = no_op
            module.log_metric = no_op
            module.log_metrics = no_op
            module.log_artifact = no_op
            module.log_artifacts = no_op
            module.start_run = lambda *args, **kwargs: NullRun()
            module.end_run = no_op
            module.set_tracking_uri = no_op
            module.set_experiment = no_op
            module.active_run = lambda: None
        elif package_name == "sklearn" and submodule_name == "metrics":
            module.mean_squared_error = lambda *args, **kwargs: 0.0
            module.mean_absolute_error = lambda *args, **kwargs: 0.0
            module.r2_score = lambda *args, **kwargs: 1.0

    stub_base = "/tmp/training_stubs"
    os.makedirs(stub_base, exist_ok=True)

    for pkg_name, pkg_def in _TRAINING_ONLY_PACKAGES.items():
        mod = types.ModuleType(pkg_name)
        mod.__path__ = [os.path.join(stub_base, pkg_name)]
        mod.__file__ = f"<stub:{pkg_name}>"
        mod.__spec__ = importlib.machinery.ModuleSpec(pkg_name, None, is_package=True)
        mod.__spec__.submodule_search_locations = mod.__path__
        populate_module(mod, pkg_name)
        sys.modules[pkg_name] = mod

        for sub in pkg_def.get("submodules", []):
            fqn = f"{pkg_name}.{sub}"
            sub_mod = types.ModuleType(fqn)
            sub_mod.__path__ = [os.path.join(stub_base, pkg_name, sub)]
            sub_mod.__spec__ = importlib.machinery.ModuleSpec(fqn, None, is_package=True)
            sub_mod.__spec__.submodule_search_locations = sub_mod.__path__
            populate_module(sub_mod, pkg_name, sub)
            sys.modules[fqn] = sub_mod
            setattr(mod, sub, sub_mod)

        # The CLI runs in a separate process and imports these static files.
        pkg_dir = os.path.join(stub_base, pkg_name)
        os.makedirs(pkg_dir, exist_ok=True)
        with open(os.path.join(pkg_dir, "__init__.py"), "w") as f:
            f.write(f"# Auto-generated {pkg_name} stub\n")
            f.write(pkg_def["__init__"])
        for sub in pkg_def.get("submodules", []):
            sub_dir = os.path.join(pkg_dir, sub)
            os.makedirs(sub_dir, exist_ok=True)
            sub_code = pkg_def.get(sub, "")
            with open(os.path.join(sub_dir, "__init__.py"), "w") as f:
                f.write(f"# Auto-generated {pkg_name}.{sub} stub\n")
                if sub_code:
                    f.write(sub_code)

    print(f"[mlsimkit-bootstrap] Training stubs installed: {list(_TRAINING_ONLY_PACKAGES.keys())}")
    return stub_base



def _patch_all_training_imports(mlsimkit_root: str):
    """Scan EVERY .py file under mlsimkit and neutralize training-only imports.

    Instead of patching individual files one-by-one (whack-a-mole), this
    walks the entire tree and replaces any line that imports accelerate or
    mlflow with a safe inline stub. This is the nuclear option.
    """
    import re

    # Full inline stubs for accelerate classes — used by catch-all replacement
    _ACCEL_INLINE = (
        "class PartialState:\n"
        "    _shared_state = {}\n"
        "    def __init__(self, *a, **kw):\n"
        "        self.process_index = 0\n"
        "        self.local_process_index = 0\n"
        "        self.num_processes = 1\n"
        "        self.is_main_process = True\n"
        "        self.is_local_main_process = True\n"
        "class Accelerator:\n"
        "    def __init__(self, *a, **kw):\n"
        "        self.process_index = 0\n"
        "        self.local_process_index = 0\n"
        "        self.num_processes = 1\n"
        "        self.is_main_process = True\n"
        "        self.is_local_main_process = True\n"
        "        self.device = 'cpu'\n"
        "    def wait_for_everyone(self): pass\n"
        "    def print(self, *a, **kw): pass\n"
        "    def gather(self, x): return x\n"
    )

    def _accel_replacement(m):
        """Build indented inline stubs for any 'from accelerate import ...' line."""
        indent = m.group(1)
        return "\n".join(indent + line for line in _ACCEL_INLINE.splitlines())

    # Patterns to find and their replacements
    replacements = [
        # from accelerate[.anything] import <anything> → inline both classes
        (re.compile(r'^(\s*)from\s+accelerate(?:\.\w+)*\s+import\s+.*$', re.MULTILINE),
         _accel_replacement),
        # import accelerate → inline both classes
        (re.compile(r'^(\s*)import\s+accelerate\b.*$', re.MULTILINE),
         _accel_replacement),
        # from mlflow import / import mlflow (but NOT mlflow stub itself)
        (re.compile(r'^(\s*)from\s+mlflow(?:\.\w+)*\s+import\s+.*$', re.MULTILINE),
         r'\1pass  # stubbed: mlflow not available'),
        (re.compile(r'^(\s*)import\s+mlflow\b.*$', re.MULTILINE),
         r'\1pass  # stubbed: mlflow not available'),
        # import sklearn / from sklearn import
        (re.compile(r'^(\s*)from\s+sklearn(?:\.\w+)*\s+import\s+.*$', re.MULTILINE),
         r'\1pass  # stubbed: sklearn not available'),
        (re.compile(r'^(\s*)import\s+sklearn\b.*$', re.MULTILINE),
         r'\1pass  # stubbed: sklearn not available'),
    ]

    patched_files = []
    for root, _dirs, files in os.walk(mlsimkit_root):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r") as f:
                    original = f.read()
            except Exception:
                continue

            # Skip files that don't reference these packages
            if "accelerate" not in original and "mlflow" not in original and "sklearn" not in original:
                continue

            content = original
            for pattern, repl in replacements:
                content = pattern.sub(repl, content)

            if content != original:
                with open(fpath, "w") as f:
                    f.write(content)
                rel = os.path.relpath(fpath, mlsimkit_root)
                patched_files.append(rel)

    if patched_files:
        print(f"[mlsimkit-bootstrap] Patched {len(patched_files)} files: {patched_files}")
    else:
        print("[mlsimkit-bootstrap] No files needed patching (already clean)")




def _bootstrap_mlsimkit():
    """Bootstrap MLSimKit for inference-only use.

    ALWAYS runs — even if mlsimkit is already importable from system
    site-packages. We must:
      1. Install training-only package stubs (accelerate, mlflow, etc.)
      2. Extract our tarball to /tmp (if not already extracted)
      3. Scan and patch ALL .py files that import training-only packages
      4. Write a CLI wrapper that hardcodes sys.path (no PYTHONPATH dependency)
      5. Force /tmp path to the FRONT of sys.path and PYTHONPATH

    Never crashes the agent — all errors are caught and logged.
    """
    try:
        # Step 0: Install stubs for training-only packages FIRST
        stub_base = _install_training_stubs()

        # Step 0b: Set pyvista env var to allow new attributes BEFORE any import
        os.environ["PYVISTA_ALLOW_EMPTY_MESH"] = "true"

        extract_dir = "/tmp/mlsimkit_bootstrap"
        sp_dir = os.path.join(extract_dir, "site-packages")

        # Step 1: Extract tarball if not already extracted
        if not os.path.isdir(os.path.join(sp_dir, "mlsimkit")):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(script_dir, "..", "mlsimkit_dist", "mlsimkit_package.tar.gz"),
                os.path.join(script_dir, "mlsimkit_dist", "mlsimkit_package.tar.gz"),
                os.path.join(os.getcwd(), "mlsimkit_dist", "mlsimkit_package.tar.gz"),
            ]
            tarball = None
            for c in candidates:
                if os.path.exists(c):
                    tarball = os.path.abspath(c)
                    break

            if not tarball:
                print(f"[mlsimkit-bootstrap] WARNING: tarball not found. Searched: {candidates}")
                return

            print(f"[mlsimkit-bootstrap] Extracting {tarball}...")
            import tarfile
            os.makedirs(extract_dir, exist_ok=True)
            with tarfile.open(tarball, "r:gz") as tf:
                # Vendored, trusted tarball; use the data filter to reject unsafe
                # members (absolute paths / path traversal) per CVE-2007-4559.
                tf.extractall(extract_dir, filter="data")

        # Step 2: Replace tracking.py entirely (it's all training code)
        tracking_path = os.path.join(sp_dir, "mlsimkit", "learn", "common", "tracking.py")
        if os.path.exists(tracking_path):
            with open(tracking_path, "w") as f:
                f.write(_TRACKING_STUB)
            print("[mlsimkit-bootstrap] Replaced tracking.py with stub")

        # Step 3: Scan ALL .py files and patch any accelerate/mlflow imports
        mlsimkit_root = os.path.join(sp_dir, "mlsimkit")
        if os.path.isdir(mlsimkit_root):
            _patch_all_training_imports(mlsimkit_root)

        # Step 3b: Patch pyvista attribute access in MLSimKit source files
        # Newer pyvista blocks setting attributes like vertex_normals on PolyData.
        # We patch the source to use object.__setattr__ instead of direct assignment.
        import re as _re
        _pyvista_patches = 0
        for _root, _dirs, _files in os.walk(mlsimkit_root):
            for _fname in _files:
                if not _fname.endswith(".py"):
                    continue
                _fpath = os.path.join(_root, _fname)
                try:
                    with open(_fpath, "r") as _f:
                        _content = _f.read()
                except Exception:
                    continue
                if "vertex_normals" not in _content and "point_normals" not in _content:
                    continue
                # Replace patterns like: mesh.vertex_normals = X  →  object.__setattr__(mesh, 'vertex_normals', X)
                _new = _re.sub(
                    r'(\b\w+)\.vertex_normals\s*=\s*(.+)',
                    r"object.__setattr__(\1, 'vertex_normals', \2)",
                    _content
                )
                _new = _re.sub(
                    r'(\b\w+)\.point_normals\s*=\s*(.+)',
                    r"object.__setattr__(\1, 'point_normals', \2)",
                    _new
                )
                if _new != _content:
                    with open(_fpath, "w") as _f:
                        _f.write(_new)
                    _pyvista_patches += 1
                    print(f"[mlsimkit-bootstrap] Patched pyvista attrs in {os.path.relpath(_fpath, mlsimkit_root)}")
        if _pyvista_patches:
            print(f"[mlsimkit-bootstrap] Patched {_pyvista_patches} files for pyvista compatibility")

        # Step 4: Nuke ALL __pycache__ dirs to prevent stale .pyc
        import shutil as _sh
        for root, dirs, _files in os.walk(mlsimkit_root):
            for d in dirs:
                if d == "__pycache__":
                    _sh.rmtree(os.path.join(root, d), ignore_errors=True)
        print("[mlsimkit-bootstrap] Cleared all __pycache__")

        # Step 5: Force our path to the FRONT of sys.path
        if sp_dir in sys.path:
            sys.path.remove(sp_dir)
        sys.path.insert(0, sp_dir)
        if stub_base in sys.path:
            sys.path.remove(stub_base)
        sys.path.insert(1, stub_base)

        # Step 6: PYTHONPATH for CLI: our patched mlsimkit first, then stubs
        pp_parts = [sp_dir, stub_base]
        existing_pp = os.environ.get("PYTHONPATH", "")
        for p in existing_pp.split(":"):
            if p and p not in pp_parts:
                pp_parts.append(p)
        os.environ["PYTHONPATH"] = ":".join(pp_parts)

        # Step 7: CLI wrapper — ALWAYS overwrite with a script that hardcodes
        # sys.path manipulation BEFORE any imports. This is the nuclear option
        # that makes the CLI work regardless of PYTHONPATH propagation.
        dst_bin = "/tmp/mlsimkit-learn"
        cli_wrapper = f'''#!{sys.executable}
# Auto-generated CLI wrapper — hardcodes sys.path for inference-only env
import sys, os
sys.path.insert(0, "{sp_dir}")
sys.path.insert(1, "{stub_base}")
os.environ["PYTHONPATH"] = "{sp_dir}:{stub_base}:" + os.environ.get("PYTHONPATH", "")

# Allow pyvista to set new attributes (MLSimKit surface preprocessing needs this)
try:
    import pyvista
    pyvista.allow_new_attributes(True)
except Exception:
    pass

# Force torch.load to use CPU (AgentCore has no GPU but models were saved on CUDA)
import torch
_orig_torch_load = torch.load
def _cpu_torch_load(*args, **kwargs):
    kwargs.setdefault("map_location", "cpu")
    return _orig_torch_load(*args, **kwargs)
torch.load = _cpu_torch_load

from mlsimkit.learn.cli import learn
sys.exit(learn())
'''
        with open(dst_bin, "w") as f:
            f.write(cli_wrapper)
        os.chmod(dst_bin, 0o755)
        os.environ["MLSIMKIT_LEARN_EXECUTABLE"] = dst_bin
        print(f"[mlsimkit-bootstrap] Wrote CLI wrapper: {dst_bin}")

        if "/tmp" not in os.environ.get("PATH", "").split(":"):
            os.environ["PATH"] = f"/tmp:{os.environ.get('PATH', '')}"

        # Step 8: Clear stale mlsimkit modules from this process
        stale = [k for k in sys.modules if k.startswith("mlsimkit")]
        for k in stale:
            del sys.modules[k]

        # Verify
        import importlib
        importlib.invalidate_caches()
        import mlsimkit  # noqa: F401
        print(f"[mlsimkit-bootstrap] OK — MLSimKit {getattr(mlsimkit, '__version__', 'unknown')}")
        print(f"[mlsimkit-bootstrap] PYTHONPATH={os.environ['PYTHONPATH']}")

    except Exception as e:
        print(f"[mlsimkit-bootstrap] WARN: bootstrap failed: {e}")
        import traceback
        traceback.print_exc()


_bootstrap_mlsimkit()

# ---------------------------------------------------------------------------

import json
import logging
import os
import time

import boto3
import uvicorn
from fastapi import FastAPI
from strands import Agent, tool
from strands.agent.agent import ConcurrentInvocationMode
from strands.hooks.events import BeforeInvocationEvent
from strands.models import BedrockModel
from strands.multiagent.a2a import A2AServer
from strands.agent.conversation_manager import SlidingWindowConversationManager

# A2A TaskStore — strip history from responses to prevent payload bloat
from a2a.server.tasks import TaskStore, InMemoryTaskStore
from a2a.server.context import ServerCallContext
from a2a.types import Task as A2ATask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable pyvista new attribute setting (needed by MLSimKit surface preprocessing)
try:
    import pyvista
    pyvista.allow_new_attributes(True)
    logger.info(f"Set pyvista.allow_new_attributes(True) — version {pyvista.__version__}")
except Exception as e:
    logger.warning(f"pyvista patch failed (non-fatal): {e}")

# Configuration
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AGENT_PORT = int(os.environ.get("AERO_AGENT_PORT", "9000"))
MODEL_ID = os.environ.get("MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
MAX_TOOL_RESULT_CHARS = 5000  # KPI + surface + slices results can be large; truncate to avoid context overflow
VARIANT_CACHE_TABLE = os.environ.get("VARIANT_CACHE_TABLE", "CarDesignVariantCache")
KPI_MODEL_S3_URI = os.environ.get("KPI_MODEL_S3_URI", "")
SURFACE_MODEL_S3_URI = os.environ.get("SURFACE_MODEL_S3_URI", "")
SLICES_AE_MODEL_S3_URI = os.environ.get("SLICES_AE_MODEL_S3_URI", "")
SLICES_MGN_MODEL_S3_URI = os.environ.get("SLICES_MGN_MODEL_S3_URI", "")
GEOMETRY_S3_BUCKET = os.environ.get("GEOMETRY_S3_BUCKET", "")
VISUALIZATION_S3_BUCKET = os.environ.get("VISUALIZATION_S3_BUCKET", GEOMETRY_S3_BUCKET)
PRESIGNED_URL_EXPIRY = 3600  # 1 hour


# ---------------------------------------------------------------------------
# URL conversion helper
# ---------------------------------------------------------------------------

def _normalize_geometry_path(geometry_path: str) -> str:
    """Convert presigned HTTPS URLs or malformed paths to s3:// URIs.

    The orchestrator sometimes sends presigned HTTPS URLs instead of s3:// URIs.
    This safety net converts them so inference can download the file properly.
    Also handles truncated URLs and extracts the s3 path from various URL formats.
    """
    import re

    # Already a valid s3:// URI
    if geometry_path.startswith("s3://") and geometry_path.endswith(".stl"):
        return geometry_path

    # Standard presigned URL: https://bucket.s3.region.amazonaws.com/key?params
    if "amazonaws.com" in geometry_path:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(geometry_path)
            hostname = parsed.hostname or ""

            # Format: bucket.s3.region.amazonaws.com
            if ".s3." in hostname:
                bucket = hostname.split(".s3.")[0]
                key = parsed.path.lstrip("/")
                # Strip query params that may be part of presigned URL
                if "?" in key:
                    key = key.split("?")[0]
                s3_uri = f"s3://{bucket}/{key}"
                logger.info(f"Converted presigned URL to s3:// URI: {s3_uri}")
                return s3_uri

            # Format: s3.region.amazonaws.com/bucket/key
            if hostname.startswith("s3."):
                path_parts = parsed.path.lstrip("/").split("/", 1)
                if len(path_parts) == 2:
                    bucket, key = path_parts
                    if "?" in key:
                        key = key.split("?")[0]
                    s3_uri = f"s3://{bucket}/{key}"
                    logger.info(f"Converted path-style URL to s3:// URI: {s3_uri}")
                    return s3_uri
        except Exception as e:
            logger.warning(f"Failed to parse presigned URL: {e}")

    # Try regex extraction as last resort (handles truncated/malformed URLs)
    s3_match = re.search(r's3://[a-zA-Z0-9._-]+/[^\s"\'?\]\)]+\.stl', geometry_path)
    if s3_match:
        s3_uri = s3_match.group(0)
        logger.info(f"Extracted s3:// URI via regex: {s3_uri}")
        return s3_uri

    # Try to extract bucket/key from a truncated presigned URL
    bucket_match = re.search(r'([a-zA-Z0-9._-]+)\.s3\.[a-zA-Z0-9.-]+\.amazonaws\.com/([^\s"\'?]+\.stl)', geometry_path)
    if bucket_match:
        bucket = bucket_match.group(1)
        key = bucket_match.group(2)
        s3_uri = f"s3://{bucket}/{key}"
        logger.info(f"Extracted s3:// URI from partial URL: {s3_uri}")
        return s3_uri

    logger.warning(f"Could not normalize geometry path: {geometry_path[:200]}")
    return geometry_path


# ---------------------------------------------------------------------------
# Visualization helpers — generate heatmap PNGs and upload to S3
# ---------------------------------------------------------------------------

def _upload_to_s3_and_presign(local_path: str, s3_key: str) -> str:
    """Upload a local file to S3 and return the s3:// URI.

    Returns the local file path if VISUALIZATION_S3_BUCKET is not configured.
    """
    if not VISUALIZATION_S3_BUCKET:
        logger.warning("VISUALIZATION_S3_BUCKET not configured — skipping S3 upload, returning local path")
        return local_path

    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.upload_file(local_path, VISUALIZATION_S3_BUCKET, s3_key,
                   ExtraArgs={"ContentType": "image/png"})
    s3_uri = f"s3://{VISUALIZATION_S3_BUCKET}/{s3_key}"
    logger.info(f"Uploaded {local_path} → {s3_uri}")
    return s3_uri


def _generate_surface_heatmap_png(cpavg, cfxavg, vertices, variant_id: str) -> list[str]:
    """Generate pressure and friction heatmap PNGs from surface data.

    Returns list of presigned URLs for the generated images.
    """
    import tempfile
    urls = []

    if not vertices or (not cpavg and not cfxavg):
        return urls

    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.tri import Triangulation

        verts = np.array(vertices)
        x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]

        for field_name, field_data, cmap_name, label in [
            ("cpavg", cpavg, "RdBu_r", "Pressure Coefficient (Cp avg)"),
            ("cfxavg", cfxavg, "hot", "Skin Friction (Cfx avg)"),
        ]:
            if not field_data or len(field_data) == 0:
                continue

            vals = np.array(field_data)

            # Adaptive point size: small for dense meshes, large for sparse parametric meshes
            n_pts = len(vals)
            pt_size = max(0.3, min(8.0, 5000 / n_pts))

            # Top-down view (X-Z plane, colored by field value)
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#1a1a2e")

            for ax, (proj_x, proj_y, view_label) in zip(axes, [
                (x, z, "Top View (X-Z)"),
                (x, y, "Side View (X-Y)"),
            ]):
                ax.set_facecolor("#1a1a2e")
                scatter = ax.scatter(proj_x, proj_y, c=vals, cmap=cmap_name,
                                     s=pt_size, alpha=0.9, edgecolors="none")
                ax.set_aspect("equal")
                ax.set_title(f"{view_label}", color="white", fontsize=13, fontweight="bold")
                ax.tick_params(colors="#888")
                for spine in ax.spines.values():
                    spine.set_color("#444")
                cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
                cbar.set_label(label, color="white", fontsize=10)
                cbar.ax.tick_params(colors="#888")

            fig.suptitle(f"{variant_id} — {label}", color="#FF9900",
                         fontsize=15, fontweight="bold", y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix=f"{field_name}_")
            tmp.close()  # close handle before savefig writes to the path
            fig.savefig(tmp.name, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor(), edgecolor="none")
            plt.close(fig)

            s3_key = f"visualizations/{variant_id}/{field_name}_heatmap.png"
            url = _upload_to_s3_and_presign(tmp.name, s3_key)
            urls.append(url)

            os.unlink(tmp.name)

    except ImportError as e:
        logger.warning(f"Cannot generate heatmap PNG (missing dependency): {e}")
    except Exception as e:
        logger.error(f"Heatmap generation failed: {e}", exc_info=True)

    return urls


def _upload_slices_images(combined_images: list[str], variant_id: str) -> list[str]:
    """Upload slices combined PNGs to S3 and return presigned URLs."""
    urls = []
    for i, img_path in enumerate(combined_images):
        if not os.path.exists(img_path):
            continue
        s3_key = f"visualizations/{variant_id}/slice_{i:02d}.png"
        try:
            url = _upload_to_s3_and_presign(img_path, s3_key)
            urls.append(url)
        except Exception as e:
            logger.error(f"Failed to upload slice image {img_path}: {e}")
    return urls


# ---------------------------------------------------------------------------
# DynamoDB variant cache lookup
# ---------------------------------------------------------------------------

def _lookup_variant_cache(variant_id: str) -> dict | None:
    """Look up pre-computed KPIs from DynamoDB variant cache."""
    try:
        dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
        table = dynamodb.Table(VARIANT_CACHE_TABLE)
        response = table.get_item(Key={"pk": "VARIANT", "sk": variant_id})
        item = response.get("Item")
        if item:
            logger.info(f"Cache hit for {variant_id}")
            return {
                "variant_id": variant_id,
                "drag_coefficient": float(item.get("cd", 0)),
                "side_force_coefficient": float(item.get("cs", 0)),
                "lift_coefficient": float(item.get("cl", 0)),
                "yaw_moment_coefficient": float(item.get("cmy", 0)),
                "source": "cache",
            }
        logger.info(f"Cache miss for {variant_id}")
        return None
    except Exception as e:
        logger.warning(f"DynamoDB cache lookup failed: {e}")
        return None


def _write_variant_cache(variant_id: str, kpis: dict) -> None:
    """Write KPI results back to DynamoDB cache for future lookups."""
    try:
        cd = round(kpis.get("drag_coefficient", 0), 6)
        cs = round(kpis.get("side_force_coefficient", 0), 6)
        cl = round(kpis.get("lift_coefficient", 0), 6)
        cmy = round(kpis.get("yaw_moment_coefficient", 0), 6)

        # Skip caching if all KPIs are zero — likely a failed inference
        if cd == 0 and cs == 0 and cl == 0 and cmy == 0:
            logger.warning(f"Skipping cache write for {variant_id} — all KPIs are zero")
            return

        dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
        table = dynamodb.Table(VARIANT_CACHE_TABLE)
        from decimal import Decimal
        table.put_item(Item={
            "pk": "VARIANT",
            "sk": variant_id,
            "cd": Decimal(str(cd)),
            "cs": Decimal(str(cs)),
            "cl": Decimal(str(cl)),
            "cmy": Decimal(str(cmy)),
        })
        logger.info(f"Wrote {variant_id} to cache")
    except Exception as e:
        logger.warning(f"Cache write failed for {variant_id}: {e}")


# ---------------------------------------------------------------------------
# MLSimKit inference (stubbed — wired in Task 2.4 / 16.3)
# ---------------------------------------------------------------------------

def _run_kpi_inference(variant_id: str, geometry_path: str) -> dict:
    """Run KPI inference using MLSimKit surrogate model.

    Delegates to backend/training/inference.py which wraps the MLSimKit
    Python API. Downloads model from S3 on first call, then caches locally.
    """
    try:
        from training.inference import predict_kpi

        return predict_kpi(
            variant_id=variant_id,
            geometry_path=geometry_path,
            model_s3_key=KPI_MODEL_S3_URI or "kpi/best_model.pt",
        )
    except Exception as e:
        logger.error(f"KPI inference failed for {variant_id}: {e}")
        return {
            "variant_id": variant_id,
            "status": "error",
            "error_message": f"KPI inference failed: {e}",
        }


def _run_surface_inference(variant_id: str, geometry_path: str) -> dict:
    """Run surface variable inference using MLSimKit surrogate model.

    Returns cpavg and cfxavg field data mapped onto mesh vertices.
    Delegates to backend/training/inference.py.
    """
    try:
        from training.inference import predict_surface_variables

        return predict_surface_variables(
            variant_id=variant_id,
            geometry_path=geometry_path,
            model_s3_key=SURFACE_MODEL_S3_URI or "surface/best_model.pt",
        )
    except Exception as e:
        logger.error(f"Surface inference failed for {variant_id}: {e}")
        return {
            "variant_id": variant_id,
            "status": "error",
            "error_message": f"Surface inference failed: {e}",
        }


def _run_slices_inference(variant_id: str, geometry_path: str) -> dict:
    """Run slices inference using MLSimKit surrogate model.

    Returns velocity flow field slice images (combined PNGs) and metrics.
    Delegates to backend/training/inference.py which uses both the
    autoencoder (AE) and MeshGraphNet (MGN) models.
    """
    try:
        from training.inference import predict_slices

        return predict_slices(
            variant_id=variant_id,
            geometry_path=geometry_path,
            ae_s3_key=SLICES_AE_MODEL_S3_URI or "slices/ae_best_model.pt",
            mgn_s3_key=SLICES_MGN_MODEL_S3_URI or "slices/mgn_last_model.pt",
        )
    except Exception as e:
        logger.error(f"Slices inference failed for {variant_id}: {e}")
        return {
            "variant_id": variant_id,
            "status": "error",
            "error_message": f"Slices inference failed: {e}",
        }


# ---------------------------------------------------------------------------
# Strands @tool functions
# ---------------------------------------------------------------------------

@tool
def evaluate_aero_kpi(variant_id: str, geometry_path: str = "") -> str:
    """Evaluate aerodynamic KPIs for a car body design variant.

    Uses two-tier data access: checks DynamoDB variant cache first,
    falls back to live MLSimKit inference on cache miss.

    Args:
        variant_id: Unique identifier for the design variant (e.g. "run_15").
        geometry_path: Path or S3 URI to the STL/VTP geometry file.
            Required for live inference on cache miss.

    Returns:
        JSON string with Cd, Cs, Cl, Cmy predictions and source indicator.
    """
    start = time.time()
    try:
        # Normalize presigned HTTPS URLs to s3:// URIs
        geometry_path = _normalize_geometry_path(geometry_path)

        # Tier 1: DynamoDB cache
        cached = _lookup_variant_cache(variant_id)
        if cached:
            cached["inference_time_ms"] = round((time.time() - start) * 1000, 1)
            cached["status"] = "success"
            return json.dumps(cached, indent=2)

        # Tier 2: Live inference
        if not geometry_path:
            return json.dumps({
                "variant_id": variant_id,
                "status": "error",
                "error_message": "Cache miss and no geometry_path provided for live inference",
            })

        result = _run_kpi_inference(variant_id, geometry_path)
        result["inference_time_ms"] = round((time.time() - start) * 1000, 1)

        # If inference returned an error, propagate it immediately
        if result.get("status") == "error":
            return json.dumps(result, indent=2)

        result["status"] = "success"

        # Write back to cache for:
        # - WindsorML variants (run_*): canonical dataset
        # - User-uploaded STLs (uploaded_*): real geometries worth persisting
        # Exclude parametric variants (parametric_*) — near-identical predictions
        # due to out-of-distribution mesh topology would pollute rankings.
        if variant_id.startswith("run_") or variant_id.startswith("uploaded_"):
            _write_variant_cache(variant_id, result)

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({
            "variant_id": variant_id,
            "status": "error",
            "error_message": str(e),
        })


@tool
def evaluate_aero_batch(variants_json: str) -> str:
    """Evaluate aerodynamic KPIs for a batch of design variants.

    Args:
        variants_json: JSON array of objects with variant_id and geometry_path.

    Returns:
        JSON array of KPI evaluation results.
    """
    try:
        variants = json.loads(variants_json)
        results = []
        for v in variants:
            result_str = evaluate_aero_kpi(
                variant_id=v["variant_id"],
                geometry_path=v.get("geometry_path", ""),
            )
            results.append(json.loads(result_str))
        return json.dumps(results, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error_message": str(e)})


@tool
def get_surface_data(variant_id: str, geometry_path: str) -> str:
    """Get surface pressure (cpavg) and friction (cfxavg) heatmap images.

    Runs live inference, generates heatmap PNG images, uploads to S3,
    and returns presigned URLs that can be displayed in the frontend.

    Args:
        variant_id: Unique identifier for the design variant.
        geometry_path: Path or S3 URI to the STL/VTP geometry file.

    Returns:
        JSON with image_urls (presigned S3 URLs for heatmap PNGs),
        cpavg/cfxavg summary stats, and inference time.
    """
    start = time.time()
    try:
        geometry_path = _normalize_geometry_path(geometry_path)
        result = _run_surface_inference(variant_id, geometry_path)
        result["inference_time_ms"] = round((time.time() - start) * 1000, 1)

        # Propagate errors and geometry-not-found immediately
        if result.get("status") in ("error", "geometry_not_found"):
            return json.dumps(result, indent=2)

        # Generate heatmap PNGs and upload to S3
        image_urls = _generate_surface_heatmap_png(
            result.get("cpavg_field", []),
            result.get("cfxavg_field", []),
            result.get("mesh_vertices", []),
            variant_id,
        )
        result["image_urls"] = image_urls

        # Remove raw field data from response to avoid payload overflow
        cpavg = result.pop("cpavg_field", [])
        cfxavg = result.pop("cfxavg_field", [])
        result.pop("mesh_vertices", None)
        result.pop("mesh_faces", None)
        result.pop("vtk_file_path", None)
        result.pop("png_heatmap_path", None)

        # Add summary stats instead
        if cpavg:
            import numpy as np
            arr = np.array(cpavg)
            result["cpavg_stats"] = {"min": float(arr.min()), "max": float(arr.max()), "mean": float(arr.mean())}
        if cfxavg:
            import numpy as np
            arr = np.array(cfxavg)
            result["cfxavg_stats"] = {"min": float(arr.min()), "max": float(arr.max()), "mean": float(arr.mean())}

        result["status"] = "success"

        # Return compact response with S3 URIs (not presigned URLs)
        # The orchestrator will generate presigned URLs using generate_image_viewer_tag
        parts = []
        if cpavg:
            import numpy as np
            arr = np.array(cpavg)
            parts.append(f"Surface data for {variant_id}: cpavg range [{arr.min():.3f}, {arr.max():.3f}], mean {arr.mean():.3f}")
        if cfxavg:
            import numpy as np
            arr = np.array(cfxavg)
            parts.append(f"cfxavg range [{arr.min():.4f}, {arr.max():.4f}], mean {arr.mean():.4f}")
        parts.append(f"Inference time: {result['inference_time_ms']:.0f}ms")

        # Return S3 URIs for the heatmap images (already s3:// URIs from _upload_to_s3_and_presign)
        if image_urls:
            parts.append(f"image_s3_uris: {', '.join(image_urls)}")
        return "\n".join(parts)
    except Exception as e:
        return json.dumps({
            "variant_id": variant_id,
            "status": "error",
            "error_message": str(e),
        })


@tool
def list_cached_variants(limit: int = 10) -> str:
    """List variant IDs available in the DynamoDB cache.

    Args:
        limit: Maximum number of variants to return (max 20).

    Returns:
        JSON array of cached variant IDs with their Cd values.
    """
    try:
        limit = min(limit, 20)  # Hard cap to prevent payload overflow
        dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
        table = dynamodb.Table(VARIANT_CACHE_TABLE)
        response = table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("pk").eq("VARIANT"),
            Limit=limit,
            ProjectionExpression="sk, cd",
        )
        items = [
            {"variant_id": item["sk"], "cd": float(item.get("cd", 0))}
            for item in response.get("Items", [])
        ]
        return json.dumps({"variants": items, "count": len(items)}, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error_message": str(e)})


@tool
def query_variants(sort_by: str = "cd", limit: int = 5, ascending: bool = True,
                   max_cd: float = 0, min_cd: float = 0) -> str:
    """Query and rank all cached variants by a specific aero metric.

    Scans the full DynamoDB variant cache, sorts by the requested metric,
    applies optional filters, and returns the top N results. Use this for
    queries like "top 5 by drag", "lowest Cd variants", "compare best variants".

    This is more powerful than list_cached_variants because it sorts across
    ALL cached variants, not just the first N by ID.

    Args:
        sort_by: Metric to sort by — "cd", "cs", "cl", "cmy". Default "cd".
        limit: Number of results to return (max 20). Default 5.
        ascending: Sort ascending (True = lowest first) or descending. Default True.
        max_cd: If > 0, filter to variants with Cd <= this value.
        min_cd: If > 0, filter to variants with Cd >= this value.

    Returns:
        JSON with sorted variants array, count returned, and total cached.
    """
    try:
        limit = min(limit, 20)
        dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
        table = dynamodb.Table(VARIANT_CACHE_TABLE)

        # Scan all variants
        all_items = []
        scan_kwargs = {
            "FilterExpression": boto3.dynamodb.conditions.Key("pk").eq("VARIANT"),
            "ProjectionExpression": "sk, cd, cs, cl, cmy",
        }
        while True:
            response = table.scan(**scan_kwargs)
            all_items.extend(response.get("Items", []))
            if "LastEvaluatedKey" not in response:
                break
            scan_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]

        field_map = {"cd": "cd", "cs": "cs", "cl": "cl", "cmy": "cmy"}
        sort_field = field_map.get(sort_by, "cd")

        variants = []
        for item in all_items:
            v = {
                "variant_id": item.get("sk", ""),
                "cd": float(item.get("cd", 0)),
                "cs": float(item.get("cs", 0)),
                "cl": float(item.get("cl", 0)),
                "cmy": float(item.get("cmy", 0)),
            }
            if max_cd > 0 and v["cd"] > max_cd:
                continue
            if min_cd > 0 and v["cd"] < min_cd:
                continue
            variants.append(v)

        variants.sort(key=lambda v: v.get(sort_field, 0), reverse=not ascending)
        top_n = variants[:limit]

        return json.dumps({
            "variants": top_n,
            "returned": len(top_n),
            "total_matching": len(variants),
            "total_cached": len(all_items),
            "sort_by": sort_by,
            "ascending": ascending,
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error_message": str(e)})


@tool
def get_slices_data(variant_id: str, geometry_path: str) -> str:
    """Get velocity flow field slice images for a car body design variant.

    Runs live inference using the MLSimKit slices model (autoencoder + MGN).
    Produces cross-section images showing airflow velocity around the car body.
    Uploads images to S3 and returns presigned URLs for frontend display.

    Args:
        variant_id: Unique identifier for the design variant.
        geometry_path: Path or S3 URI to the STL/VTP geometry file.

    Returns:
        JSON with image_urls (presigned S3 URLs for slice PNGs),
        metrics, inference_time_ms, and source.
    """
    start = time.time()
    try:
        geometry_path = _normalize_geometry_path(geometry_path)
        result = _run_slices_inference(variant_id, geometry_path)
        result["inference_time_ms"] = round((time.time() - start) * 1000, 1)

        # Propagate errors and geometry-not-found immediately
        if result.get("status") in ("error", "geometry_not_found"):
            return json.dumps(result, indent=2)

        combined_images = result.get("combined_images", [])
        image_urls = _upload_slices_images(combined_images, variant_id)
        result["image_urls"] = image_urls

        # Remove local file paths from response (not useful to frontend)
        result.pop("combined_images", None)
        result.pop("npy_predictions", None)
        result.pop("images_dir", None)

        result["status"] = "success"

        # Return [IMAGE] tags directly — keep response compact
        parts = [f"Slices data for {variant_id}: {len(image_urls)} images generated"]
        if result.get("metrics"):
            parts.append(f"Metrics: {json.dumps(result['metrics'])}")
        parts.append(f"Inference time: {result['inference_time_ms']:.0f}ms")
        if image_urls:
            parts.append(f"image_s3_uris: {', '.join(image_urls)}")
        return "\n".join(parts)
    except Exception as e:
        return json.dumps({
            "variant_id": variant_id,
            "status": "error",
            "error_message": str(e),
        })


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the Aero Agent for the Car Design Space Explorer.

You evaluate aerodynamic KPIs (Cd, Cs, Cl, Cmy) and surface data for car body variants using MLSimKit surrogate models.

## Tools
1. **evaluate_aero_kpi** — Single variant KPIs. Checks DynamoDB cache first, falls back to live inference.
2. **evaluate_aero_batch** — Batch KPI evaluation.
3. **get_surface_data** — Surface cpavg/cfxavg heatmap PNGs (always live inference). Returns presigned URLs.
4. **list_cached_variants** — List cached variants (limit 20).
5. **query_variants** — Rank ALL cached variants by any metric. Use for "top N by Cd" queries.
6. **get_slices_data** — Velocity flow field slice PNGs (always live inference).

## Response Rules
- Keep total response under 1500 chars. Verbose responses crash the pipeline.
- Return tool output directly for single variants. For lists >5 items, sort and return top 5 only.
- No prose or markdown formatting — just structured data.
- When get_surface_data or get_slices_data returns image_urls, include them as [IMAGE]url[/IMAGE] tags.
- Never simulate responses — always use actual tools.
"""

# ---------------------------------------------------------------------------
# Conversation manager — truncates large tool results before Bedrock API calls
# ---------------------------------------------------------------------------
class TruncatingConversationManager(SlidingWindowConversationManager):
    """Sliding window manager that truncates large tool results.

    KPI + surface + slices tool chains can return megabytes of data.
    Truncating keeps each conversation turn within Bedrock's token limit.
    """

    def apply_management(self, agent):
        for msg in agent.messages:
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or not block.get("toolResult"):
                    continue
                tr = block["toolResult"]
                tr_content = tr.get("content", [])
                total_len = sum(
                    len(p.get("text", "")) for p in tr_content
                    if isinstance(p, dict)
                )
                if total_len > MAX_TOOL_RESULT_CHARS:
                    truncated_parts = []
                    remaining = MAX_TOOL_RESULT_CHARS
                    for part in tr_content:
                        if isinstance(part, dict) and "text" in part:
                            if remaining > 0:
                                truncated_parts.append({"text": part["text"][:remaining]})
                                remaining -= len(part["text"])
                        else:
                            truncated_parts.append(part)
                    truncated_parts.append({
                        "text": f"\n[TRUNCATED from {total_len} chars. Use the data you have.]"
                    })
                    tr["content"] = truncated_parts
                    logger.info(f"Truncated tool result from {total_len} to ~{MAX_TOOL_RESULT_CHARS} chars")
        super().apply_management(agent)


# ---------------------------------------------------------------------------
# Agent + A2A Server setup
# ---------------------------------------------------------------------------
logger.info("Creating Aero Agent...")

aero_model = BedrockModel(
    model_id=MODEL_ID,
    max_tokens=2048,
)

agent = Agent(
    name="Aero Agent",
    description="Evaluates aerodynamic KPIs (Cd, Cs, Cl, Cmy) and surface pressure/friction distributions for car body design variants using MLSimKit surrogate models",
    system_prompt=SYSTEM_PROMPT,
    model=aero_model,
    tools=[evaluate_aero_kpi, evaluate_aero_batch, get_surface_data, get_slices_data, list_cached_variants, query_variants],
    conversation_manager=TruncatingConversationManager(window_size=6, per_turn=True),
    concurrent_invocation_mode=ConcurrentInvocationMode.UNSAFE_REENTRANT,
)

# Clear conversation history before each A2A request so the agent is stateless
# across requests. Without this, messages accumulate and context grows unbounded.
def _clear_messages(event: BeforeInvocationEvent):
    event.agent.messages.clear()

agent.add_hook(_clear_messages)
logger.info("✅ Aero Agent created")

# OTel guard — patch from_converse to handle missing 'output' key
# MaxTokensReachedException or throttling returns a response without 'output',
# causing OTel instrumentation to crash with KeyError: 'output'
try:
    from opentelemetry.instrumentation.botocore.extensions import bedrock_utils
    _original_from_converse = bedrock_utils._Choice.from_converse

    @classmethod
    def _safe_from_converse(cls, response, capture_content=False):
        try:
            return _original_from_converse.__func__(cls, response, capture_content)
        except (KeyError, TypeError, Exception) as e:
            logger.warning(f"[otel_guard] ConverseStream response issue: {e} — returning empty choice")
            try:
                return cls(finish_reason="error", message=None, index=0)
            except TypeError:
                return cls(finish_reason="error", message=None)

    bedrock_utils._Choice.from_converse = _safe_from_converse
    logger.info("✅ OTel from_converse patched for KeyError guard")
except Exception as e:
    logger.warning(f"OTel patch skipped: {e}")

runtime_url = os.environ.get("AGENTCORE_RUNTIME_URL", f"http://127.0.0.1:{AGENT_PORT}/")
logger.info(f"Runtime URL: {runtime_url}")


# ---------------------------------------------------------------------------
# HistoryStrippingTaskStore — prevents A2A response payload bloat
# ---------------------------------------------------------------------------
class HistoryStrippingTaskStore(TaskStore):
    """TaskStore that strips conversation history from tasks on retrieval."""

    def __init__(self) -> None:
        self._inner = InMemoryTaskStore()

    async def save(self, task: A2ATask, context: ServerCallContext | None = None) -> None:
        await self._inner.save(task, context)

    async def get(self, task_id: str, context: ServerCallContext | None = None) -> A2ATask | None:
        task = await self._inner.get(task_id, context)
        if task is not None:
            task.history = None
        return task

    async def delete(self, task_id: str, context: ServerCallContext | None = None) -> None:
        await self._inner.delete(task_id, context)


a2a_server = A2AServer(
    agent=agent,
    http_url=runtime_url,
    serve_at_root=True,
    task_store=HistoryStrippingTaskStore(),
    enable_a2a_compliant_streaming=False,  # Synchronous — one complete payload
)

# FastAPI app
app = FastAPI()


@app.get("/ping")
def ping():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agent": "aero_agent",
        "version": "1.0.0",
        "features": [
            "a2a_protocol",
            "kpi_prediction",
            "surface_variable_prediction",
            "slices_flow_field_prediction",
            "dynamodb_variant_cache",
            "two_tier_data_access",
        ],
        "models": {
            "kpi": KPI_MODEL_S3_URI or "not configured",
            "surface": SURFACE_MODEL_S3_URI or "not configured",
            "slices_ae": SLICES_AE_MODEL_S3_URI or "not configured",
            "slices_mgn": SLICES_MGN_MODEL_S3_URI or "not configured",
        },
        "cache_table": VARIANT_CACHE_TABLE,
    }


_a2a_app = a2a_server.to_fastapi_app()


# ---------------------------------------------------------------------------
# ASGI middleware: wrap non-JSON-RPC payloads arriving at POST /
# ---------------------------------------------------------------------------
import uuid as _uuid
from starlette.types import ASGIApp, Receive, Scope, Send


MAX_RESPONSE_BYTES = 60_000  # Synchronous A2A — safe to allow larger payloads


def _truncate_a2a_response(body: bytes) -> bytes:
    """Truncate an A2A JSON-RPC response if it exceeds MAX_RESPONSE_BYTES."""
    if len(body) <= MAX_RESPONSE_BYTES:
        return body
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, Exception):
        return body[:MAX_RESPONSE_BYTES]

    result = data.get("result", {})
    if not isinstance(result, dict):
        return json.dumps(data).encode()[:MAX_RESPONSE_BYTES]

    # 1. ALWAYS strip history unconditionally
    result.pop("history", None)
    task_obj = result.get("task", result) if "task" in result else result
    task_obj.pop("history", None)

    # 2. Merge ALL artifacts' text into one, collapse to single artifact
    artifacts = task_obj.get("artifacts", result.get("artifacts", []))
    if artifacts:
        all_text = ""
        for artifact in artifacts:
            for part in artifact.get("parts", []):
                if isinstance(part, dict) and part.get("text"):
                    all_text += part["text"]
                elif isinstance(part, dict) and part.get("kind") == "text" and "text" in part:
                    all_text += part["text"]
        trimmed = all_text.strip()[:3000]
        collapsed = [{
            "artifactId": artifacts[0].get("artifactId", ""),
            "name": artifacts[0].get("name", "agent_response"),
            "parts": [{"kind": "text", "text": trimmed}],
        }]
        if "artifacts" in task_obj:
            task_obj["artifacts"] = collapsed
        elif "artifacts" in result:
            result["artifacts"] = collapsed

    # 3. Nuke status.message if artifacts exist
    status = task_obj.get("status", result.get("status", {}))
    if isinstance(status, dict) and artifacts:
        status.pop("message", None)

    truncated = json.dumps(data, separators=(",", ":")).encode()

    # 4. Nuclear: raw byte truncation
    if len(truncated) > MAX_RESPONSE_BYTES:
        truncated = truncated[:MAX_RESPONSE_BYTES]

    logger.info(f"Truncated A2A response from {len(body)} to {len(truncated)} bytes")
    return truncated


class A2APayloadNormalizer:
    """ASGI middleware that normalizes A2A payloads in both directions.

    Inbound: wraps raw payloads into A2A JSON-RPC envelopes if needed.
    Outbound: truncates oversized response payloads to prevent -32603 errors.
    """

    def __init__(self, wrapped_app: ASGIApp):
        self.app = wrapped_app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http" or scope.get("method", "") != "POST":
            await self.app(scope, receive, send)
            return

        # --- INBOUND ---
        body_parts = []
        while True:
            message = await receive()
            body_parts.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        raw_body = b"".join(body_parts)

        needs_wrap = False
        try:
            parsed = json.loads(raw_body)
            if (isinstance(parsed, dict)
                    and parsed.get("jsonrpc") == "2.0"
                    and "method" in parsed):
                msg = parsed.get("params", {}).get("message", {})
                if msg and "messageId" not in msg:
                    msg["messageId"] = str(_uuid.uuid4())
                    raw_body = json.dumps(parsed).encode()
            else:
                needs_wrap = True
        except (json.JSONDecodeError, Exception):
            needs_wrap = True

        if needs_wrap:
            try:
                parsed = json.loads(raw_body)
                if isinstance(parsed, dict):
                    prompt = parsed.get("prompt", parsed.get("input", {}).get("prompt", ""))
                    if not prompt:
                        prompt = json.dumps(parsed)
                else:
                    prompt = str(parsed)
            except Exception:
                prompt = raw_body.decode("utf-8", errors="replace")

            a2a_envelope = {
                "jsonrpc": "2.0",
                "id": str(_uuid.uuid4()),
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": prompt}],
                        "messageId": str(_uuid.uuid4()),
                    }
                },
            }
            raw_body = json.dumps(a2a_envelope).encode()
            logger.info(f"Wrapped raw payload into A2A JSON-RPC envelope")

        body_sent = False

        async def wrapped_receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": raw_body, "more_body": False}
            return {"type": "http.disconnect"}

        # --- OUTBOUND: intercept and truncate ---
        response_headers_sent = False
        response_body_parts = []

        async def capturing_send(message):
            nonlocal response_headers_sent
            if message["type"] == "http.response.start":
                response_headers_sent = message
            elif message["type"] == "http.response.body":
                response_body_parts.append(message.get("body", b""))
                if not message.get("more_body", False):
                    full_body = b"".join(response_body_parts)
                    truncated_body = _truncate_a2a_response(full_body)
                    if response_headers_sent:
                        headers = list(response_headers_sent.get("headers", []))
                        new_headers = []
                        for h_name, h_val in headers:
                            if h_name.lower() == b"content-length":
                                new_headers.append((h_name, str(len(truncated_body)).encode()))
                            else:
                                new_headers.append((h_name, h_val))
                        response_headers_sent["headers"] = new_headers
                        await send(response_headers_sent)
                    await send({
                        "type": "http.response.body",
                        "body": truncated_body,
                        "more_body": False,
                    })
            else:
                await send(message)

        await self.app(scope, wrapped_receive, capturing_send)


_wrapped_a2a_app = A2APayloadNormalizer(_a2a_app)
app.mount("/", _wrapped_a2a_app)


# ---------------------------------------------------------------------------
# /invocations route — AgentCore Runtime forwards payloads here
# ---------------------------------------------------------------------------
from fastapi import Request
from fastapi.responses import JSONResponse
import httpx


@app.post("/invocations")
async def invocations(request: Request):
    """Handle /invocations calls (belt-and-suspenders fallback).

    For A2A protocol agents, AgentCore sends JSON-RPC to POST / directly.
    This route exists as a safety net. It handles two payload formats:
    1. Already valid A2A JSON-RPC → pass through to the A2A app
    2. Simple {"prompt": "..."} → wrap into JSON-RPC and forward
    """
    try:
        body = await request.json()
    except Exception:
        raw = await request.body()
        try:
            body = json.loads(raw)
        except Exception:
            body = {"prompt": raw.decode("utf-8", errors="replace")}

    import uuid as _uuid
    if (isinstance(body, dict)
            and body.get("jsonrpc") == "2.0"
            and body.get("method") in ("message/send", "message/stream")):
        a2a_payload = body
        msg = a2a_payload.get("params", {}).get("message", {})
        if "messageId" not in msg:
            msg["messageId"] = str(_uuid.uuid4())
    else:
        prompt = body.get("prompt", body.get("input", {}).get("prompt", "")) if isinstance(body, dict) else str(body)
        if not prompt and isinstance(body, dict):
            prompt = json.dumps(body)

        a2a_payload = {
            "jsonrpc": "2.0",
            "id": str(_uuid.uuid4()),
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": prompt}],
                    "messageId": str(_uuid.uuid4()),
                }
            },
        }

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_a2a_app), base_url="http://internal"
    ) as client:
        headers = {}
        for k, v in request.headers.items():
            if k.lower() not in ("host", "content-length", "content-type", "transfer-encoding"):
                headers[k] = v
        headers["content-type"] = "application/json"

        resp = await client.post("/", json=a2a_payload, headers=headers, timeout=600.0)

    return JSONResponse(content=resp.json(), status_code=resp.status_code)


if __name__ == "__main__":
    # AgentCore containers require an all-interface bind; ingress is protected
    # by the Runtime JWT authorizer and AgentCore network boundary.
    host, port = "0.0.0.0", AGENT_PORT  # nosec B104
    print()
    print("=" * 60)
    print("Aero Agent — Car Design Space Explorer")
    print(f"  Agent Card: http://{host}:{port}/.well-known/agent-card.json")
    print("=" * 60)
    uvicorn.run(app, host=host, port=port)
