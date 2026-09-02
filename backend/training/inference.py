#!/usr/bin/env python3
"""
MLSimKit Inference Wrapper for the Aero Agent.

Provides Python API functions that wrap MLSimKit's inference modules
for KPI prediction (Cd, Cs, Cl, Cmy) and surface variable prediction
(cpavg, cfxavg). Handles model loading from S3, temporary manifest
creation, and result parsing into Pydantic data models.

Usage from Aero Agent:
    from backend.training.inference import predict_kpi, predict_surface_variables

    kpi = predict_kpi("run_237", "/tmp/windsor_237.stl", "/models/kpi/best_model.pt")
    surface = predict_surface_variables("run_237", "/tmp/windsor_237.stl", "/models/surface/best_model.pt")
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

import boto3

logger = logging.getLogger(__name__)

# Local cache directory for downloaded models
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/tmp/mlsimkit_models")
S3_MODEL_BUCKET = os.environ.get(
    "S3_MODEL_BUCKET", ""
)


def _resolve_mlsimkit_executable() -> str:
    """Resolve MLSimKit to an absolute executable path before invocation."""
    configured = os.environ.get("MLSIMKIT_LEARN_EXECUTABLE", "")
    candidate = configured or shutil.which("mlsimkit-learn")
    if not candidate:
        raise FileNotFoundError(
            "mlsimkit-learn was not found; set MLSIMKIT_LEARN_EXECUTABLE"
        )

    path = Path(candidate).expanduser()
    if not path.is_absolute():
        raise ValueError("MLSimKit executable path must be absolute")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or (os.name != "nt" and not os.access(resolved, os.X_OK)):
        raise PermissionError(f"MLSimKit executable is not runnable: {resolved}")
    return str(resolved)


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _s3_object_exists(bucket: str, key: str) -> bool:
    """Return whether an exact S3 object exists.

    ``head_object`` is authorized by the existing object-level ``s3:GetObject``
    grant, avoiding bucket-wide ``s3:ListBucket`` permission.
    """
    try:
        boto3.client("s3").head_object(Bucket=bucket, Key=key)
        return True
    except Exception as e:
        logger.warning(f"S3 existence check error for s3://{bucket}/{key}: {e}")
        return False


# ---------------------------------------------------------------------------
# S3 model management
# ---------------------------------------------------------------------------

def _ensure_model_local(model_s3_key: str) -> str:
    """Download model from S3 if not already cached locally.

    Falls back to bundled weights in backend/models/weights/ when S3 is not configured.
    Returns local file path to the model checkpoint.
    """
    # Check local cache first
    local_path = os.path.join(MODEL_CACHE_DIR, model_s3_key)
    if os.path.exists(local_path):
        return local_path

    # Check bundled weights (e.g. backend/models/weights/kpi/best_model.pt)
    bundled_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "weights", model_s3_key)
    if os.path.exists(bundled_path):
        logger.info(f"Using bundled model weights: {bundled_path}")
        return bundled_path

    # Try S3 download if bucket is configured
    if not S3_MODEL_BUCKET:
        raise FileNotFoundError(
            f"S3_MODEL_BUCKET is not configured and no local model found at "
            f"{local_path} or {bundled_path}. Set the S3_MODEL_BUCKET environment "
            f"variable or ensure model weights are available locally."
        )

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    logger.info(f"Downloading model s3://{S3_MODEL_BUCKET}/{model_s3_key} -> {local_path}")
    s3 = boto3.client("s3")
    s3.download_file(S3_MODEL_BUCKET, model_s3_key, local_path)
    return local_path



def _ensure_geometry_local(geometry_path: str) -> str:
    """Ensure geometry file is available locally.

    If geometry_path is an S3 URI (s3://bucket/key), download it.
    If it's a presigned HTTPS URL for our S3 bucket, convert to s3:// and download.
    If it's already a local path, return as-is.
    """
    # Convert presigned HTTPS URLs to s3:// URIs
    # Pattern: https://<bucket>.s3.amazonaws.com/<key>?... or https://<bucket>.s3.<region>.amazonaws.com/<key>?...
    if geometry_path.startswith("https://") and ".s3." in geometry_path and "amazonaws.com" in geometry_path:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(geometry_path)
            # Extract bucket from hostname: <bucket>.s3.amazonaws.com or <bucket>.s3.<region>.amazonaws.com
            host_parts = parsed.hostname.split(".s3.")
            bucket = host_parts[0]
            # Key is the path without leading /
            key = parsed.path.lstrip("/")
            geometry_path = f"s3://{bucket}/{key}"
            logger.info(f"Converted presigned URL to s3:// URI: {geometry_path}")
        except Exception as e:
            logger.warning(f"Failed to convert presigned URL: {e}")

    if geometry_path.startswith("s3://"):
        parts = geometry_path.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        local_path = os.path.join(tempfile.gettempdir(), "geometries", os.path.basename(key))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if not os.path.exists(local_path):
            s3 = boto3.client("s3")
            s3.download_file(bucket, key, local_path)
        return local_path
    return geometry_path


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _create_single_manifest(geometry_path: str, include_surface: bool = False) -> str:
    """Create a temporary single-entry manifest for one geometry file.

    MLSimKit inference requires a manifest file (JSON Lines) pointing to
    the geometry. For runtime single-variant inference, we create a
    throwaway one-line manifest.

    Args:
        geometry_path: Absolute path to the STL geometry file.
        include_surface: If True, include surface variable fields in manifest.

    Returns:
        Path to the temporary manifest file.
    """
    manifest_fd, manifest_path = tempfile.mkstemp(suffix=".manifest")
    with os.fdopen(manifest_fd, "w") as f:
        # MLSimKit expects file:// URIs for local paths
        uri = geometry_path if geometry_path.startswith("file://") else f"file://{geometry_path}"
        entry = {"geometry_files": [uri]}
        f.write(json.dumps(entry) + "\n")
    return manifest_path


# ---------------------------------------------------------------------------
# KPI Inference
# ---------------------------------------------------------------------------

def predict_kpi(
    variant_id: str,
    geometry_path: str,
    model_path: str | None = None,
    model_s3_key: str = "kpi/best_model.pt",
) -> dict:
    """Run KPI inference on a car body geometry file.

    Uses MLSimKit's Python API to load the trained KPI surrogate model
    and predict Cd, Cs, Cl, Cmy for the given geometry.

    Args:
        variant_id: Identifier for the design variant (e.g., "run_237").
        geometry_path: Path to STL geometry file (local or s3:// URI).
        model_path: Direct local path to model checkpoint. If None,
                     downloads from S3 using model_s3_key.
        model_s3_key: S3 key for the KPI model within S3_MODEL_BUCKET.

    Returns:
        Dict with keys: variant_id, drag_coefficient, side_force_coefficient,
        lift_coefficient, yaw_moment_coefficient, inference_time_ms, source.
    """
    start = time.time()

    try:
        # Resolve model and geometry paths
        if model_path is None:
            model_path = _ensure_model_local(model_s3_key)
        geometry_local = _ensure_geometry_local(geometry_path)

        # Create single-entry manifest
        manifest_path = _create_single_manifest(geometry_local)

        # Create a temp output directory for predictions
        output_dir = tempfile.mkdtemp(prefix="kpi_predict_")

        try:
            # Use MLSimKit Python API
            from mlsimkit.learn.kpi.inference import run_predict as kpi_run_predict
            from mlsimkit.learn.kpi.schema.inference import InferenceSettings

            settings = InferenceSettings(
                model_path=model_path,
                manifest_path=manifest_path,
                output_dir=output_dir,
            )
            kpi_run_predict(settings, compare_groundtruth=False)

            # Parse prediction results CSV
            results_csv = os.path.join(output_dir, "predictions", "prediction_results.csv")
            kpis = _parse_kpi_results(results_csv, variant_id)

        except Exception as api_err:
            logger.warning(f"MLSimKit Python API failed: {api_err} — falling back to CLI")
            kpis = _predict_kpi_cli(variant_id, geometry_local, model_path, manifest_path, output_dir)
        finally:
            # Clean up temp manifest
            _safe_remove(manifest_path)

        elapsed_ms = (time.time() - start) * 1000
        kpis["inference_time_ms"] = round(elapsed_ms, 1)
        kpis["source"] = "mlsimkit"
        return kpis

    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        logger.error(f"KPI inference failed for {variant_id}: {e}")
        return {
            "variant_id": variant_id,
            "status": "error",
            "error_message": f"KPI inference failed: {e}",
            "inference_time_ms": round(elapsed_ms, 1),
        }


def _parse_kpi_results(csv_path: str, variant_id: str) -> dict:
    """Parse MLSimKit KPI prediction CSV into our data model format.

    MLSimKit outputs one of two CSV formats:
    1. Column-per-KPI: columns like cd, cs, cl, cmy (one row)
    2. Row-per-KPI: columns kpi_index, prediction (4 rows, index 0=cd, 1=cs, 2=cl, 3=cmy)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"KPI results not found at {csv_path}")

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No prediction rows found in {csv_path}")

    logger.info(f"KPI CSV columns: {list(rows[0].keys())}")
    for row in rows:
        logger.info(f"KPI CSV row: {dict(row)}")

    # Format 2: Row-per-KPI (kpi_index + prediction columns)
    if "kpi_index" in rows[0] and "prediction" in rows[0]:
        # Map kpi_index to coefficient names
        kpi_map = {0: "drag_coefficient", 1: "side_force_coefficient",
                    2: "lift_coefficient", 3: "yaw_moment_coefficient"}
        result = {"variant_id": variant_id,
                  "drag_coefficient": 0.0, "side_force_coefficient": 0.0,
                  "lift_coefficient": 0.0, "yaw_moment_coefficient": 0.0}
        for row in rows:
            idx = int(row["kpi_index"])
            val = float(row["prediction"])
            if idx in kpi_map:
                result[kpi_map[idx]] = val
        logger.info(f"Parsed KPIs (row-per-KPI): cd={result['drag_coefficient']}, "
                     f"cs={result['side_force_coefficient']}, cl={result['lift_coefficient']}, "
                     f"cmy={result['yaw_moment_coefficient']}")
        return result

    # Format 1: Column-per-KPI (single row with cd/cs/cl/cmy columns)
    row = rows[0]
    cd = float(row.get("cd", row.get("Cd", row.get("drag_coefficient", 0))))
    cs = float(row.get("cs", row.get("Cs", row.get("side_force_coefficient", 0))))
    cl = float(row.get("cl", row.get("Cl", row.get("lift_coefficient", 0))))
    cmy = float(row.get("cmy", row.get("Cmy", row.get("yaw_moment_coefficient", 0))))

    # Guard: if all KPIs are zero, the CSV format was likely unrecognized
    if cd == 0 and cs == 0 and cl == 0 and cmy == 0:
        raise ValueError(
            f"All KPIs are zero for {variant_id} — CSV format may be unrecognized. "
            f"Columns: {list(row.keys())}"
        )

    logger.info(f"Parsed KPIs (column-per-KPI): cd={cd}, cs={cs}, cl={cl}, cmy={cmy}")
    return {
        "variant_id": variant_id,
        "drag_coefficient": cd,
        "side_force_coefficient": cs,
        "lift_coefficient": cl,
        "yaw_moment_coefficient": cmy,
    }



def _predict_kpi_cli(
    variant_id: str,
    geometry_path: str,
    model_path: str,
    manifest_path: str,
    output_dir: str,
) -> dict:
    """Fallback: run KPI inference via MLSimKit CLI subprocess.

    Uses YAML config + 'kpi preprocess predict' flow which is the
    correct MLSimKit invocation pattern.
    """
    import subprocess
    import yaml

    # Write YAML config that mlsimkit-learn expects
    config = {
        "output-dir": output_dir,
        "log": {
            "prefix-dir": os.path.join(output_dir, "logs"),
        },
        "kpi": {
            "manifest-uri": manifest_path,
            "preprocess": {
                "split-manifest": False,
            },
            "predict": {
                "model-path": model_path,
                "compare-groundtruth": False,
            },
        },
    }
    config_path = os.path.join(output_dir, "prediction_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    cmd = [
        _resolve_mlsimkit_executable(),
        "--config", config_path,
        "kpi", "preprocess", "predict",
    ]
    logger.info(f"Running CLI KPI inference: {' '.join(cmd)}")
    logger.info(f"PYTHONPATH={os.environ.get('PYTHONPATH', '<not set>')}")

    env = os.environ.copy()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)  # nosemgrep

    if result.returncode != 0:
        raise RuntimeError(f"mlsimkit-learn kpi predict failed: {result.stderr}")

    logger.info(f"CLI KPI stdout: {result.stdout[:500] if result.stdout else '<empty>'}")
    logger.info(f"CLI KPI stderr: {result.stderr[:500] if result.stderr else '<empty>'}")

    # MLSimKit outputs to output_dir/predictions/ or output_dir/prediction/
    for subdir in ["predictions", "prediction"]:
        results_csv = os.path.join(output_dir, subdir, "prediction_results.csv")
        if os.path.exists(results_csv):
            with open(results_csv, "r") as f:
                csv_content = f.read()
            logger.info(f"KPI CSV content ({results_csv}):\n{csv_content[:1000]}")
            return _parse_kpi_results(results_csv, variant_id)

    # List what files ARE in the output dir for debugging
    for root, dirs, files in os.walk(output_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            logger.info(f"KPI output file: {os.path.relpath(fpath, output_dir)}")

    raise FileNotFoundError(f"KPI results CSV not found in {output_dir}")




# ---------------------------------------------------------------------------
# Surface Variable Inference
# ---------------------------------------------------------------------------

def predict_surface_variables(
    variant_id: str,
    geometry_path: str,
    model_path: str | None = None,
    model_s3_key: str = "surface/best_model.pt",
) -> dict:
    """Run surface variable inference on a car body geometry file.

    Uses MLSimKit's Python API to predict cpavg (pressure coefficient)
    and cfxavg (skin friction coefficient) distributions mapped onto
    the mesh surface.

    Args:
        variant_id: Identifier for the design variant.
        geometry_path: Path to STL geometry file (local or s3:// URI).
        model_path: Direct local path to model checkpoint.
        model_s3_key: S3 key for the surface model.

    Returns:
        Dict with keys: variant_id, cpavg_field, cfxavg_field,
        mesh_vertices, mesh_faces, vtk_file_path, source.
    """
    start = time.time()

    try:
        # Check geometry STL exists before attempting live inference
        if geometry_path.startswith("s3://"):
            parts = geometry_path[5:].split("/", 1)
            if len(parts) == 2:
                geo_bucket, geo_key = parts
                if not _s3_object_exists(geo_bucket, geo_key):
                    return {
                        "variant_id": variant_id,
                        "status": "geometry_not_found",
                        "error_message": (
                            f"The geometry file for {variant_id} is not available in storage. "
                            f"Please upload the STL file using the upload button in the 3D viewer "
                            f"panel, then retry."
                        ),
                    }

        # Live inference path
        if model_path is None:
            model_path = _ensure_model_local(model_s3_key)
        geometry_local = _ensure_geometry_local(geometry_path)

        manifest_path = _create_single_manifest(geometry_local)
        output_dir = tempfile.mkdtemp(prefix="surface_predict_")

        try:
            from mlsimkit.learn.surface.inference import run_predict as surface_run_predict
            from mlsimkit.learn.surface.schema.inference import InferenceSettings

            settings = InferenceSettings(
                model_path=model_path,
                manifest_path=manifest_path,
                output_dir=output_dir,
            )
            surface_run_predict(settings)

            # Parse surface prediction output (VTP files)
            surface_data = _parse_surface_results(output_dir, variant_id)

        except Exception as api_err:
            logger.warning(f"MLSimKit surface API failed: {api_err} — falling back to CLI")
            surface_data = _predict_surface_cli(
                variant_id, geometry_local, model_path, manifest_path, output_dir
            )
        finally:
            _safe_remove(manifest_path)

        elapsed_ms = (time.time() - start) * 1000
        surface_data["inference_time_ms"] = round(elapsed_ms, 1)
        surface_data["source"] = "mlsimkit"
        return surface_data

    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        logger.error(f"Surface inference failed for {variant_id}: {e}")
        return {
            "variant_id": variant_id,
            "status": "error",
            "error_message": f"Surface inference failed: {e}",
            "inference_time_ms": round(elapsed_ms, 1),
        }


def _parse_surface_results(output_dir: str, variant_id: str) -> dict:
    """Parse MLSimKit surface prediction output.

    MLSimKit writes predicted surface variables as VTP files. We look for
    the prediction output and extract cpavg/cfxavg arrays plus mesh data.
    """
    predictions_dir = os.path.join(output_dir, "predictions")

    # Find VTP output files
    vtp_files = list(Path(predictions_dir).rglob("*.vtp")) if os.path.exists(predictions_dir) else []

    if vtp_files:
        vtp_path = str(vtp_files[0])
        cpavg, cfxavg, vertices, faces = _read_vtp_surface_data(vtp_path)
        return {
            "variant_id": variant_id,
            "cpavg_field": cpavg,
            "cfxavg_field": cfxavg,
            "mesh_vertices": vertices,
            "mesh_faces": faces,
            "vtk_file_path": vtp_path,
            "png_heatmap_path": "",
        }

    # Fallback: check for numpy arrays or other output formats
    logger.warning(f"No VTP files found in {predictions_dir}, returning empty surface data")
    return {
        "variant_id": variant_id,
        "cpavg_field": [],
        "cfxavg_field": [],
        "mesh_vertices": [],
        "mesh_faces": [],
        "vtk_file_path": "",
        "png_heatmap_path": "",
    }


def _read_vtp_surface_data(vtp_path: str) -> tuple[list, list, list, list]:
    """Read surface variable data from a VTP file.

    Uses pyvista (VTK wrapper) to extract point data arrays and mesh topology.
    Returns (cpavg, cfxavg, vertices, faces) as Python lists.
    """
    try:
        import pyvista as pv

        mesh = pv.read(vtp_path)
        vertices = mesh.points.tolist()

        # Extract faces — pyvista stores as flat array with counts
        if hasattr(mesh, "faces") and mesh.faces is not None and len(mesh.faces) > 0:
            faces = _extract_faces(mesh)
        else:
            faces = []

        # Extract scalar arrays
        cpavg = mesh.point_data.get("cpavg", mesh.point_data.get("Cpavg", []))
        cfxavg = mesh.point_data.get("cfxavg", mesh.point_data.get("Cfxavg", []))

        cpavg = cpavg.tolist() if hasattr(cpavg, "tolist") else list(cpavg)
        cfxavg = cfxavg.tolist() if hasattr(cfxavg, "tolist") else list(cfxavg)

        return cpavg, cfxavg, vertices, faces

    except ImportError:
        logger.warning("pyvista not installed — cannot read VTP surface data")
        return [], [], [], []
    except Exception as e:
        logger.error(f"Failed to read VTP file {vtp_path}: {e}")
        return [], [], [], []


def _extract_faces(mesh) -> list[list[int]]:
    """Extract face connectivity from a pyvista mesh as list of index lists."""
    faces = []
    raw = mesh.faces
    i = 0
    while i < len(raw):
        n = raw[i]
        face = raw[i + 1 : i + 1 + n].tolist()
        faces.append(face)
        i += 1 + n
    return faces


def _predict_surface_cli(
    variant_id: str,
    geometry_path: str,
    model_path: str,
    manifest_path: str,
    output_dir: str,
) -> dict:
    """Fallback: run surface inference via MLSimKit CLI subprocess."""
    import subprocess
    import yaml

    config = {
        "output-dir": output_dir,
        "log": {"prefix-dir": os.path.join(output_dir, "logs")},
        "surface": {
            "manifest-uri": manifest_path,
            "preprocess": {"split-manifest": False},
            "predict": {
                "model-path": model_path,
                "compare-groundtruth": False,
            },
        },
    }
    config_path = os.path.join(output_dir, "prediction_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    cmd = [
        _resolve_mlsimkit_executable(),
        "--config", config_path,
        "surface", "preprocess", "predict",
    ]
    logger.info(f"Running CLI surface inference: {' '.join(cmd)}")
    env = os.environ.copy()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)  # nosemgrep

    if result.returncode != 0:
        raise RuntimeError(f"mlsimkit-learn surface predict failed: {result.stderr}")

    return _parse_surface_results(output_dir, variant_id)


# ---------------------------------------------------------------------------
# Slices Inference (velocity flow field)
# ---------------------------------------------------------------------------

def predict_slices(
    variant_id: str,
    geometry_path: str,
    ae_model_path: str | None = None,
    mgn_model_path: str | None = None,
    ae_s3_key: str = "slices/ae_best_model.pt",
    mgn_s3_key: str = "slices/mgn_last_model.pt",
) -> dict:
    """Run slices inference on a car body geometry file.

    Uses both the autoencoder and MeshGraphNet prediction models
    to generate velocity flow field slice images.

    Args:
        variant_id: Identifier for the design variant.
        geometry_path: Path to STL geometry file.
        ae_model_path: Local path to autoencoder model.
        mgn_model_path: Local path to MGN prediction model.
        ae_s3_key: S3 key for autoencoder model.
        mgn_s3_key: S3 key for MGN model.

    Returns:
        Dict with variant_id, image paths, and status.
    """
    start = time.time()

    try:
        # Check geometry STL exists before attempting live inference
        if geometry_path.startswith("s3://"):
            parts = geometry_path[5:].split("/", 1)
            if len(parts) == 2:
                geo_bucket, geo_key = parts
                if not _s3_object_exists(geo_bucket, geo_key):
                    return {
                        "variant_id": variant_id,
                        "status": "geometry_not_found",
                        "error_message": (
                            f"The geometry file for {variant_id} is not available in storage. "
                            f"Please upload the STL file using the upload button in the 3D viewer "
                            f"panel, then retry."
                        ),
                    }

        # Live inference path
        if ae_model_path is None:
            ae_model_path = _ensure_model_local(ae_s3_key)
        if mgn_model_path is None:
            mgn_model_path = _ensure_model_local(mgn_s3_key)
        geometry_local = _ensure_geometry_local(geometry_path)

        manifest_path = _create_single_manifest(geometry_local)
        output_dir = tempfile.mkdtemp(prefix="slices_predict_")

        try:
            from mlsimkit.learn.slices.inference import run_predict as slices_run_predict
            from mlsimkit.learn.slices.schema.inference import InferenceSettings

            settings = InferenceSettings(
                ae_model_path=ae_model_path,
                mgn_model_path=mgn_model_path,
                manifest_path=manifest_path,
                output_dir=output_dir,
            )
            slices_run_predict(settings)

            # Collect outputs — slices produces:
            #   prediction/geometry-group-{id}-prediction.npy  (raw numpy)
            #   prediction/images/geometry-group-{id}-prediction-{0-9}.png (predict-only mode)
            #   prediction/images/geometry-group-{id}-combined-{0-9}.png (compare-groundtruth mode)
            #   prediction/results.jsonl
            prediction_dir = os.path.join(output_dir, "prediction")
            images_dir = os.path.join(prediction_dir, "images")

            # Collect prediction PNGs — try prediction-* first (predict-only), fall back to combined-*
            combined_images = sorted(
                str(p) for p in Path(images_dir).glob("*-prediction-*.png")
            ) if os.path.exists(images_dir) else []
            if not combined_images:
                combined_images = sorted(
                    str(p) for p in Path(images_dir).glob("*-combined-*.png")
                ) if os.path.exists(images_dir) else []

            # Collect raw numpy prediction files
            npy_files = sorted(
                str(p) for p in Path(prediction_dir).glob("*-prediction.npy")
            ) if os.path.exists(prediction_dir) else []

            # Parse metrics from results.jsonl if available
            metrics = _parse_slices_results(
                os.path.join(prediction_dir, "results.jsonl")
            )

            elapsed_ms = (time.time() - start) * 1000
            return {
                "variant_id": variant_id,
                "combined_images": combined_images,
                "npy_predictions": npy_files,
                "images_dir": images_dir,
                "metrics": metrics,
                "source": "mlsimkit",
                "inference_time_ms": round(elapsed_ms, 1),
            }

        except Exception as api_err:
            logger.warning(f"MLSimKit slices API failed: {api_err} — falling back to CLI")
            return _predict_slices_cli(
                variant_id, geometry_local, ae_model_path, mgn_model_path,
                manifest_path, output_dir, start,
            )
        finally:
            _safe_remove(manifest_path)

    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        logger.error(f"Slices inference failed for {variant_id}: {e}")
        return {
            "variant_id": variant_id,
            "status": "error",
            "error_message": f"Slices inference failed: {e}",
            "inference_time_ms": round(elapsed_ms, 1),
        }


def _parse_slices_results(results_jsonl_path: str) -> dict:
    """Parse metrics from MLSimKit slices prediction results.jsonl.

    Each line in results.jsonl is a JSON object with structure:
        {"row": {"geometry_files": [...], "id": N, ...},
         "metrics": {"mse": ..., "mae": ..., "mape": ..., "psnr": ...}}

    Returns aggregated metrics dict. For single-variant inference,
    returns the first entry's metrics directly.
    """
    if not os.path.exists(results_jsonl_path):
        logger.warning(f"Slices results.jsonl not found at {results_jsonl_path}")
        return {}

    metrics_list = []
    try:
        with open(results_jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if "metrics" in entry:
                    metrics_list.append(entry["metrics"])
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to parse slices results.jsonl: {e}")
        return {}

    if not metrics_list:
        return {}

    # Single-variant inference: return the first entry's metrics
    if len(metrics_list) == 1:
        return metrics_list[0]

    # Batch: compute averages across all entries
    keys = metrics_list[0].keys()
    averaged = {}
    for key in keys:
        vals = [m[key] for m in metrics_list if key in m]
        averaged[key] = sum(vals) / len(vals) if vals else 0.0
    averaged["num_entries"] = len(metrics_list)
    return averaged


def _predict_slices_cli(
    variant_id: str,
    geometry_path: str,
    ae_model_path: str,
    mgn_model_path: str,
    manifest_path: str,
    output_dir: str,
    start_time: float,
) -> dict:
    """Fallback: run slices inference via MLSimKit CLI subprocess."""
    import subprocess
    import yaml

    config = {
        "output-dir": output_dir,
        "log": {"prefix-dir": os.path.join(output_dir, "logs")},
        "slices": {
            "predict": {
                "manifest-path": manifest_path,
                "ae-model-path": ae_model_path,
                "mgn-model-path": mgn_model_path,
                "compare-groundtruth": False,
            },
        },
    }
    config_path = os.path.join(output_dir, "prediction_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    cmd = [
        _resolve_mlsimkit_executable(),
        "--config", config_path,
        "slices", "predict",
    ]
    logger.info(f"Running CLI slices inference: {' '.join(cmd)}")
    logger.info(f"Slices config: ae={ae_model_path}, mgn={mgn_model_path}, manifest={manifest_path}")
    env = os.environ.copy()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)  # nosemgrep

    logger.info(f"CLI slices stdout: {result.stdout[:1000] if result.stdout else '<empty>'}")
    logger.info(f"CLI slices stderr: {result.stderr[:1000] if result.stderr else '<empty>'}")
    logger.info(f"CLI slices returncode: {result.returncode}")

    # List output directory contents for debugging
    for root, dirs, files in os.walk(output_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            logger.info(f"Slices output file: {os.path.relpath(fpath, output_dir)}")

    if result.returncode != 0:
        raise RuntimeError(f"mlsimkit-learn slices predict failed (rc={result.returncode}): {result.stderr[:500]}")

    prediction_dir = os.path.join(output_dir, "prediction")
    images_dir = os.path.join(prediction_dir, "images")

    # Collect prediction PNGs — try prediction-* first (predict-only), fall back to combined-*
    combined_images = sorted(
        str(p) for p in Path(images_dir).glob("*-prediction-*.png")
    ) if os.path.exists(images_dir) else []
    if not combined_images:
        combined_images = sorted(
            str(p) for p in Path(images_dir).glob("*-combined-*.png")
        ) if os.path.exists(images_dir) else []

    npy_files = sorted(
        str(p) for p in Path(prediction_dir).glob("*-prediction.npy")
    ) if os.path.exists(prediction_dir) else []

    metrics = _parse_slices_results(
        os.path.join(prediction_dir, "results.jsonl")
    )

    elapsed_ms = (time.time() - start_time) * 1000
    return {
        "variant_id": variant_id,
        "combined_images": combined_images,
        "npy_predictions": npy_files,
        "images_dir": images_dir,
        "metrics": metrics,
        "source": "mlsimkit_cli",
        "inference_time_ms": round(elapsed_ms, 1),
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _safe_remove(path: str) -> None:
    """Remove a file if it exists, silently ignoring errors."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
