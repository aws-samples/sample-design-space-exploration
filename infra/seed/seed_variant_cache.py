#!/usr/bin/env python3
"""
Seed CarDesignVariantCache DynamoDB table with pre-computed KPIs,
structural metrics, and cost estimates for all 355 WindsorML variants.

Run this on the EC2 training instance where MLSimKit and models are available,
or provide pre-computed CSV data to skip live inference.

Usage:
    # From EC2 with MLSimKit installed:
    python seed_variant_cache.py --mode inference --dataset-dir /path/to/windsorml

    # From local with pre-computed CSV:
    python seed_variant_cache.py --mode csv --csv-path variant_data.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from decimal import Decimal
from pathlib import Path

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TABLE_NAME = os.environ.get("VARIANT_CACHE_TABLE", "CarDesignVariantCache")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Default material properties for structural/cost estimation
MATERIAL_DENSITY = {"steel": 7850.0, "aluminum": 2700.0}  # kg/m³
BASE_THICKNESS_MM = {"steel": 0.8, "aluminum": 1.2}
MATERIAL_COST_PER_KG = {"steel": 1.20, "aluminum": 3.50}
STAMPING_COST_PER_OP = 150.0
TOOLING_BASE_COST = 50000.0
WELDING_COST_PER_M = 12.0


def d(val: float) -> Decimal:
    return Decimal(str(round(val, 6)))


def estimate_structural(surface_area_m2: float, material: str = "steel") -> dict:
    """Quick structural estimation from surface area."""
    density = MATERIAL_DENSITY[material]
    thickness_mm = BASE_THICKNESS_MM[material]
    thickness_m = thickness_mm / 1000.0
    weight_kg = surface_area_m2 * thickness_m * density
    stiffness_score = min(1.0, max(0.0, 0.7 + 0.1 * (1.0 - surface_area_m2 / 5.0)))
    return {
        "weight_kg": round(weight_kg, 2),
        "stiffness_score": round(stiffness_score, 3),
        "is_feasible": True,
    }


def estimate_cost(weight_kg: float, surface_area_m2: float, material: str = "steel") -> dict:
    """Quick cost estimation from weight and surface area."""
    material_cost = weight_kg * MATERIAL_COST_PER_KG[material]
    stamping_cost = 2 * STAMPING_COST_PER_OP  # Assume 2 operations
    complexity = min(1.0, surface_area_m2 / 4.0)
    multiplier = 1.0 + 0.8 * complexity
    tooling_cost = TOOLING_BASE_COST * multiplier
    weld_length = surface_area_m2 * 0.5  # Rough estimate
    assembly_cost = 4 * 50.0 + weld_length * WELDING_COST_PER_M
    total = material_cost + stamping_cost + tooling_cost + assembly_cost
    return {
        "total_cost": round(total, 2),
        "material_cost": round(material_cost, 2),
        "stamping_cost": round(stamping_cost, 2),
        "tooling_cost": round(tooling_cost, 2),
        "assembly_cost": round(assembly_cost, 2),
    }


def seed_from_inference(dataset_dir: str) -> None:
    """Run KPI inference on all variants and seed the cache."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from backend.training.inference import predict_kpi

    geometry_dir = Path(dataset_dir)
    stl_files = sorted(geometry_dir.rglob("*.stl"))
    logger.info(f"Found {len(stl_files)} STL files in {dataset_dir}")

    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    table = dynamodb.Table(TABLE_NAME)

    with table.batch_writer() as batch:
        for i, stl_path in enumerate(stl_files):
            variant_id = stl_path.parent.name  # e.g., "run_15"
            logger.info(f"[{i+1}/{len(stl_files)}] Processing {variant_id}...")

            try:
                kpis = predict_kpi(variant_id, str(stl_path))
                if kpis.get("status") == "error":
                    logger.warning(f"  KPI inference failed: {kpis.get('error_message')}")
                    continue

                # Estimate surface area from file size heuristic (~0.5-3.0 m²)
                file_size_mb = stl_path.stat().st_size / (1024 * 1024)
                surface_area = max(0.5, min(4.0, file_size_mb * 0.3))

                structural = estimate_structural(surface_area)
                cost = estimate_cost(structural["weight_kg"], surface_area)

                item = {
                    "pk": "VARIANT",
                    "sk": variant_id,
                    "cd": d(kpis.get("drag_coefficient", 0)),
                    "cs": d(kpis.get("side_force_coefficient", 0)),
                    "cl": d(kpis.get("lift_coefficient", 0)),
                    "cmy": d(kpis.get("yaw_moment_coefficient", 0)),
                    "weight_kg": d(structural["weight_kg"]),
                    "stiffness_score": d(structural["stiffness_score"]),
                    "is_feasible": structural["is_feasible"],
                    "total_cost": d(cost["total_cost"]),
                    "material_cost": d(cost["material_cost"]),
                    "stamping_cost": d(cost["stamping_cost"]),
                    "tooling_cost": d(cost["tooling_cost"]),
                    "assembly_cost": d(cost["assembly_cost"]),
                    "geometry_s3_key": f"geometries/{variant_id}/{stl_path.name}",
                }
                batch.put_item(Item=item)
                logger.info(f"  ✓ Cd={kpis.get('drag_coefficient', 0):.4f}, cost=${cost['total_cost']:.0f}")

            except Exception as e:
                logger.error(f"  ✗ Failed: {e}")

    logger.info("Done seeding variant cache.")


def seed_from_csv(csv_path: str) -> None:
    """Seed cache from a pre-computed CSV file."""
    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    table = dynamodb.Table(TABLE_NAME)

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        with table.batch_writer() as batch:
            for row in reader:
                item = {
                    "pk": "VARIANT",
                    "sk": row["variant_id"],
                    # KPIs
                    "cd": d(float(row.get("cd", 0))),
                    "cs": d(float(row.get("cs", 0))),
                    "cl": d(float(row.get("cl", 0))),
                    "cmy": d(float(row.get("cmy", 0))),
                    # Geometry metrics
                    "surface_area_m2": d(float(row.get("surface_area_m2", 0))),
                    "vertex_count": int(row.get("vertex_count", 0)),
                    "curvature_variation": d(float(row.get("curvature_variation", 0))),
                    "surface_patch_count": int(row.get("surface_patch_count", 0)),
                    "max_draw_depth_mm": d(float(row.get("max_draw_depth_mm", 0))),
                    "has_undercuts": row.get("has_undercuts", "false").lower() == "true",
                    # Structural estimates
                    "weight_kg": d(float(row.get("weight_kg", 0))),
                    "stiffness_score": d(float(row.get("stiffness_score", 0))),
                    "is_feasible": row.get("is_feasible", "true").lower() == "true",
                    # Cost estimates
                    "total_cost": d(float(row.get("total_cost", 0))),
                    "material_cost": d(float(row.get("material_cost", 0))),
                    "stamping_cost": d(float(row.get("stamping_cost", 0))),
                    "tooling_cost": d(float(row.get("tooling_cost", 0))),
                    "assembly_cost": d(float(row.get("assembly_cost", 0))),
                    # Asset paths (for frontend 2D/3D visualization)
                    "vtp_s3_key": row.get("vtp_s3_key", ""),
                    "heatmap_png_s3_key": row.get("heatmap_png_s3_key", ""),
                    "slice_images_json": row.get("slice_images_json", "[]"),
                    "geometry_s3_key": row.get("geometry_s3_key", ""),
                }
                batch.put_item(Item=item)

    logger.info(f"Done seeding from CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Seed CarDesignVariantCache DynamoDB table")
    parser.add_argument("--mode", choices=["inference", "csv"], default="inference")
    parser.add_argument("--dataset-dir", default="/home/ubuntu/ai-surrogate-models-in-engineering-on-aws/tutorials/kpi/windsor/dataset")
    parser.add_argument("--csv-path", default="variant_data.csv")
    parser.add_argument("--table", default=TABLE_NAME)
    parser.add_argument("--region", default=AWS_REGION)
    args = parser.parse_args()

    global TABLE_NAME, AWS_REGION
    TABLE_NAME = args.table
    AWS_REGION = args.region

    if args.mode == "inference":
        seed_from_inference(args.dataset_dir)
    else:
        seed_from_csv(args.csv_path)


if __name__ == "__main__":
    main()
