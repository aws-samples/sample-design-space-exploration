#!/usr/bin/env python3
"""
Manifest creation for MLSimKit training pipeline.

Scans a WindsorML dataset directory for geometry (STL/VTP) and CFD result
files, creates a JSON Lines manifest mapping each geometry to its results.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def create_manifest(
    dataset_dir: str,
    output_path: str,
    geometry_ext: str = ".stl",
    result_pattern: str = "*_kpi.csv",
) -> str:
    """Scan dataset directory and create a JSON Lines manifest file.

    The WindsorML dataset structure is:
        dataset_dir/
            run_1/windsor_1.stl
            run_1/results/kpi.csv
            run_2/windsor_2.stl
            ...

    Each manifest line maps a geometry file to its CFD result file.

    Args:
        dataset_dir: Root directory of the WindsorML dataset.
        output_path: Path to write the manifest file.
        geometry_ext: File extension for geometry files (default: .stl).
        result_pattern: Glob pattern for result files.

    Returns:
        Path to the created manifest file.
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    entries = []

    # Scan for geometry files
    geometry_files = sorted(dataset_path.rglob(f"*{geometry_ext}"))
    logger.info(f"Found {len(geometry_files)} geometry files in {dataset_dir}")

    for geo_path in geometry_files:
        variant_dir = geo_path.parent
        variant_id = variant_dir.name  # e.g., "run_15"

        # Look for result files in the variant directory or a results subdirectory
        result_files = list(variant_dir.rglob(result_pattern))
        if not result_files:
            # Try common alternative locations
            for alt_pattern in ["*.csv", "results/*.csv", "kpi.csv"]:
                result_files = list(variant_dir.rglob(alt_pattern))
                if result_files:
                    break

        entry = {
            "geometry_files": [f"file://{geo_path.resolve()}"],
            "variant_id": variant_id,
        }

        if result_files:
            entry["result_files"] = [f"file://{r.resolve()}" for r in result_files]

        entries.append(entry)

    # Write manifest
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Created manifest with {len(entries)} entries at {output_path}")
    return output_path


def parse_manifest(manifest_path: str) -> list[dict]:
    """Parse a JSON Lines manifest file back into a list of entries."""
    entries = []
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
