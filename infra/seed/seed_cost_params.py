#!/usr/bin/env python3
"""
Seed script for the CarDesignCostParameters DynamoDB table.

Populates ~50 rows of internal cost parameters used by the Cost Agent's
Internal Cost Parameters MCP Server. Categories:
  - MATERIAL: Base cost per kg for 7 automotive materials
  - STAMPING: Press operation costs, multipliers, setup times
  - TOOLING: Die costs by type, maintenance, amortization
  - ASSEMBLY: Welding, adhesive bonding, riveting, panel handling
  - MULTIPLIER: Complexity multipliers (5 tiers)
  - SURFACE_TREATMENT: Paint, e-coat, galvanizing, anodizing
  - QUALITY: Inspection, rework, scrap rates
  - LOGISTICS: Handling, packaging, intra-plant transport

Usage:
  python seed_cost_params.py [--table TABLE_NAME] [--region REGION] [--create-table]
"""

from __future__ import annotations

import argparse
import logging
import sys
from decimal import Decimal

import boto3

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def d(val: float) -> Decimal:
    """Convert float to Decimal for DynamoDB."""
    return Decimal(str(val))


# ---------------------------------------------------------------------------
# Data definitions
# ---------------------------------------------------------------------------

MATERIALS = [
    {"pk": "MATERIAL", "sk": "steel", "value": d(1.20), "unit": "USD/kg", "description": "CR4 mild steel sheet, cold-rolled, 0.7-1.2mm gauge"},
    {"pk": "MATERIAL", "sk": "aluminum", "value": d(3.50), "unit": "USD/kg", "description": "6061-T6 aluminum alloy sheet, aerospace grade"},
    {"pk": "MATERIAL", "sk": "carbon_fiber", "value": d(25.00), "unit": "USD/kg", "description": "T700S carbon fiber reinforced polymer, pre-preg layup"},
    {"pk": "MATERIAL", "sk": "high_strength_steel", "value": d(1.85), "unit": "USD/kg", "description": "DP780 dual-phase high-strength steel, hot-stamped"},
    {"pk": "MATERIAL", "sk": "magnesium", "value": d(4.20), "unit": "USD/kg", "description": "AZ91D magnesium alloy die-cast sheet"},
    {"pk": "MATERIAL", "sk": "titanium", "value": d(35.00), "unit": "USD/kg", "description": "Ti-6Al-4V Grade 5 titanium sheet, superplastic forming"},
    {"pk": "MATERIAL", "sk": "glass_fiber", "value": d(8.50), "unit": "USD/kg", "description": "E-glass fiber reinforced SMC, compression molded"},
]

STAMPING = [
    {"pk": "STAMPING", "sk": "cost_per_operation", "value": d(150.0), "unit": "USD/op", "description": "Base cost per stamping press operation (1000-ton hydraulic)"},
    {"pk": "STAMPING", "sk": "multi_stage_multiplier", "value": d(1.15), "unit": "multiplier", "description": "Cost multiplier for multi-stage progressive stamping"},
    {"pk": "STAMPING", "sk": "setup_time_hours", "value": d(4.5), "unit": "hours", "description": "Average die changeover and setup time per production run"},
    {"pk": "STAMPING", "sk": "setup_cost_per_hour", "value": d(220.0), "unit": "USD/hr", "description": "Skilled technician rate for press setup and alignment"},
    {"pk": "STAMPING", "sk": "blanking_cost_per_kg", "value": d(0.35), "unit": "USD/kg", "description": "Sheet metal blanking and nesting cost per kg of material"},
    {"pk": "STAMPING", "sk": "scrap_rate_steel", "value": d(0.12), "unit": "ratio", "description": "Typical scrap rate for steel stamping (12% material waste)"},
    {"pk": "STAMPING", "sk": "scrap_rate_aluminum", "value": d(0.18), "unit": "ratio", "description": "Typical scrap rate for aluminum stamping (18% material waste)"},
    {"pk": "STAMPING", "sk": "hot_stamping_surcharge", "value": d(85.0), "unit": "USD/op", "description": "Additional cost per operation for hot-stamping (furnace + quench)"},
]

TOOLING = [
    {"pk": "TOOLING", "sk": "base_cost", "value": d(50000.0), "unit": "USD", "description": "Base tooling cost for a single stamping die set (draw die)"},
    {"pk": "TOOLING", "sk": "trim_die_cost", "value": d(22000.0), "unit": "USD", "description": "Trim and pierce die for edge finishing after draw"},
    {"pk": "TOOLING", "sk": "flange_die_cost", "value": d(18000.0), "unit": "USD", "description": "Flange and hem die for panel edge folding"},
    {"pk": "TOOLING", "sk": "checking_fixture_cost", "value": d(15000.0), "unit": "USD", "description": "Dimensional checking fixture for quality verification"},
    {"pk": "TOOLING", "sk": "prototype_die_cost", "value": d(12000.0), "unit": "USD", "description": "Soft-tool prototype die (kirksite) for validation runs"},
    {"pk": "TOOLING", "sk": "maintenance_annual_pct", "value": d(0.08), "unit": "ratio", "description": "Annual die maintenance cost as percentage of die value (8%)"},
    {"pk": "TOOLING", "sk": "amortization_units", "value": d(150000), "unit": "parts", "description": "Expected die life for cost amortization (150k parts)"},
]

ASSEMBLY = [
    {"pk": "ASSEMBLY", "sk": "welding_cost_per_meter", "value": d(12.0), "unit": "USD/m", "description": "Robotic MIG/MAG welding cost per meter of weld line"},
    {"pk": "ASSEMBLY", "sk": "assembly_cost_per_panel", "value": d(85.0), "unit": "USD/panel", "description": "Assembly labor and fixturing cost per body panel"},
    {"pk": "ASSEMBLY", "sk": "spot_weld_cost", "value": d(0.08), "unit": "USD/weld", "description": "Robotic resistance spot weld cost per weld point"},
    {"pk": "ASSEMBLY", "sk": "laser_weld_cost_per_meter", "value": d(18.50), "unit": "USD/m", "description": "Robotic laser welding cost per meter (higher precision)"},
    {"pk": "ASSEMBLY", "sk": "adhesive_bond_cost_per_meter", "value": d(6.50), "unit": "USD/m", "description": "Structural adhesive bonding cost per meter of bond line"},
    {"pk": "ASSEMBLY", "sk": "rivet_cost_per_point", "value": d(0.45), "unit": "USD/rivet", "description": "Self-piercing rivet (SPR) cost per fastening point"},
    {"pk": "ASSEMBLY", "sk": "hemming_cost_per_meter", "value": d(8.00), "unit": "USD/m", "description": "Roller hemming cost per meter for closure panels"},
    {"pk": "ASSEMBLY", "sk": "fixture_cost_per_station", "value": d(35000.0), "unit": "USD", "description": "Assembly fixture cost per welding station"},
]

MULTIPLIERS = [
    {"pk": "MULTIPLIER", "sk": "very_low", "value": d(0.85), "unit": "multiplier", "description": "Very simple geometry — flat panels, minimal forming"},
    {"pk": "MULTIPLIER", "sk": "low", "value": d(1.0), "unit": "multiplier", "description": "Simple geometry — gentle curvature, shallow draw depth"},
    {"pk": "MULTIPLIER", "sk": "medium", "value": d(1.3), "unit": "multiplier", "description": "Moderate complexity — typical sedan body panels"},
    {"pk": "MULTIPLIER", "sk": "high", "value": d(1.8), "unit": "multiplier", "description": "Complex geometry — deep draws, compound curves, undercuts"},
    {"pk": "MULTIPLIER", "sk": "very_high", "value": d(2.5), "unit": "multiplier", "description": "Extreme complexity — supercar body, multi-piece compound forms"},
]

SURFACE_TREATMENT = [
    {"pk": "SURFACE_TREATMENT", "sk": "e_coat_cost_per_m2", "value": d(4.50), "unit": "USD/m²", "description": "Electrophoretic coating (e-coat) corrosion protection per m²"},
    {"pk": "SURFACE_TREATMENT", "sk": "primer_cost_per_m2", "value": d(3.20), "unit": "USD/m²", "description": "Primer coat application cost per m² of body surface"},
    {"pk": "SURFACE_TREATMENT", "sk": "basecoat_cost_per_m2", "value": d(5.80), "unit": "USD/m²", "description": "Basecoat (color) paint application cost per m²"},
    {"pk": "SURFACE_TREATMENT", "sk": "clearcoat_cost_per_m2", "value": d(4.10), "unit": "USD/m²", "description": "Clearcoat application cost per m² for UV and scratch protection"},
    {"pk": "SURFACE_TREATMENT", "sk": "galvanizing_cost_per_m2", "value": d(2.80), "unit": "USD/m²", "description": "Hot-dip galvanizing cost per m² for underbody panels"},
    {"pk": "SURFACE_TREATMENT", "sk": "anodizing_cost_per_m2", "value": d(8.50), "unit": "USD/m²", "description": "Anodizing cost per m² for aluminum body panels"},
]

QUALITY = [
    {"pk": "QUALITY", "sk": "cmm_inspection_per_panel", "value": d(12.0), "unit": "USD/panel", "description": "CMM (coordinate measuring machine) dimensional inspection per panel"},
    {"pk": "QUALITY", "sk": "visual_inspection_per_unit", "value": d(35.0), "unit": "USD/unit", "description": "Manual visual quality inspection per body-in-white unit"},
    {"pk": "QUALITY", "sk": "rework_cost_per_hour", "value": d(95.0), "unit": "USD/hr", "description": "Skilled rework technician rate for defect correction"},
    {"pk": "QUALITY", "sk": "target_scrap_rate", "value": d(0.02), "unit": "ratio", "description": "Target finished body scrap rate (2% of production)"},
    {"pk": "QUALITY", "sk": "ultrasonic_test_per_weld", "value": d(1.50), "unit": "USD/test", "description": "Ultrasonic weld quality testing cost per weld point"},
]

LOGISTICS = [
    {"pk": "LOGISTICS", "sk": "intra_plant_transport_per_unit", "value": d(18.0), "unit": "USD/unit", "description": "AGV/conveyor transport cost per body unit within plant"},
    {"pk": "LOGISTICS", "sk": "packaging_cost_per_panel", "value": d(3.50), "unit": "USD/panel", "description": "Interleaving and rack packaging cost per stamped panel"},
    {"pk": "LOGISTICS", "sk": "warehouse_cost_per_m3_day", "value": d(0.85), "unit": "USD/m³/day", "description": "WIP buffer storage cost per cubic meter per day"},
    {"pk": "LOGISTICS", "sk": "energy_cost_per_kwh", "value": d(0.09), "unit": "USD/kWh", "description": "Industrial electricity rate for stamping and welding operations"},
]


# ---------------------------------------------------------------------------
# Seed logic
# ---------------------------------------------------------------------------

def create_table_if_not_exists(table_name: str, region: str) -> None:
    """Create the DynamoDB table if it doesn't exist."""
    client = boto3.client("dynamodb", region_name=region)
    existing = [t for t in client.list_tables()["TableNames"] if t == table_name]
    if existing:
        logger.info(f"Table '{table_name}' already exists")
        return

    logger.info(f"Creating table '{table_name}'...")
    client.create_table(
        TableName=table_name,
        KeySchema=[
            {"AttributeName": "pk", "KeyType": "HASH"},
            {"AttributeName": "sk", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "pk", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    waiter = client.get_waiter("table_exists")
    waiter.wait(TableName=table_name)
    logger.info(f"Table '{table_name}' created")


def seed_data(table_name: str, region: str) -> None:
    """Write all seed data to the table."""
    dynamodb = boto3.resource("dynamodb", region_name=region)
    table = dynamodb.Table(table_name)

    all_items = MATERIALS + STAMPING + TOOLING + ASSEMBLY + MULTIPLIERS + SURFACE_TREATMENT + QUALITY + LOGISTICS
    logger.info(f"Seeding {len(all_items)} items into '{table_name}'...")

    with table.batch_writer() as batch:
        for item in all_items:
            batch.put_item(Item=item)

    logger.info(f"✅ Seeded {len(all_items)} items successfully")

    categories = {}
    for item in all_items:
        cat = item["pk"]
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        logger.info(f"  {cat}: {count} items")


def main():
    parser = argparse.ArgumentParser(description="Seed CarDesignCostParameters table")
    parser.add_argument("--table", default="CarDesignCostParameters", help="DynamoDB table name")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--create-table", action="store_true", help="Create table if not exists")
    args = parser.parse_args()

    if args.create_table:
        create_table_if_not_exists(args.table, args.region)

    seed_data(args.table, args.region)


if __name__ == "__main__":
    main()
