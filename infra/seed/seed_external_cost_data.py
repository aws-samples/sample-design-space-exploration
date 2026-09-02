#!/usr/bin/env python3
"""
Seed script for the CarDesignExternalCostData DynamoDB table.

Populates ~55 rows of realistic automotive manufacturing cost data
across five categories:
  - MARKET_PRICE: Commodity pricing for 6 materials × 3 regions
  - SUPPLIER: Tooling/fixture quotes from multiple suppliers
  - HISTORICAL: Cost benchmarks by material × complexity × body style
  - REGIONAL: Manufacturing cost multipliers for 5 regions
  - VOLUME_DISCOUNT: Volume-based pricing tiers for 3 materials

Usage:
  python seed_external_cost_data.py [--table TABLE_NAME] [--region REGION]
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

MARKET_PRICES = [
    # Global base prices
    {"pk": "MARKET_PRICE", "sk": "steel", "price_per_kg": d(1.20), "currency": "USD", "trend": "stable", "trend_pct_30d": d(-0.5), "source": "LME_Index", "effective_date": "2026-02-01", "grade": "CR4_mild"},
    {"pk": "MARKET_PRICE", "sk": "aluminum", "price_per_kg": d(3.50), "currency": "USD", "trend": "rising", "trend_pct_30d": d(2.3), "source": "LME_Index", "effective_date": "2026-02-01", "grade": "6061-T6"},
    {"pk": "MARKET_PRICE", "sk": "carbon_fiber", "price_per_kg": d(25.00), "currency": "USD", "trend": "declining", "trend_pct_30d": d(-3.1), "source": "Toray_Quote", "effective_date": "2026-01-15", "grade": "T700S"},
    {"pk": "MARKET_PRICE", "sk": "high_strength_steel", "price_per_kg": d(1.85), "currency": "USD", "trend": "stable", "trend_pct_30d": d(0.2), "source": "LME_Index", "effective_date": "2026-02-01", "grade": "DP780"},
    {"pk": "MARKET_PRICE", "sk": "magnesium", "price_per_kg": d(4.20), "currency": "USD", "trend": "rising", "trend_pct_30d": d(1.8), "source": "Shanghai_Metal", "effective_date": "2026-02-01", "grade": "AZ91D"},
    {"pk": "MARKET_PRICE", "sk": "titanium", "price_per_kg": d(35.00), "currency": "USD", "trend": "stable", "trend_pct_30d": d(0.0), "source": "USGS_Index", "effective_date": "2026-02-01", "grade": "Ti-6Al-4V"},
    # Regional variants
    {"pk": "MARKET_PRICE", "sk": "steel#north_america", "price_per_kg": d(1.25), "currency": "USD", "trend": "stable", "trend_pct_30d": d(-0.3), "source": "CRU_Steel", "effective_date": "2026-02-01", "grade": "CR4_mild"},
    {"pk": "MARKET_PRICE", "sk": "steel#europe", "price_per_kg": d(1.35), "currency": "USD", "trend": "rising", "trend_pct_30d": d(1.1), "source": "Platts_EU", "effective_date": "2026-02-01", "grade": "CR4_mild"},
    {"pk": "MARKET_PRICE", "sk": "steel#asia_pacific", "price_per_kg": d(1.05), "currency": "USD", "trend": "declining", "trend_pct_30d": d(-1.5), "source": "MySteel_CN", "effective_date": "2026-02-01", "grade": "CR4_mild"},
    {"pk": "MARKET_PRICE", "sk": "aluminum#north_america", "price_per_kg": d(3.65), "currency": "USD", "trend": "rising", "trend_pct_30d": d(2.8), "source": "Midwest_Premium", "effective_date": "2026-02-01", "grade": "6061-T6"},
    {"pk": "MARKET_PRICE", "sk": "aluminum#europe", "price_per_kg": d(3.80), "currency": "USD", "trend": "rising", "trend_pct_30d": d(3.2), "source": "LME_EU_Duty", "effective_date": "2026-02-01", "grade": "6061-T6"},
    {"pk": "MARKET_PRICE", "sk": "aluminum#asia_pacific", "price_per_kg": d(3.20), "currency": "USD", "trend": "stable", "trend_pct_30d": d(0.5), "source": "SHFE_Index", "effective_date": "2026-02-01", "grade": "6061-T6"},
    {"pk": "MARKET_PRICE", "sk": "carbon_fiber#north_america", "price_per_kg": d(24.50), "currency": "USD", "trend": "declining", "trend_pct_30d": d(-2.8), "source": "Hexcel_Quote", "effective_date": "2026-01-20", "grade": "T700S"},
    {"pk": "MARKET_PRICE", "sk": "carbon_fiber#europe", "price_per_kg": d(26.00), "currency": "USD", "trend": "declining", "trend_pct_30d": d(-2.5), "source": "SGL_Carbon", "effective_date": "2026-01-20", "grade": "T700S"},
    {"pk": "MARKET_PRICE", "sk": "carbon_fiber#asia_pacific", "price_per_kg": d(22.00), "currency": "USD", "trend": "declining", "trend_pct_30d": d(-4.0), "source": "Toray_JP", "effective_date": "2026-01-20", "grade": "T700S"},
]

SUPPLIERS = [
    # Stamping dies — 3 suppliers
    {"pk": "SUPPLIER", "sk": "stamping_die#supplier_1", "supplier_name": "PrecisionDie Corp", "supplier_id": "SUP-001", "base_price": d(48000), "currency": "USD", "lead_time_weeks": d(10), "availability": "available", "warranty_months": d(24), "rating": d(4.5)},
    {"pk": "SUPPLIER", "sk": "stamping_die#supplier_2", "supplier_name": "ToolTech Industries", "supplier_id": "SUP-002", "base_price": d(45000), "currency": "USD", "lead_time_weeks": d(12), "availability": "available", "warranty_months": d(18), "rating": d(4.2)},
    {"pk": "SUPPLIER", "sk": "stamping_die#supplier_3", "supplier_name": "Shanghai Precision", "supplier_id": "SUP-003", "base_price": d(38000), "currency": "USD", "lead_time_weeks": d(16), "availability": "available", "warranty_months": d(12), "rating": d(3.8)},
    # Welding fixtures — 2 suppliers
    {"pk": "SUPPLIER", "sk": "welding_fixture#supplier_1", "supplier_name": "WeldTech Solutions", "supplier_id": "SUP-004", "base_price": d(8500), "currency": "USD", "lead_time_weeks": d(6), "availability": "available", "warranty_months": d(12), "rating": d(4.3)},
    {"pk": "SUPPLIER", "sk": "welding_fixture#supplier_2", "supplier_name": "JoinPro Systems", "supplier_id": "SUP-005", "base_price": d(7800), "currency": "USD", "lead_time_weeks": d(8), "availability": "limited", "warranty_months": d(12), "rating": d(4.0)},
    # Assembly jigs — 2 suppliers
    {"pk": "SUPPLIER", "sk": "assembly_jig#supplier_1", "supplier_name": "AssemblyMaster", "supplier_id": "SUP-006", "base_price": d(12000), "currency": "USD", "lead_time_weeks": d(8), "availability": "available", "warranty_months": d(18), "rating": d(4.4)},
    {"pk": "SUPPLIER", "sk": "assembly_jig#supplier_2", "supplier_name": "FixturePro GmbH", "supplier_id": "SUP-007", "base_price": d(14500), "currency": "USD", "lead_time_weeks": d(6), "availability": "available", "warranty_months": d(24), "rating": d(4.7)},
    # Trim dies
    {"pk": "SUPPLIER", "sk": "trim_die#supplier_1", "supplier_name": "PrecisionDie Corp", "supplier_id": "SUP-001", "base_price": d(22000), "currency": "USD", "lead_time_weeks": d(8), "availability": "available", "warranty_months": d(18), "rating": d(4.5)},
    # Inspection fixtures
    {"pk": "SUPPLIER", "sk": "inspection_fixture#supplier_1", "supplier_name": "MetroTech QA", "supplier_id": "SUP-008", "base_price": d(18000), "currency": "USD", "lead_time_weeks": d(10), "availability": "available", "warranty_months": d(24), "rating": d(4.6)},
    # Paint booth time
    {"pk": "SUPPLIER", "sk": "paint_booth_time#supplier_1", "supplier_name": "FinishPro Coatings", "supplier_id": "SUP-009", "base_price": d(250), "currency": "USD/hour", "lead_time_weeks": d(2), "availability": "available", "warranty_months": d(0), "rating": d(4.1)},
]

HISTORICAL = [
    # Steel variants
    {"pk": "HISTORICAL", "sk": "steel#low#sedan", "sample_size": d(85), "date_range": "2023-2025", "min_cost": d(38000), "max_cost": d(52000), "avg_cost": d(44500), "median_cost": d(43800), "p25_cost": d(41200), "p75_cost": d(47500), "std_dev": d(3800), "material_pct": d(15), "stamping_pct": d(25), "tooling_pct": d(40), "assembly_pct": d(20), "ci_95": "42800-46200"},
    {"pk": "HISTORICAL", "sk": "steel#medium#sedan", "sample_size": d(120), "date_range": "2023-2025", "min_cost": d(48000), "max_cost": d(68000), "avg_cost": d(56000), "median_cost": d(55200), "p25_cost": d(52000), "p75_cost": d(60000), "std_dev": d(4500), "material_pct": d(14), "stamping_pct": d(26), "tooling_pct": d(38), "assembly_pct": d(22), "ci_95": "54200-57800"},
    {"pk": "HISTORICAL", "sk": "steel#high#sedan", "sample_size": d(45), "date_range": "2023-2025", "min_cost": d(65000), "max_cost": d(95000), "avg_cost": d(78000), "median_cost": d(76500), "p25_cost": d(72000), "p75_cost": d(84000), "std_dev": d(7200), "material_pct": d(12), "stamping_pct": d(28), "tooling_pct": d(36), "assembly_pct": d(24), "ci_95": "74800-81200"},
    {"pk": "HISTORICAL", "sk": "steel#medium#suv", "sample_size": d(95), "date_range": "2023-2025", "min_cost": d(55000), "max_cost": d(78000), "avg_cost": d(65000), "median_cost": d(64200), "p25_cost": d(60000), "p75_cost": d(70000), "std_dev": d(5200), "material_pct": d(16), "stamping_pct": d(24), "tooling_pct": d(37), "assembly_pct": d(23), "ci_95": "62800-67200"},
    # Aluminum variants
    {"pk": "HISTORICAL", "sk": "aluminum#low#sedan", "sample_size": d(35), "date_range": "2023-2025", "min_cost": d(52000), "max_cost": d(72000), "avg_cost": d(61000), "median_cost": d(60500), "p25_cost": d(57000), "p75_cost": d(65000), "std_dev": d(4800), "material_pct": d(22), "stamping_pct": d(22), "tooling_pct": d(35), "assembly_pct": d(21), "ci_95": "58500-63500"},
    {"pk": "HISTORICAL", "sk": "aluminum#medium#sedan", "sample_size": d(62), "date_range": "2023-2025", "min_cost": d(62000), "max_cost": d(88000), "avg_cost": d(74000), "median_cost": d(73000), "p25_cost": d(69000), "p75_cost": d(79000), "std_dev": d(5800), "material_pct": d(24), "stamping_pct": d(21), "tooling_pct": d(34), "assembly_pct": d(21), "ci_95": "71200-76800"},
    {"pk": "HISTORICAL", "sk": "aluminum#high#sedan", "sample_size": d(22), "date_range": "2023-2025", "min_cost": d(85000), "max_cost": d(125000), "avg_cost": d(102000), "median_cost": d(100000), "p25_cost": d(94000), "p75_cost": d(110000), "std_dev": d(9500), "material_pct": d(20), "stamping_pct": d(23), "tooling_pct": d(33), "assembly_pct": d(24), "ci_95": "96000-108000"},
    # Carbon fiber variants
    {"pk": "HISTORICAL", "sk": "carbon_fiber#medium#sedan", "sample_size": d(18), "date_range": "2023-2025", "min_cost": d(120000), "max_cost": d(180000), "avg_cost": d(148000), "median_cost": d(145000), "p25_cost": d(135000), "p75_cost": d(160000), "std_dev": d(15000), "material_pct": d(35), "stamping_pct": d(15), "tooling_pct": d(30), "assembly_pct": d(20), "ci_95": "137000-159000"},
    {"pk": "HISTORICAL", "sk": "carbon_fiber#high#coupe", "sample_size": d(12), "date_range": "2023-2025", "min_cost": d(160000), "max_cost": d(240000), "avg_cost": d(195000), "median_cost": d(190000), "p25_cost": d(175000), "p75_cost": d(215000), "std_dev": d(22000), "material_pct": d(38), "stamping_pct": d(12), "tooling_pct": d(28), "assembly_pct": d(22), "ci_95": "178000-212000"},
]


REGIONAL = [
    {"pk": "REGIONAL", "sk": "north_america", "labor_multiplier": d(1.0), "energy_multiplier": d(0.95), "logistics_multiplier": d(1.0), "overhead_multiplier": d(1.0), "composite_multiplier": d(0.99), "currency": "USD", "effective_date": "2026-01-01", "notes": "Baseline region. Competitive energy costs from natural gas."},
    {"pk": "REGIONAL", "sk": "europe", "labor_multiplier": d(1.25), "energy_multiplier": d(1.35), "logistics_multiplier": d(1.10), "overhead_multiplier": d(1.15), "composite_multiplier": d(1.21), "currency": "USD", "effective_date": "2026-01-01", "notes": "Higher labor and energy costs. Strong environmental regulations add overhead."},
    {"pk": "REGIONAL", "sk": "asia_pacific", "labor_multiplier": d(0.55), "energy_multiplier": d(0.80), "logistics_multiplier": d(1.20), "overhead_multiplier": d(0.85), "composite_multiplier": d(0.78), "currency": "USD", "effective_date": "2026-01-01", "notes": "Low labor costs offset by higher logistics for export."},
    {"pk": "REGIONAL", "sk": "south_america", "labor_multiplier": d(0.65), "energy_multiplier": d(0.90), "logistics_multiplier": d(1.30), "overhead_multiplier": d(1.05), "composite_multiplier": d(0.92), "currency": "USD", "effective_date": "2026-01-01", "notes": "Growing automotive sector. Import tariffs on raw materials."},
    {"pk": "REGIONAL", "sk": "middle_east", "labor_multiplier": d(0.70), "energy_multiplier": d(0.60), "logistics_multiplier": d(1.25), "overhead_multiplier": d(1.10), "composite_multiplier": d(0.86), "currency": "USD", "effective_date": "2026-01-01", "notes": "Very low energy costs. Limited local supplier base increases logistics."},
]

VOLUME_DISCOUNTS = [
    # Steel tiers
    {"pk": "VOLUME_DISCOUNT", "sk": "steel#tier_1", "tier_name": "Spot", "min_volume_kg": d(0), "max_volume_kg": d(5000), "discount_pct": d(0), "price_per_kg": d(1.25), "min_order_kg": d(100)},
    {"pk": "VOLUME_DISCOUNT", "sk": "steel#tier_2", "tier_name": "Small Batch", "min_volume_kg": d(5000), "max_volume_kg": d(25000), "discount_pct": d(5), "price_per_kg": d(1.19), "min_order_kg": d(1000)},
    {"pk": "VOLUME_DISCOUNT", "sk": "steel#tier_3", "tier_name": "Production", "min_volume_kg": d(25000), "max_volume_kg": d(100000), "discount_pct": d(12), "price_per_kg": d(1.10), "min_order_kg": d(5000)},
    {"pk": "VOLUME_DISCOUNT", "sk": "steel#tier_4", "tier_name": "High Volume", "min_volume_kg": d(100000), "max_volume_kg": d(999999), "discount_pct": d(18), "price_per_kg": d(1.03), "min_order_kg": d(10000)},
    # Aluminum tiers
    {"pk": "VOLUME_DISCOUNT", "sk": "aluminum#tier_1", "tier_name": "Spot", "min_volume_kg": d(0), "max_volume_kg": d(2000), "discount_pct": d(0), "price_per_kg": d(3.65), "min_order_kg": d(50)},
    {"pk": "VOLUME_DISCOUNT", "sk": "aluminum#tier_2", "tier_name": "Small Batch", "min_volume_kg": d(2000), "max_volume_kg": d(10000), "discount_pct": d(4), "price_per_kg": d(3.50), "min_order_kg": d(500)},
    {"pk": "VOLUME_DISCOUNT", "sk": "aluminum#tier_3", "tier_name": "Production", "min_volume_kg": d(10000), "max_volume_kg": d(50000), "discount_pct": d(10), "price_per_kg": d(3.29), "min_order_kg": d(2000)},
    {"pk": "VOLUME_DISCOUNT", "sk": "aluminum#tier_4", "tier_name": "High Volume", "min_volume_kg": d(50000), "max_volume_kg": d(999999), "discount_pct": d(15), "price_per_kg": d(3.10), "min_order_kg": d(5000)},
    # Carbon fiber tiers
    {"pk": "VOLUME_DISCOUNT", "sk": "carbon_fiber#tier_1", "tier_name": "Prototype", "min_volume_kg": d(0), "max_volume_kg": d(500), "discount_pct": d(0), "price_per_kg": d(25.00), "min_order_kg": d(10)},
    {"pk": "VOLUME_DISCOUNT", "sk": "carbon_fiber#tier_2", "tier_name": "Small Batch", "min_volume_kg": d(500), "max_volume_kg": d(2000), "discount_pct": d(8), "price_per_kg": d(23.00), "min_order_kg": d(100)},
    {"pk": "VOLUME_DISCOUNT", "sk": "carbon_fiber#tier_3", "tier_name": "Production", "min_volume_kg": d(2000), "max_volume_kg": d(10000), "discount_pct": d(15), "price_per_kg": d(21.25), "min_order_kg": d(500)},
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

    all_items = MARKET_PRICES + SUPPLIERS + HISTORICAL + REGIONAL + VOLUME_DISCOUNTS
    logger.info(f"Seeding {len(all_items)} items into '{table_name}'...")

    with table.batch_writer() as batch:
        for item in all_items:
            batch.put_item(Item=item)

    logger.info(f"✅ Seeded {len(all_items)} items successfully")

    # Print summary
    categories = {}
    for item in all_items:
        cat = item["pk"]
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        logger.info(f"  {cat}: {count} items")


def main():
    parser = argparse.ArgumentParser(description="Seed CarDesignExternalCostData table")
    parser.add_argument("--table", default="CarDesignExternalCostData", help="DynamoDB table name")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--create-table", action="store_true", help="Create table if not exists")
    args = parser.parse_args()

    if args.create_table:
        create_table_if_not_exists(args.table, args.region)

    seed_data(args.table, args.region)


if __name__ == "__main__":
    main()
