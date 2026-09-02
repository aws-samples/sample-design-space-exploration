#!/usr/bin/env python3
"""
Cost Data MCP Server — DynamoDB-backed external cost data source.

Deployed as a standalone HTTP service. The Cost Agent connects to this
server via Strands MCPClient using Streamable HTTP transport.

Simulates an external enterprise system (commodity pricing feed, supplier ERP,
historical project database) backed by a dedicated DynamoDB table:
  Table: CarDesignExternalCostData
  pk: Category — MARKET_PRICE, SUPPLIER, HISTORICAL, REGIONAL, VOLUME_DISCOUNT
  sk: Specific item key

Run locally:  python cost_data_server.py
Deploy:       Lambda behind API Gateway, or ECS Fargate service

Tools:
- get_material_market_price: Live commodity pricing with regional variants
- get_supplier_quotes: Component-level supplier quotes with lead times
- get_historical_cost_benchmarks: Statistical cost distributions
- get_regional_cost_factors: Regional manufacturing cost multipliers
- get_volume_discount_schedule: Volume-based pricing tiers
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Key
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
EXTERNAL_COST_TABLE = os.environ.get("EXTERNAL_COST_TABLE", "CarDesignExternalCostData")

mcp = FastMCP(
    "Cost Data Server",
    description=(
        "DynamoDB-backed external cost data source for the Car Design Space Explorer. "
        "Provides live market pricing, supplier quotes, historical benchmarks, "
        "regional cost factors, and volume discount schedules."
    ),
)


# ---------------------------------------------------------------------------
# DynamoDB helpers
# ---------------------------------------------------------------------------

def _get_table():
    """Get DynamoDB table resource."""
    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    return dynamodb.Table(EXTERNAL_COST_TABLE)


def _decimal_to_float(obj):
    """Recursively convert Decimal values to float for JSON serialization."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _decimal_to_float(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decimal_to_float(i) for i in obj]
    return obj


def _query_items(pk: str, sk_prefix: str | None = None) -> list[dict]:
    """Query items from the external cost data table."""
    try:
        table = _get_table()
        if sk_prefix:
            response = table.query(
                KeyConditionExpression=(
                    Key("pk").eq(pk) & Key("sk").begins_with(sk_prefix)
                )
            )
        else:
            response = table.query(
                KeyConditionExpression=Key("pk").eq(pk)
            )
        return [_decimal_to_float(item) for item in response.get("Items", [])]
    except Exception as e:
        logger.error(f"DynamoDB query failed (pk={pk}): {e}")
        return []


def _get_item(pk: str, sk: str) -> dict | None:
    """Get a single item from the external cost data table."""
    try:
        table = _get_table()
        response = table.get_item(Key={"pk": pk, "sk": sk})
        item = response.get("Item")
        return _decimal_to_float(item) if item else None
    except Exception as e:
        logger.error(f"DynamoDB get_item failed (pk={pk}, sk={sk}): {e}")
        return None


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_material_market_price(material: str, region: str = "north_america") -> str:
    """Retrieve current commodity market pricing for a material.

    Queries the external pricing database for the latest market price,
    including regional adjustments, price trends, and supplier source.

    Args:
        material: Material type — "steel", "aluminum", "carbon_fiber",
            "high_strength_steel", "magnesium", "titanium".
        region: Manufacturing region — "north_america", "europe", "asia_pacific".

    Returns:
        JSON with price_per_kg, currency, trend, source, effective_date.
    """
    # Try region-specific price first
    item = _get_item("MARKET_PRICE", f"{material}#{region}")
    if not item:
        # Fall back to global price
        item = _get_item("MARKET_PRICE", material)
    if not item:
        return json.dumps({
            "status": "error",
            "error_message": f"No market price data for material='{material}', region='{region}'",
        })

    return json.dumps({
        "material": material,
        "region": region,
        "price_per_kg": item.get("price_per_kg", 0),
        "currency": item.get("currency", "USD"),
        "trend": item.get("trend", "stable"),
        "trend_pct_30d": item.get("trend_pct_30d", 0),
        "source": item.get("source", "market_index"),
        "effective_date": item.get("effective_date", ""),
        "grade": item.get("grade", "standard"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "success",
    }, indent=2)


@mcp.tool()
def get_supplier_quotes(
    component_type: str,
    complexity: str = "medium",
    quantity: int = 1,
) -> str:
    """Query supplier pricing for manufacturing components and tooling.

    Returns quotes from multiple suppliers with lead times, volume
    adjustments, and availability status.

    Args:
        component_type: Component — "stamping_die", "welding_fixture",
            "paint_booth_time", "assembly_jig", "trim_die", "inspection_fixture".
        complexity: Geometry complexity — "low", "medium", "high".
        quantity: Number of units (affects volume pricing).

    Returns:
        JSON with supplier quotes, lead times, and adjusted pricing.
    """
    items = _query_items("SUPPLIER", f"{component_type}#")
    if not items:
        # Try without complexity suffix
        item = _get_item("SUPPLIER", component_type)
        items = [item] if item else []

    if not items:
        return json.dumps({
            "status": "error",
            "error_message": f"No supplier data for component='{component_type}'",
        })

    complexity_mult = {"low": 0.8, "medium": 1.0, "high": 1.4}.get(complexity, 1.0)
    # Volume discount: 5% off per 10 units, max 25% off
    volume_discount = min(0.25, (quantity // 10) * 0.05)

    quotes = []
    for item in items:
        base_price = item.get("base_price", 0)
        adjusted = round(base_price * complexity_mult * (1 - volume_discount), 2)
        quotes.append({
            "supplier": item.get("supplier_name", "unknown"),
            "supplier_id": item.get("supplier_id", ""),
            "base_price": base_price,
            "adjusted_price": adjusted,
            "currency": item.get("currency", "USD"),
            "lead_time_weeks": item.get("lead_time_weeks", 0),
            "availability": item.get("availability", "available"),
            "warranty_months": item.get("warranty_months", 12),
            "rating": item.get("rating", 0),
        })

    return json.dumps({
        "component_type": component_type,
        "complexity": complexity,
        "quantity": quantity,
        "volume_discount_pct": round(volume_discount * 100, 1),
        "quotes": quotes,
        "quote_count": len(quotes),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "success",
    }, indent=2)


@mcp.tool()
def get_historical_cost_benchmarks(
    material: str = "steel",
    complexity: str = "medium",
    body_style: str = "sedan",
) -> str:
    """Retrieve historical manufacturing cost benchmarks for similar designs.

    Returns statistical distributions (min, max, avg, percentiles) from
    past car body manufacturing projects, segmented by material, complexity,
    and body style.

    Args:
        material: Material type — "steel", "aluminum", "carbon_fiber".
        complexity: Complexity level — "low", "medium", "high".
        body_style: Vehicle body style — "sedan", "suv", "coupe", "hatchback", "truck".

    Returns:
        JSON with cost statistics, sample size, and confidence interval.
    """
    sk = f"{material}#{complexity}#{body_style}"
    item = _get_item("HISTORICAL", sk)
    if not item:
        # Try without body_style
        item = _get_item("HISTORICAL", f"{material}#{complexity}")
    if not item:
        return json.dumps({
            "status": "error",
            "error_message": f"No historical data for material='{material}', complexity='{complexity}', body_style='{body_style}'",
        })

    return json.dumps({
        "material": material,
        "complexity": complexity,
        "body_style": body_style,
        "sample_size": item.get("sample_size", 0),
        "date_range": item.get("date_range", ""),
        "statistics": {
            "min_total_cost": item.get("min_cost", 0),
            "max_total_cost": item.get("max_cost", 0),
            "avg_total_cost": item.get("avg_cost", 0),
            "median_total_cost": item.get("median_cost", 0),
            "p25_total_cost": item.get("p25_cost", 0),
            "p75_total_cost": item.get("p75_cost", 0),
            "std_dev": item.get("std_dev", 0),
        },
        "cost_breakdown_avg": {
            "material_pct": item.get("material_pct", 0),
            "stamping_pct": item.get("stamping_pct", 0),
            "tooling_pct": item.get("tooling_pct", 0),
            "assembly_pct": item.get("assembly_pct", 0),
        },
        "confidence_interval_95": item.get("ci_95", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "success",
    }, indent=2)


@mcp.tool()
def get_regional_cost_factors(region: str = "north_america") -> str:
    """Get regional manufacturing cost multipliers.

    Different regions have different labor rates, energy costs, and
    logistics overhead that affect total manufacturing cost.

    Args:
        region: Manufacturing region — "north_america", "europe",
            "asia_pacific", "south_america", "middle_east".

    Returns:
        JSON with labor, energy, logistics, and overhead multipliers.
    """
    item = _get_item("REGIONAL", region)
    if not item:
        return json.dumps({
            "status": "error",
            "error_message": f"No regional data for region='{region}'",
        })

    return json.dumps({
        "region": region,
        "labor_multiplier": item.get("labor_multiplier", 1.0),
        "energy_multiplier": item.get("energy_multiplier", 1.0),
        "logistics_multiplier": item.get("logistics_multiplier", 1.0),
        "overhead_multiplier": item.get("overhead_multiplier", 1.0),
        "composite_multiplier": item.get("composite_multiplier", 1.0),
        "currency": item.get("currency", "USD"),
        "effective_date": item.get("effective_date", ""),
        "notes": item.get("notes", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "success",
    }, indent=2)


@mcp.tool()
def get_volume_discount_schedule(material: str = "steel") -> str:
    """Get volume-based pricing tiers for material procurement.

    Returns discount schedules based on annual procurement volume,
    helping the Cost Agent factor in economies of scale.

    Args:
        material: Material type — "steel", "aluminum", "carbon_fiber".

    Returns:
        JSON with volume tiers, discount percentages, and minimum order quantities.
    """
    items = _query_items("VOLUME_DISCOUNT", f"{material}#")
    if not items:
        return json.dumps({
            "status": "error",
            "error_message": f"No volume discount data for material='{material}'",
        })

    tiers = []
    for item in sorted(items, key=lambda x: x.get("min_volume_kg", 0)):
        tiers.append({
            "tier_name": item.get("tier_name", ""),
            "min_volume_kg": item.get("min_volume_kg", 0),
            "max_volume_kg": item.get("max_volume_kg", 0),
            "discount_pct": item.get("discount_pct", 0),
            "price_per_kg": item.get("price_per_kg", 0),
            "min_order_kg": item.get("min_order_kg", 0),
        })

    return json.dumps({
        "material": material,
        "tiers": tiers,
        "tier_count": len(tiers),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "success",
    }, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    transport = "streamable-http"
    port = int(os.environ.get("MCP_SERVER_PORT", "8100"))

    if "--stdio" in sys.argv:
        transport = "stdio"

    logger.info(f"Starting Cost Data MCP Server (transport={transport}, port={port})")
    mcp.run(transport=transport, port=port)
