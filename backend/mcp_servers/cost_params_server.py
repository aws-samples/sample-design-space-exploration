#!/usr/bin/env python3
"""
Cost Parameters MCP Server — DynamoDB-backed internal cost parameters.

Wraps the CarDesignCostParameters DynamoDB table as an MCP server,
providing the Cost Agent with base manufacturing cost parameters
(material costs, stamping rates, tooling costs, assembly costs,
complexity multipliers) via the MCP protocol.

This is the "internal" data source. The Cost Agent also connects to
the external Cost Data MCP Server for market pricing, supplier quotes,
and historical benchmarks.

DynamoDB Table: CarDesignCostParameters
  pk: Category — MATERIAL, STAMPING, TOOLING, ASSEMBLY, MULTIPLIER
  sk: Specific item key

Run locally:  python cost_params_server.py
Deploy:       Lambda behind API Gateway, or ECS Fargate service
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
COST_TABLE_NAME = os.environ.get("COST_TABLE_NAME", "CarDesignCostParameters")

mcp = FastMCP(
    "Cost Parameters Server",
    description=(
        "Internal cost parameters for the Car Design Space Explorer. "
        "Provides base material costs, stamping rates, tooling costs, "
        "assembly costs, complexity multipliers, surface treatment costs, "
        "quality inspection rates, and logistics parameters from DynamoDB."
    ),
)


# ---------------------------------------------------------------------------
# DynamoDB helpers
# ---------------------------------------------------------------------------

def _get_table():
    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    return dynamodb.Table(COST_TABLE_NAME)


def _decimal_to_float(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _decimal_to_float(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decimal_to_float(i) for i in obj]
    return obj


def _query_items(pk: str) -> list[dict]:
    try:
        table = _get_table()
        response = table.query(KeyConditionExpression=Key("pk").eq(pk))
        return [_decimal_to_float(item) for item in response.get("Items", [])]
    except Exception as e:
        logger.error(f"DynamoDB query failed (pk={pk}): {e}")
        return []


def _get_item(pk: str, sk: str) -> dict | None:
    try:
        table = _get_table()
        response = table.get_item(Key={"pk": pk, "sk": sk})
        item = response.get("Item")
        return _decimal_to_float(item) if item else None
    except Exception as e:
        logger.error(f"DynamoDB get_item failed: {e}")
        return None


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_material_cost(material: str) -> str:
    """Get the base manufacturing cost per kilogram for a material.

    Args:
        material: Material type — "steel", "aluminum", or "carbon_fiber".

    Returns:
        JSON with cost_per_kg, unit, and description.
    """
    item = _get_item("MATERIAL", material)
    if not item:
        return json.dumps({
            "status": "error",
            "error_message": f"No cost data for material '{material}'",
            "available": [i.get("sk") for i in _query_items("MATERIAL")],
        })
    return json.dumps({
        "material": material,
        "cost_per_kg": item.get("value", 0),
        "unit": item.get("unit", "USD/kg"),
        "description": item.get("description", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "success",
    }, indent=2)


@mcp.tool()
def get_all_material_costs() -> str:
    """Get base manufacturing costs for all available materials.

    Returns:
        JSON with material costs for steel, aluminum, carbon_fiber, etc.
    """
    items = _query_items("MATERIAL")
    if not items:
        return json.dumps({"status": "error", "error_message": "No material cost data found"})

    materials = {}
    for item in items:
        materials[item.get("sk", "")] = {
            "cost_per_kg": item.get("value", 0),
            "unit": item.get("unit", "USD/kg"),
            "description": item.get("description", ""),
        }
    return json.dumps({
        "materials": materials,
        "count": len(materials),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "success",
    }, indent=2)


@mcp.tool()
def get_stamping_parameters() -> str:
    """Get stamping cost parameters (cost per operation, multi-stage multiplier).

    Returns:
        JSON with stamping_cost_per_op and multi_stage_multiplier.
    """
    items = _query_items("STAMPING")
    if not items:
        return json.dumps({"status": "error", "error_message": "No stamping data found"})

    params = {}
    for item in items:
        params[item.get("sk", "")] = {
            "value": item.get("value", 0),
            "unit": item.get("unit", ""),
            "description": item.get("description", ""),
        }
    return json.dumps({
        "stamping_parameters": params,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "success",
    }, indent=2)


@mcp.tool()
def get_tooling_base_cost() -> str:
    """Get all tooling cost parameters (base die cost, trim die, flange die, fixtures, maintenance).

    Returns:
        JSON with all tooling cost parameters.
    """
    items = _query_items("TOOLING")
    if not items:
        return json.dumps({"status": "error", "error_message": "No tooling cost data found"})

    params = {}
    for item in items:
        params[item.get("sk", "")] = {
            "value": item.get("value", 0),
            "unit": item.get("unit", "USD"),
            "description": item.get("description", ""),
        }
    return json.dumps({
        "tooling_parameters": params,
        "tooling_base_cost": next(
            (item.get("value", 0) for item in items if item.get("sk") == "base_cost"), 50000.0
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "success",
    }, indent=2)


@mcp.tool()
def get_assembly_parameters() -> str:
    """Get assembly cost parameters (welding cost per meter, etc.).

    Returns:
        JSON with assembly cost rates.
    """
    items = _query_items("ASSEMBLY")
    if not items:
        return json.dumps({"status": "error", "error_message": "No assembly data found"})

    params = {}
    for item in items:
        params[item.get("sk", "")] = {
            "value": item.get("value", 0),
            "unit": item.get("unit", ""),
            "description": item.get("description", ""),
        }
    return json.dumps({
        "assembly_parameters": params,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "success",
    }, indent=2)


@mcp.tool()
def get_complexity_multipliers() -> str:
    """Get complexity-based cost multipliers (low, medium, high).

    Returns:
        JSON with multiplier values for each complexity level.
    """
    items = _query_items("MULTIPLIER")
    if not items:
        return json.dumps({"status": "error", "error_message": "No multiplier data found"})

    multipliers = {}
    for item in items:
        multipliers[item.get("sk", "")] = {
            "value": item.get("value", 1.0),
            "unit": item.get("unit", "multiplier"),
            "description": item.get("description", ""),
        }
    return json.dumps({
        "complexity_multipliers": multipliers,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "success",
    }, indent=2)


@mcp.tool()
def get_surface_treatment_costs() -> str:
    """Get surface treatment cost parameters (e-coat, primer, paint, galvanizing, anodizing).

    Returns:
        JSON with per-m² costs for each surface treatment stage.
    """
    items = _query_items("SURFACE_TREATMENT")
    if not items:
        return json.dumps({"status": "error", "error_message": "No surface treatment data found"})

    params = {}
    for item in items:
        params[item.get("sk", "")] = {
            "value": item.get("value", 0),
            "unit": item.get("unit", ""),
            "description": item.get("description", ""),
        }
    return json.dumps({
        "surface_treatment_parameters": params,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "success",
    }, indent=2)


@mcp.tool()
def get_quality_parameters() -> str:
    """Get quality inspection and rework cost parameters.

    Returns:
        JSON with CMM inspection, visual inspection, rework rates, and scrap targets.
    """
    items = _query_items("QUALITY")
    if not items:
        return json.dumps({"status": "error", "error_message": "No quality data found"})

    params = {}
    for item in items:
        params[item.get("sk", "")] = {
            "value": item.get("value", 0),
            "unit": item.get("unit", ""),
            "description": item.get("description", ""),
        }
    return json.dumps({
        "quality_parameters": params,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "success",
    }, indent=2)


@mcp.tool()
def get_logistics_parameters() -> str:
    """Get logistics and overhead cost parameters (transport, packaging, warehousing, energy).

    Returns:
        JSON with intra-plant transport, packaging, storage, and energy costs.
    """
    items = _query_items("LOGISTICS")
    if not items:
        return json.dumps({"status": "error", "error_message": "No logistics data found"})

    params = {}
    for item in items:
        params[item.get("sk", "")] = {
            "value": item.get("value", 0),
            "unit": item.get("unit", ""),
            "description": item.get("description", ""),
        }
    return json.dumps({
        "logistics_parameters": params,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "success",
    }, indent=2)


@mcp.tool()
def get_all_cost_parameters() -> str:
    """Get all cost parameters in a single call.

    Retrieves material costs, stamping parameters, tooling costs,
    assembly parameters, complexity multipliers, surface treatment costs,
    quality parameters, and logistics costs in one response.

    Returns:
        JSON with complete cost parameter set (all 8 categories).
    """
    result = {
        "materials": {},
        "stamping": {},
        "tooling": {},
        "assembly": {},
        "multipliers": {},
        "surface_treatment": {},
        "quality": {},
        "logistics": {},
    }

    for item in _query_items("MATERIAL"):
        result["materials"][item.get("sk", "")] = item.get("value", 0)

    for item in _query_items("STAMPING"):
        result["stamping"][item.get("sk", "")] = item.get("value", 0)

    for item in _query_items("TOOLING"):
        result["tooling"][item.get("sk", "")] = item.get("value", 0)

    for item in _query_items("ASSEMBLY"):
        result["assembly"][item.get("sk", "")] = item.get("value", 0)

    for item in _query_items("MULTIPLIER"):
        result["multipliers"][item.get("sk", "")] = item.get("value", 1.0)

    for item in _query_items("SURFACE_TREATMENT"):
        result["surface_treatment"][item.get("sk", "")] = item.get("value", 0)

    for item in _query_items("QUALITY"):
        result["quality"][item.get("sk", "")] = item.get("value", 0)

    for item in _query_items("LOGISTICS"):
        result["logistics"][item.get("sk", "")] = item.get("value", 0)

    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    result["status"] = "success"
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    transport = "streamable-http"
    port = int(os.environ.get("COST_PARAMS_MCP_PORT", "8101"))

    if "--stdio" in sys.argv:
        transport = "stdio"

    logger.info(f"Starting Cost Parameters MCP Server (transport={transport}, port={port})")
    mcp.run(transport=transport, port=port)
