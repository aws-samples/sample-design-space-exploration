#!/usr/bin/env python3
"""
Lambda MCP Handler for Car Design Cost Data.

Single Lambda function that serves as the MCP target behind the AgentCore
Gateway. Handles tool calls from both the Internal Cost Parameters and
External Cost Data MCP servers by routing based on tool name.

The AgentCore Gateway invokes this Lambda with MCP tool call payloads.
The Lambda reads from two DynamoDB tables:
  - CarDesignCostParameters (internal cost params)
  - CarDesignExternalCostData (external market data)

Deployed as a standard Lambda function, registered as a gateway target.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Key

logger = logging.getLogger()
logger.setLevel(logging.INFO)

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
COST_TABLE = os.environ.get("COST_TABLE_NAME", "CarDesignCostParameters")
EXTERNAL_TABLE = os.environ.get("EXTERNAL_COST_TABLE", "CarDesignExternalCostData")

_dynamodb = None


def _get_dynamodb():
    global _dynamodb
    if _dynamodb is None:
        _dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    return _dynamodb


def _decimal_to_float(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _decimal_to_float(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decimal_to_float(i) for i in obj]
    return obj


def _query(table_name: str, pk: str, sk_prefix: str | None = None) -> list[dict]:
    try:
        table = _get_dynamodb().Table(table_name)
        if sk_prefix:
            resp = table.query(
                KeyConditionExpression=Key("pk").eq(pk) & Key("sk").begins_with(sk_prefix)
            )
        else:
            resp = table.query(KeyConditionExpression=Key("pk").eq(pk))
        return [_decimal_to_float(i) for i in resp.get("Items", [])]
    except Exception as e:
        logger.error(f"DynamoDB query failed: {e}")
        return []


def _get_item(table_name: str, pk: str, sk: str) -> dict | None:
    try:
        table = _get_dynamodb().Table(table_name)
        resp = table.get_item(Key={"pk": pk, "sk": sk})
        item = resp.get("Item")
        return _decimal_to_float(item) if item else None
    except Exception as e:
        logger.error(f"DynamoDB get_item failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Internal Cost Parameters tools
# ---------------------------------------------------------------------------

def get_material_cost(args: dict) -> dict:
    material = args.get("material", "steel")
    item = _get_item(COST_TABLE, "MATERIAL", material)
    if not item:
        return {"status": "error", "error_message": f"No cost data for '{material}'"}
    return {
        "material": material,
        "cost_per_kg": item.get("value", 0),
        "unit": item.get("unit", "USD/kg"),
        "description": item.get("description", ""),
        "status": "success",
    }


def get_all_material_costs(args: dict) -> dict:
    items = _query(COST_TABLE, "MATERIAL")
    materials = {}
    for item in items:
        materials[item.get("sk", "")] = {
            "cost_per_kg": item.get("value", 0),
            "unit": item.get("unit", "USD/kg"),
            "description": item.get("description", ""),
        }
    return {"materials": materials, "count": len(materials), "status": "success"}


def get_stamping_parameters(args: dict) -> dict:
    items = _query(COST_TABLE, "STAMPING")
    params = {}
    for item in items:
        params[item.get("sk", "")] = {
            "value": item.get("value", 0),
            "unit": item.get("unit", ""),
            "description": item.get("description", ""),
        }
    return {"stamping_parameters": params, "status": "success"}


def get_tooling_base_cost(args: dict) -> dict:
    items = _query(COST_TABLE, "TOOLING")
    params = {}
    for item in items:
        params[item.get("sk", "")] = {
            "value": item.get("value", 0),
            "unit": item.get("unit", "USD"),
            "description": item.get("description", ""),
        }
    base = next((i.get("value", 50000.0) for i in items if i.get("sk") == "base_cost"), 50000.0)
    return {"tooling_parameters": params, "tooling_base_cost": base, "status": "success"}


def get_assembly_parameters(args: dict) -> dict:
    items = _query(COST_TABLE, "ASSEMBLY")
    params = {}
    for item in items:
        params[item.get("sk", "")] = {
            "value": item.get("value", 0),
            "unit": item.get("unit", ""),
            "description": item.get("description", ""),
        }
    return {"assembly_parameters": params, "status": "success"}


def get_complexity_multipliers(args: dict) -> dict:
    items = _query(COST_TABLE, "MULTIPLIER")
    mults = {}
    for item in items:
        mults[item.get("sk", "")] = {
            "value": item.get("value", 1.0),
            "description": item.get("description", ""),
        }
    return {"complexity_multipliers": mults, "status": "success"}


def get_surface_treatment_costs(args: dict) -> dict:
    items = _query(COST_TABLE, "SURFACE_TREATMENT")
    params = {}
    for item in items:
        params[item.get("sk", "")] = {
            "value": item.get("value", 0),
            "unit": item.get("unit", ""),
            "description": item.get("description", ""),
        }
    return {"surface_treatment_parameters": params, "status": "success"}


def get_quality_parameters(args: dict) -> dict:
    items = _query(COST_TABLE, "QUALITY")
    params = {}
    for item in items:
        params[item.get("sk", "")] = {
            "value": item.get("value", 0),
            "unit": item.get("unit", ""),
            "description": item.get("description", ""),
        }
    return {"quality_parameters": params, "status": "success"}


def get_logistics_parameters(args: dict) -> dict:
    items = _query(COST_TABLE, "LOGISTICS")
    params = {}
    for item in items:
        params[item.get("sk", "")] = {
            "value": item.get("value", 0),
            "unit": item.get("unit", ""),
            "description": item.get("description", ""),
        }
    return {"logistics_parameters": params, "status": "success"}


def get_all_cost_parameters(args: dict) -> dict:
    result = {
        "materials": {}, "stamping": {}, "tooling": {}, "assembly": {},
        "multipliers": {}, "surface_treatment": {}, "quality": {}, "logistics": {},
    }
    for item in _query(COST_TABLE, "MATERIAL"):
        result["materials"][item.get("sk", "")] = item.get("value", 0)
    for item in _query(COST_TABLE, "STAMPING"):
        result["stamping"][item.get("sk", "")] = item.get("value", 0)
    for item in _query(COST_TABLE, "TOOLING"):
        result["tooling"][item.get("sk", "")] = item.get("value", 0)
    for item in _query(COST_TABLE, "ASSEMBLY"):
        result["assembly"][item.get("sk", "")] = item.get("value", 0)
    for item in _query(COST_TABLE, "MULTIPLIER"):
        result["multipliers"][item.get("sk", "")] = item.get("value", 1.0)
    for item in _query(COST_TABLE, "SURFACE_TREATMENT"):
        result["surface_treatment"][item.get("sk", "")] = item.get("value", 0)
    for item in _query(COST_TABLE, "QUALITY"):
        result["quality"][item.get("sk", "")] = item.get("value", 0)
    for item in _query(COST_TABLE, "LOGISTICS"):
        result["logistics"][item.get("sk", "")] = item.get("value", 0)
    result["status"] = "success"
    return result


# ---------------------------------------------------------------------------
# External Cost Data tools
# ---------------------------------------------------------------------------

def get_material_market_price(args: dict) -> dict:
    material = args.get("material", "steel")
    region = args.get("region", "north_america")
    item = _get_item(EXTERNAL_TABLE, "MARKET_PRICE", f"{material}#{region}")
    if not item:
        item = _get_item(EXTERNAL_TABLE, "MARKET_PRICE", material)
    if not item:
        return {"status": "error", "error_message": f"No market price for '{material}' in '{region}'"}
    return {
        "material": material, "region": region,
        "price_per_kg": item.get("price_per_kg", 0),
        "currency": item.get("currency", "USD"),
        "trend": item.get("trend", "stable"),
        "source": item.get("source", "market_index"),
        "status": "success",
    }


def get_supplier_quotes(args: dict) -> dict:
    comp = args.get("component_type", "stamping_die")
    complexity = args.get("complexity", "medium")
    quantity = int(args.get("quantity", 1))
    items = _query(EXTERNAL_TABLE, "SUPPLIER", f"{comp}#")
    if not items:
        item = _get_item(EXTERNAL_TABLE, "SUPPLIER", comp)
        items = [item] if item else []
    if not items:
        return {"status": "error", "error_message": f"No supplier data for '{comp}'"}
    cmult = {"low": 0.8, "medium": 1.0, "high": 1.4}.get(complexity, 1.0)
    vdisc = min(0.25, (quantity // 10) * 0.05)
    quotes = []
    for item in items:
        bp = item.get("base_price", 0)
        quotes.append({
            "supplier": item.get("supplier_name", "unknown"),
            "base_price": bp,
            "adjusted_price": round(bp * cmult * (1 - vdisc), 2),
            "lead_time_weeks": item.get("lead_time_weeks", 0),
            "availability": item.get("availability", "available"),
        })
    return {"component_type": comp, "quotes": quotes, "status": "success"}


def get_historical_cost_benchmarks(args: dict) -> dict:
    material = args.get("material", "steel")
    complexity = args.get("complexity", "medium")
    body_style = args.get("body_style", "sedan")
    item = _get_item(EXTERNAL_TABLE, "HISTORICAL", f"{material}#{complexity}#{body_style}")
    if not item:
        item = _get_item(EXTERNAL_TABLE, "HISTORICAL", f"{material}#{complexity}")
    if not item:
        return {"status": "error", "error_message": "No historical data found"}
    return {
        "material": material, "complexity": complexity, "body_style": body_style,
        "sample_size": item.get("sample_size", 0),
        "statistics": {
            "min_total_cost": item.get("min_cost", 0),
            "max_total_cost": item.get("max_cost", 0),
            "avg_total_cost": item.get("avg_cost", 0),
        },
        "status": "success",
    }


def get_regional_cost_factors(args: dict) -> dict:
    region = args.get("region", "north_america")
    item = _get_item(EXTERNAL_TABLE, "REGIONAL", region)
    if not item:
        return {"status": "error", "error_message": f"No regional data for '{region}'"}
    return {
        "region": region,
        "labor_multiplier": item.get("labor_multiplier", 1.0),
        "energy_multiplier": item.get("energy_multiplier", 1.0),
        "logistics_multiplier": item.get("logistics_multiplier", 1.0),
        "composite_multiplier": item.get("composite_multiplier", 1.0),
        "status": "success",
    }


def get_volume_discount_schedule(args: dict) -> dict:
    material = args.get("material", "steel")
    items = _query(EXTERNAL_TABLE, "VOLUME_DISCOUNT", f"{material}#")
    if not items:
        return {"status": "error", "error_message": f"No volume discount data for '{material}'"}
    tiers = sorted([{
        "tier_name": i.get("tier_name", ""),
        "min_volume_kg": i.get("min_volume_kg", 0),
        "max_volume_kg": i.get("max_volume_kg", 0),
        "discount_pct": i.get("discount_pct", 0),
        "price_per_kg": i.get("price_per_kg", 0),
    } for i in items], key=lambda x: x["min_volume_kg"])
    return {"material": material, "tiers": tiers, "status": "success"}


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "get_material_cost": get_material_cost,
    "get_all_material_costs": get_all_material_costs,
    "get_stamping_parameters": get_stamping_parameters,
    "get_tooling_base_cost": get_tooling_base_cost,
    "get_assembly_parameters": get_assembly_parameters,
    "get_complexity_multipliers": get_complexity_multipliers,
    "get_surface_treatment_costs": get_surface_treatment_costs,
    "get_quality_parameters": get_quality_parameters,
    "get_logistics_parameters": get_logistics_parameters,
    "get_all_cost_parameters": get_all_cost_parameters,
    "get_material_market_price": get_material_market_price,
    "get_supplier_quotes": get_supplier_quotes,
    "get_historical_cost_benchmarks": get_historical_cost_benchmarks,
    "get_regional_cost_factors": get_regional_cost_factors,
    "get_volume_discount_schedule": get_volume_discount_schedule,
}


def handler(event, context):
    """Lambda handler for AgentCore Gateway MCP tool calls.

    The Gateway sends:
      - event: a dict of input properties (the tool arguments directly)
      - context.client_context.custom['bedrockAgentCoreToolName']:
            the tool name in format "targetName___toolName"
    """
    logger.info(f"Event: {json.dumps(event)[:500]}")

    # Extract tool name from Lambda context (set by AgentCore Gateway)
    tool_name = ""
    try:
        custom = context.client_context.custom or {}
        raw_tool_name = custom.get("bedrockAgentCoreToolName", "")
        # Strip target name prefix: "targetName___toolName" → "toolName"
        delimiter = "___"
        if delimiter in raw_tool_name:
            tool_name = raw_tool_name[raw_tool_name.index(delimiter) + len(delimiter):]
        else:
            tool_name = raw_tool_name
    except Exception as e:
        logger.warning(f"Could not extract tool name from context: {e}")

    # The event IS the arguments (not wrapped in a "arguments" key)
    arguments = event if isinstance(event, dict) else {}

    logger.info(f"Tool: {tool_name}, Args: {arguments}")

    func = TOOL_MAP.get(tool_name)
    if not func:
        return {
            "status": "error",
            "error_message": f"Unknown tool: {tool_name}",
            "available_tools": list(TOOL_MAP.keys()),
        }

    try:
        result = func(arguments)
        result["timestamp"] = datetime.now(timezone.utc).isoformat()
        return result
    except Exception as e:
        logger.error(f"Tool execution failed: {e}", exc_info=True)
        return {"status": "error", "error_message": str(e)}
