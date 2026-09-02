"""
DynamoDB Seeder Lambda — Custom Resource handler.

Seeds three DynamoDB tables with cost parameters, external cost data,
and variant cache data (from S3 KPI predictions CSV).

Runs as a CloudFormation Custom Resource during CDK deployment.
"""

import csv
import io
import json
import logging
import os
from decimal import Decimal

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

COST_PARAMS_TABLE = os.environ["COST_PARAMS_TABLE"]
EXTERNAL_COST_TABLE = os.environ["EXTERNAL_COST_TABLE"]
VARIANT_CACHE_TABLE = os.environ["VARIANT_CACHE_TABLE"]
MODEL_BUCKET = os.environ["MODEL_BUCKET"]
KPI_CSV_KEY = os.environ.get("KPI_CSV_KEY", "predictions/kpi/variant_kpis_all4.csv")

dynamodb_resource = boto3.resource("dynamodb")
s3_client = boto3.client("s3")


def d(val):
    return Decimal(str(val))


# ===== COST PARAMETERS DATA =====

COST_PARAMS_DATA = [
    # MATERIAL
    {"pk": "MATERIAL", "sk": "steel", "value": d(1.20), "unit": "USD/kg", "description": "CR4 mild steel sheet"},
    {"pk": "MATERIAL", "sk": "aluminum", "value": d(3.50), "unit": "USD/kg", "description": "6061-T6 aluminum alloy sheet"},
    {"pk": "MATERIAL", "sk": "carbon_fiber", "value": d(25.00), "unit": "USD/kg", "description": "T700S carbon fiber reinforced polymer"},
    {"pk": "MATERIAL", "sk": "high_strength_steel", "value": d(1.85), "unit": "USD/kg", "description": "DP780 dual-phase high-strength steel"},
    {"pk": "MATERIAL", "sk": "magnesium", "value": d(4.20), "unit": "USD/kg", "description": "AZ91D magnesium alloy die-cast sheet"},
    {"pk": "MATERIAL", "sk": "titanium", "value": d(35.00), "unit": "USD/kg", "description": "Ti-6Al-4V Grade 5 titanium sheet"},
    {"pk": "MATERIAL", "sk": "glass_fiber", "value": d(8.50), "unit": "USD/kg", "description": "E-glass fiber reinforced SMC"},
    # STAMPING
    {"pk": "STAMPING", "sk": "cost_per_operation", "value": d(150.0), "unit": "USD/op", "description": "Base cost per stamping press operation"},
    {"pk": "STAMPING", "sk": "multi_stage_multiplier", "value": d(1.15), "unit": "multiplier", "description": "Cost multiplier for multi-stage progressive stamping"},
    {"pk": "STAMPING", "sk": "setup_time_hours", "value": d(4.5), "unit": "hours", "description": "Average die changeover and setup time"},
    {"pk": "STAMPING", "sk": "setup_cost_per_hour", "value": d(220.0), "unit": "USD/hr", "description": "Skilled technician rate for press setup"},
    {"pk": "STAMPING", "sk": "blanking_cost_per_kg", "value": d(0.35), "unit": "USD/kg", "description": "Sheet metal blanking and nesting cost"},
    {"pk": "STAMPING", "sk": "scrap_rate_steel", "value": d(0.12), "unit": "ratio", "description": "Typical scrap rate for steel stamping (12%)"},
    {"pk": "STAMPING", "sk": "scrap_rate_aluminum", "value": d(0.18), "unit": "ratio", "description": "Typical scrap rate for aluminum stamping (18%)"},
    {"pk": "STAMPING", "sk": "hot_stamping_surcharge", "value": d(85.0), "unit": "USD/op", "description": "Additional cost for hot-stamping"},
    # TOOLING
    {"pk": "TOOLING", "sk": "base_cost", "value": d(50000.0), "unit": "USD", "description": "Base tooling cost for a single stamping die set"},
    {"pk": "TOOLING", "sk": "trim_die_cost", "value": d(22000.0), "unit": "USD", "description": "Trim and pierce die for edge finishing"},
    {"pk": "TOOLING", "sk": "flange_die_cost", "value": d(18000.0), "unit": "USD", "description": "Flange and hem die for panel edge folding"},
    {"pk": "TOOLING", "sk": "checking_fixture_cost", "value": d(15000.0), "unit": "USD", "description": "Dimensional checking fixture"},
    {"pk": "TOOLING", "sk": "prototype_die_cost", "value": d(12000.0), "unit": "USD", "description": "Soft-tool prototype die for validation"},
    {"pk": "TOOLING", "sk": "maintenance_annual_pct", "value": d(0.08), "unit": "ratio", "description": "Annual die maintenance cost (8%)"},
    {"pk": "TOOLING", "sk": "amortization_units", "value": d(150000), "unit": "parts", "description": "Expected die life for cost amortization"},
    # ASSEMBLY
    {"pk": "ASSEMBLY", "sk": "welding_cost_per_meter", "value": d(12.0), "unit": "USD/m", "description": "Robotic MIG/MAG welding cost per meter"},
    {"pk": "ASSEMBLY", "sk": "assembly_cost_per_panel", "value": d(85.0), "unit": "USD/panel", "description": "Assembly labor and fixturing cost per panel"},
    {"pk": "ASSEMBLY", "sk": "spot_weld_cost", "value": d(0.08), "unit": "USD/weld", "description": "Robotic resistance spot weld cost"},
    {"pk": "ASSEMBLY", "sk": "laser_weld_cost_per_meter", "value": d(18.50), "unit": "USD/m", "description": "Robotic laser welding cost per meter"},
    {"pk": "ASSEMBLY", "sk": "adhesive_bond_cost_per_meter", "value": d(6.50), "unit": "USD/m", "description": "Structural adhesive bonding cost per meter"},
    {"pk": "ASSEMBLY", "sk": "rivet_cost_per_point", "value": d(0.45), "unit": "USD/rivet", "description": "Self-piercing rivet cost per point"},
    {"pk": "ASSEMBLY", "sk": "hemming_cost_per_meter", "value": d(8.00), "unit": "USD/m", "description": "Roller hemming cost per meter"},
    {"pk": "ASSEMBLY", "sk": "fixture_cost_per_station", "value": d(35000.0), "unit": "USD", "description": "Assembly fixture cost per welding station"},
    # MULTIPLIER
    {"pk": "MULTIPLIER", "sk": "very_low", "value": d(0.85), "unit": "multiplier", "description": "Very simple geometry"},
    {"pk": "MULTIPLIER", "sk": "low", "value": d(1.0), "unit": "multiplier", "description": "Simple geometry"},
    {"pk": "MULTIPLIER", "sk": "medium", "value": d(1.3), "unit": "multiplier", "description": "Moderate complexity"},
    {"pk": "MULTIPLIER", "sk": "high", "value": d(1.8), "unit": "multiplier", "description": "Complex geometry"},
    {"pk": "MULTIPLIER", "sk": "very_high", "value": d(2.5), "unit": "multiplier", "description": "Extreme complexity"},
    # SURFACE_TREATMENT
    {"pk": "SURFACE_TREATMENT", "sk": "e_coat_cost_per_m2", "value": d(4.50), "unit": "USD/m2", "description": "Electrophoretic coating per m2"},
    {"pk": "SURFACE_TREATMENT", "sk": "primer_cost_per_m2", "value": d(3.20), "unit": "USD/m2", "description": "Primer coat application per m2"},
    {"pk": "SURFACE_TREATMENT", "sk": "basecoat_cost_per_m2", "value": d(5.80), "unit": "USD/m2", "description": "Basecoat paint application per m2"},
    {"pk": "SURFACE_TREATMENT", "sk": "clearcoat_cost_per_m2", "value": d(4.10), "unit": "USD/m2", "description": "Clearcoat application per m2"},
    {"pk": "SURFACE_TREATMENT", "sk": "galvanizing_cost_per_m2", "value": d(2.80), "unit": "USD/m2", "description": "Hot-dip galvanizing per m2"},
    {"pk": "SURFACE_TREATMENT", "sk": "anodizing_cost_per_m2", "value": d(8.50), "unit": "USD/m2", "description": "Anodizing for aluminum panels per m2"},
    # QUALITY
    {"pk": "QUALITY", "sk": "cmm_inspection_per_panel", "value": d(12.0), "unit": "USD/panel", "description": "CMM dimensional inspection per panel"},
    {"pk": "QUALITY", "sk": "visual_inspection_per_unit", "value": d(35.0), "unit": "USD/unit", "description": "Manual visual quality inspection per unit"},
    {"pk": "QUALITY", "sk": "rework_cost_per_hour", "value": d(95.0), "unit": "USD/hr", "description": "Skilled rework technician rate"},
    {"pk": "QUALITY", "sk": "target_scrap_rate", "value": d(0.02), "unit": "ratio", "description": "Target finished body scrap rate (2%)"},
    {"pk": "QUALITY", "sk": "ultrasonic_test_per_weld", "value": d(1.50), "unit": "USD/test", "description": "Ultrasonic weld quality testing"},
    # LOGISTICS
    {"pk": "LOGISTICS", "sk": "intra_plant_transport_per_unit", "value": d(18.0), "unit": "USD/unit", "description": "AGV/conveyor transport cost per unit"},
    {"pk": "LOGISTICS", "sk": "packaging_cost_per_panel", "value": d(3.50), "unit": "USD/panel", "description": "Interleaving and rack packaging per panel"},
    {"pk": "LOGISTICS", "sk": "warehouse_cost_per_m3_day", "value": d(0.85), "unit": "USD/m3/day", "description": "WIP buffer storage cost"},
    {"pk": "LOGISTICS", "sk": "energy_cost_per_kwh", "value": d(0.09), "unit": "USD/kWh", "description": "Industrial electricity rate"},
]

# ===== EXTERNAL COST DATA =====

EXTERNAL_COST_DATA = [
    # MARKET_PRICE — global base prices
    {"pk": "MARKET_PRICE", "sk": "steel", "price_per_kg": d(1.20), "currency": "USD", "trend": "stable", "trend_pct_30d": d(-0.5), "source": "LME_Index", "effective_date": "2026-02-01", "grade": "CR4_mild"},
    {"pk": "MARKET_PRICE", "sk": "aluminum", "price_per_kg": d(3.50), "currency": "USD", "trend": "rising", "trend_pct_30d": d(2.3), "source": "LME_Index", "effective_date": "2026-02-01", "grade": "6061-T6"},
    {"pk": "MARKET_PRICE", "sk": "carbon_fiber", "price_per_kg": d(25.00), "currency": "USD", "trend": "declining", "trend_pct_30d": d(-3.1), "source": "Toray_Quote", "effective_date": "2026-01-15", "grade": "T700S"},
    {"pk": "MARKET_PRICE", "sk": "high_strength_steel", "price_per_kg": d(1.85), "currency": "USD", "trend": "stable", "trend_pct_30d": d(0.2), "source": "LME_Index", "effective_date": "2026-02-01", "grade": "DP780"},
    {"pk": "MARKET_PRICE", "sk": "magnesium", "price_per_kg": d(4.20), "currency": "USD", "trend": "rising", "trend_pct_30d": d(1.8), "source": "Shanghai_Metal", "effective_date": "2026-02-01", "grade": "AZ91D"},
    {"pk": "MARKET_PRICE", "sk": "titanium", "price_per_kg": d(35.00), "currency": "USD", "trend": "stable", "trend_pct_30d": d(0.0), "source": "USGS_Index", "effective_date": "2026-02-01", "grade": "Ti-6Al-4V"},
    # MARKET_PRICE — regional variants
    {"pk": "MARKET_PRICE", "sk": "steel#north_america", "price_per_kg": d(1.25), "currency": "USD", "trend": "stable", "trend_pct_30d": d(-0.3), "source": "CRU_Steel", "effective_date": "2026-02-01", "grade": "CR4_mild"},
    {"pk": "MARKET_PRICE", "sk": "steel#europe", "price_per_kg": d(1.35), "currency": "USD", "trend": "rising", "trend_pct_30d": d(1.1), "source": "Platts_EU", "effective_date": "2026-02-01", "grade": "CR4_mild"},
    {"pk": "MARKET_PRICE", "sk": "steel#asia_pacific", "price_per_kg": d(1.05), "currency": "USD", "trend": "declining", "trend_pct_30d": d(-1.5), "source": "MySteel_CN", "effective_date": "2026-02-01", "grade": "CR4_mild"},
    {"pk": "MARKET_PRICE", "sk": "aluminum#north_america", "price_per_kg": d(3.65), "currency": "USD", "trend": "rising", "trend_pct_30d": d(2.8), "source": "Midwest_Premium", "effective_date": "2026-02-01", "grade": "6061-T6"},
    {"pk": "MARKET_PRICE", "sk": "aluminum#europe", "price_per_kg": d(3.80), "currency": "USD", "trend": "rising", "trend_pct_30d": d(3.2), "source": "LME_EU_Duty", "effective_date": "2026-02-01", "grade": "6061-T6"},
    {"pk": "MARKET_PRICE", "sk": "aluminum#asia_pacific", "price_per_kg": d(3.20), "currency": "USD", "trend": "stable", "trend_pct_30d": d(0.5), "source": "SHFE_Index", "effective_date": "2026-02-01", "grade": "6061-T6"},
    {"pk": "MARKET_PRICE", "sk": "carbon_fiber#north_america", "price_per_kg": d(24.50), "currency": "USD", "trend": "declining", "trend_pct_30d": d(-2.8), "source": "Hexcel_Quote", "effective_date": "2026-01-20", "grade": "T700S"},
    {"pk": "MARKET_PRICE", "sk": "carbon_fiber#europe", "price_per_kg": d(26.00), "currency": "USD", "trend": "declining", "trend_pct_30d": d(-2.5), "source": "SGL_Carbon", "effective_date": "2026-01-20", "grade": "T700S"},
    {"pk": "MARKET_PRICE", "sk": "carbon_fiber#asia_pacific", "price_per_kg": d(22.00), "currency": "USD", "trend": "declining", "trend_pct_30d": d(-4.0), "source": "Toray_JP", "effective_date": "2026-01-20", "grade": "T700S"},
    # SUPPLIER
    {"pk": "SUPPLIER", "sk": "stamping_die#supplier_1", "supplier_name": "PrecisionDie Corp", "supplier_id": "SUP-001", "base_price": d(48000), "currency": "USD", "lead_time_weeks": d(10), "availability": "available", "warranty_months": d(24), "rating": d(4.5)},
    {"pk": "SUPPLIER", "sk": "stamping_die#supplier_2", "supplier_name": "ToolTech Industries", "supplier_id": "SUP-002", "base_price": d(45000), "currency": "USD", "lead_time_weeks": d(12), "availability": "available", "warranty_months": d(18), "rating": d(4.2)},
    {"pk": "SUPPLIER", "sk": "stamping_die#supplier_3", "supplier_name": "Shanghai Precision", "supplier_id": "SUP-003", "base_price": d(38000), "currency": "USD", "lead_time_weeks": d(16), "availability": "available", "warranty_months": d(12), "rating": d(3.8)},
    {"pk": "SUPPLIER", "sk": "welding_fixture#supplier_1", "supplier_name": "WeldTech Solutions", "supplier_id": "SUP-004", "base_price": d(8500), "currency": "USD", "lead_time_weeks": d(6), "availability": "available", "warranty_months": d(12), "rating": d(4.3)},
    {"pk": "SUPPLIER", "sk": "welding_fixture#supplier_2", "supplier_name": "JoinPro Systems", "supplier_id": "SUP-005", "base_price": d(7800), "currency": "USD", "lead_time_weeks": d(8), "availability": "limited", "warranty_months": d(12), "rating": d(4.0)},
    {"pk": "SUPPLIER", "sk": "assembly_jig#supplier_1", "supplier_name": "AssemblyMaster", "supplier_id": "SUP-006", "base_price": d(12000), "currency": "USD", "lead_time_weeks": d(8), "availability": "available", "warranty_months": d(18), "rating": d(4.4)},
    {"pk": "SUPPLIER", "sk": "assembly_jig#supplier_2", "supplier_name": "FixturePro GmbH", "supplier_id": "SUP-007", "base_price": d(14500), "currency": "USD", "lead_time_weeks": d(6), "availability": "available", "warranty_months": d(24), "rating": d(4.7)},
    {"pk": "SUPPLIER", "sk": "trim_die#supplier_1", "supplier_name": "PrecisionDie Corp", "supplier_id": "SUP-001", "base_price": d(22000), "currency": "USD", "lead_time_weeks": d(8), "availability": "available", "warranty_months": d(18), "rating": d(4.5)},
    {"pk": "SUPPLIER", "sk": "inspection_fixture#supplier_1", "supplier_name": "MetroTech QA", "supplier_id": "SUP-008", "base_price": d(18000), "currency": "USD", "lead_time_weeks": d(10), "availability": "available", "warranty_months": d(24), "rating": d(4.6)},
    {"pk": "SUPPLIER", "sk": "paint_booth_time#supplier_1", "supplier_name": "FinishPro Coatings", "supplier_id": "SUP-009", "base_price": d(250), "currency": "USD/hour", "lead_time_weeks": d(2), "availability": "available", "warranty_months": d(0), "rating": d(4.1)},
    # HISTORICAL
    {"pk": "HISTORICAL", "sk": "steel#low#sedan", "sample_size": d(85), "date_range": "2023-2025", "min_cost": d(38000), "max_cost": d(52000), "avg_cost": d(44500), "median_cost": d(43800), "p25_cost": d(41200), "p75_cost": d(47500), "std_dev": d(3800), "material_pct": d(15), "stamping_pct": d(25), "tooling_pct": d(40), "assembly_pct": d(20)},
    {"pk": "HISTORICAL", "sk": "steel#medium#sedan", "sample_size": d(120), "date_range": "2023-2025", "min_cost": d(48000), "max_cost": d(68000), "avg_cost": d(56000), "median_cost": d(55200), "p25_cost": d(52000), "p75_cost": d(60000), "std_dev": d(4500), "material_pct": d(14), "stamping_pct": d(26), "tooling_pct": d(38), "assembly_pct": d(22)},
    {"pk": "HISTORICAL", "sk": "steel#high#sedan", "sample_size": d(45), "date_range": "2023-2025", "min_cost": d(65000), "max_cost": d(95000), "avg_cost": d(78000), "median_cost": d(76500), "p25_cost": d(72000), "p75_cost": d(84000), "std_dev": d(7200), "material_pct": d(12), "stamping_pct": d(28), "tooling_pct": d(36), "assembly_pct": d(24)},
    {"pk": "HISTORICAL", "sk": "steel#medium#suv", "sample_size": d(95), "date_range": "2023-2025", "min_cost": d(55000), "max_cost": d(78000), "avg_cost": d(65000), "median_cost": d(64200), "p25_cost": d(60000), "p75_cost": d(70000), "std_dev": d(5200), "material_pct": d(16), "stamping_pct": d(24), "tooling_pct": d(37), "assembly_pct": d(23)},
    {"pk": "HISTORICAL", "sk": "aluminum#low#sedan", "sample_size": d(35), "date_range": "2023-2025", "min_cost": d(52000), "max_cost": d(72000), "avg_cost": d(61000), "median_cost": d(60500), "p25_cost": d(57000), "p75_cost": d(65000), "std_dev": d(4800), "material_pct": d(22), "stamping_pct": d(22), "tooling_pct": d(35), "assembly_pct": d(21)},
    {"pk": "HISTORICAL", "sk": "aluminum#medium#sedan", "sample_size": d(62), "date_range": "2023-2025", "min_cost": d(62000), "max_cost": d(88000), "avg_cost": d(74000), "median_cost": d(73000), "p25_cost": d(69000), "p75_cost": d(79000), "std_dev": d(5800), "material_pct": d(24), "stamping_pct": d(21), "tooling_pct": d(34), "assembly_pct": d(21)},
    {"pk": "HISTORICAL", "sk": "aluminum#high#sedan", "sample_size": d(22), "date_range": "2023-2025", "min_cost": d(85000), "max_cost": d(125000), "avg_cost": d(102000), "median_cost": d(100000), "p25_cost": d(94000), "p75_cost": d(110000), "std_dev": d(9500), "material_pct": d(20), "stamping_pct": d(23), "tooling_pct": d(33), "assembly_pct": d(24)},
    {"pk": "HISTORICAL", "sk": "carbon_fiber#medium#sedan", "sample_size": d(18), "date_range": "2023-2025", "min_cost": d(120000), "max_cost": d(180000), "avg_cost": d(148000), "median_cost": d(145000), "p25_cost": d(135000), "p75_cost": d(160000), "std_dev": d(15000), "material_pct": d(35), "stamping_pct": d(15), "tooling_pct": d(30), "assembly_pct": d(20)},
    {"pk": "HISTORICAL", "sk": "carbon_fiber#high#coupe", "sample_size": d(12), "date_range": "2023-2025", "min_cost": d(160000), "max_cost": d(240000), "avg_cost": d(195000), "median_cost": d(190000), "p25_cost": d(175000), "p75_cost": d(215000), "std_dev": d(22000), "material_pct": d(38), "stamping_pct": d(12), "tooling_pct": d(28), "assembly_pct": d(22)},
    # REGIONAL
    {"pk": "REGIONAL", "sk": "north_america", "labor_multiplier": d(1.0), "energy_multiplier": d(0.95), "logistics_multiplier": d(1.0), "overhead_multiplier": d(1.0), "composite_multiplier": d(0.99), "currency": "USD", "effective_date": "2026-01-01"},
    {"pk": "REGIONAL", "sk": "europe", "labor_multiplier": d(1.25), "energy_multiplier": d(1.35), "logistics_multiplier": d(1.10), "overhead_multiplier": d(1.15), "composite_multiplier": d(1.21), "currency": "USD", "effective_date": "2026-01-01"},
    {"pk": "REGIONAL", "sk": "asia_pacific", "labor_multiplier": d(0.55), "energy_multiplier": d(0.80), "logistics_multiplier": d(1.20), "overhead_multiplier": d(0.85), "composite_multiplier": d(0.78), "currency": "USD", "effective_date": "2026-01-01"},
    {"pk": "REGIONAL", "sk": "south_america", "labor_multiplier": d(0.65), "energy_multiplier": d(0.90), "logistics_multiplier": d(1.30), "overhead_multiplier": d(1.05), "composite_multiplier": d(0.92), "currency": "USD", "effective_date": "2026-01-01"},
    {"pk": "REGIONAL", "sk": "middle_east", "labor_multiplier": d(0.70), "energy_multiplier": d(0.60), "logistics_multiplier": d(1.25), "overhead_multiplier": d(1.10), "composite_multiplier": d(0.86), "currency": "USD", "effective_date": "2026-01-01"},
    # VOLUME_DISCOUNT — steel
    {"pk": "VOLUME_DISCOUNT", "sk": "steel#tier_1", "tier_name": "Spot", "min_volume_kg": d(0), "max_volume_kg": d(5000), "discount_pct": d(0), "price_per_kg": d(1.25), "min_order_kg": d(100)},
    {"pk": "VOLUME_DISCOUNT", "sk": "steel#tier_2", "tier_name": "Small Batch", "min_volume_kg": d(5000), "max_volume_kg": d(25000), "discount_pct": d(5), "price_per_kg": d(1.19), "min_order_kg": d(1000)},
    {"pk": "VOLUME_DISCOUNT", "sk": "steel#tier_3", "tier_name": "Production", "min_volume_kg": d(25000), "max_volume_kg": d(100000), "discount_pct": d(12), "price_per_kg": d(1.10), "min_order_kg": d(5000)},
    {"pk": "VOLUME_DISCOUNT", "sk": "steel#tier_4", "tier_name": "High Volume", "min_volume_kg": d(100000), "max_volume_kg": d(999999), "discount_pct": d(18), "price_per_kg": d(1.03), "min_order_kg": d(10000)},
    # VOLUME_DISCOUNT — aluminum
    {"pk": "VOLUME_DISCOUNT", "sk": "aluminum#tier_1", "tier_name": "Spot", "min_volume_kg": d(0), "max_volume_kg": d(2000), "discount_pct": d(0), "price_per_kg": d(3.65), "min_order_kg": d(50)},
    {"pk": "VOLUME_DISCOUNT", "sk": "aluminum#tier_2", "tier_name": "Small Batch", "min_volume_kg": d(2000), "max_volume_kg": d(10000), "discount_pct": d(4), "price_per_kg": d(3.50), "min_order_kg": d(500)},
    {"pk": "VOLUME_DISCOUNT", "sk": "aluminum#tier_3", "tier_name": "Production", "min_volume_kg": d(10000), "max_volume_kg": d(50000), "discount_pct": d(10), "price_per_kg": d(3.29), "min_order_kg": d(2000)},
    {"pk": "VOLUME_DISCOUNT", "sk": "aluminum#tier_4", "tier_name": "High Volume", "min_volume_kg": d(50000), "max_volume_kg": d(999999), "discount_pct": d(15), "price_per_kg": d(3.10), "min_order_kg": d(5000)},
    # VOLUME_DISCOUNT — carbon fiber
    {"pk": "VOLUME_DISCOUNT", "sk": "carbon_fiber#tier_1", "tier_name": "Prototype", "min_volume_kg": d(0), "max_volume_kg": d(500), "discount_pct": d(0), "price_per_kg": d(25.00), "min_order_kg": d(10)},
    {"pk": "VOLUME_DISCOUNT", "sk": "carbon_fiber#tier_2", "tier_name": "Small Batch", "min_volume_kg": d(500), "max_volume_kg": d(2000), "discount_pct": d(8), "price_per_kg": d(23.00), "min_order_kg": d(100)},
    {"pk": "VOLUME_DISCOUNT", "sk": "carbon_fiber#tier_3", "tier_name": "Production", "min_volume_kg": d(2000), "max_volume_kg": d(10000), "discount_pct": d(15), "price_per_kg": d(21.25), "min_order_kg": d(500)},
]


# ===== STRUCTURAL/COST ESTIMATION CONSTANTS =====

MATERIAL_DENSITIES = {"steel": 7850.0, "aluminum": 2700.0, "carbon_fiber": 1600.0}
BASE_THICKNESS_MM = {"steel": 0.8, "aluminum": 1.2, "carbon_fiber": 2.0}
MATERIAL_COST_PER_KG = {"steel": 1.20, "aluminum": 3.50, "carbon_fiber": 25.00}
STAMPING_COST_PER_OP = 150.0
TOOLING_BASE_COST = 50000.0
WELDING_COST_PER_M = 12.0


def estimate_variant_cost(cd, cs, cl, cmy, variant_id):
    """Estimate structural + cost data for a variant from its KPIs.

    Uses heuristic surface area estimation from drag coefficient.
    """
    material = "steel"
    # Heuristic: higher Cd → larger frontal area → larger surface area
    surface_area_m2 = 1.5 + abs(cd) * 3.0  # ~1.5-2.5 m2 range
    density = MATERIAL_DENSITIES[material]
    thickness_mm = BASE_THICKNESS_MM[material]
    weight_kg = surface_area_m2 * (thickness_mm / 1000.0) * density

    # Complexity from KPI spread
    complexity = min(1.0, abs(cd - 0.3) * 2 + abs(cl) * 0.5)
    multiplier = 1.0 + 0.8 * complexity

    mat_cost = weight_kg * MATERIAL_COST_PER_KG[material]
    stamp_cost = 4 * STAMPING_COST_PER_OP * 1.15
    tool_cost = TOOLING_BASE_COST * multiplier
    weld_length = surface_area_m2 * 0.5
    panel_count = max(1, int(surface_area_m2 * 2))
    asm_cost = panel_count * 85.0 + weld_length * WELDING_COST_PER_M
    total = mat_cost + stamp_cost + tool_cost + asm_cost

    stiffness_score = min(1.0, max(0.0, 0.7 + 0.1 * (1.0 - surface_area_m2 / 5.0)))

    return {
        "pk": "VARIANT",
        "sk": variant_id,
        "cd": d(cd),
        "cs": d(cs),
        "cl": d(cl),
        "cmy": d(cmy),
        "weight_kg": d(round(weight_kg, 2)),
        "stiffness_score": d(round(stiffness_score, 3)),
        "is_feasible": True,
        "total_cost": d(round(total, 2)),
        "material_cost": d(round(mat_cost, 2)),
        "stamping_cost": d(round(stamp_cost, 2)),
        "tooling_cost": d(round(tool_cost, 2)),
        "assembly_cost": d(round(asm_cost, 2)),
        "surface_area_m2": d(round(surface_area_m2, 3)),
        "material": material,
        # S3 asset references
        "vtp_s3_key": f"predictions/surface/{variant_id}.vtp",
        "geometry_s3_key": f"geometries/{variant_id}/geometry.stl",
    }


def seed_cost_params():
    """Seed the CarDesignCostParameters table."""
    table = dynamodb_resource.Table(COST_PARAMS_TABLE)
    with table.batch_writer() as batch:
        for item in COST_PARAMS_DATA:
            batch.put_item(Item=item)
    logger.info(f"Seeded {len(COST_PARAMS_DATA)} items into {COST_PARAMS_TABLE}")


def seed_external_cost():
    """Seed the CarDesignExternalCostData table."""
    table = dynamodb_resource.Table(EXTERNAL_COST_TABLE)
    with table.batch_writer() as batch:
        for item in EXTERNAL_COST_DATA:
            batch.put_item(Item=item)
    logger.info(f"Seeded {len(EXTERNAL_COST_DATA)} items into {EXTERNAL_COST_TABLE}")


def seed_variant_cache():
    """Seed the CarDesignVariantCache table from KPI predictions CSV in S3."""
    table = dynamodb_resource.Table(VARIANT_CACHE_TABLE)

    try:
        response = s3_client.get_object(Bucket=MODEL_BUCKET, Key=KPI_CSV_KEY)
        csv_content = response["Body"].read().decode("utf-8")
    except s3_client.exceptions.NoSuchKey:
        raise RuntimeError(
            f"KPI CSV not found at s3://{MODEL_BUCKET}/{KPI_CSV_KEY}. "
            "Ensure CarDesignSeed stack deployed successfully before this stack."
        )

    reader = csv.DictReader(io.StringIO(csv_content))
    count = 0
    with table.batch_writer() as batch:
        for row in reader:
            variant_id = row.get("variant_id", row.get("run_id", f"run_{count}"))
            cd = float(row.get("cd", row.get("drag_coefficient", 0)))
            cs = float(row.get("cs", row.get("side_force_coefficient", 0)))
            cl = float(row.get("cl", row.get("lift_coefficient", 0)))
            cmy = float(row.get("cmy", row.get("yaw_moment_coefficient", 0)))

            item = estimate_variant_cost(cd, cs, cl, cmy, variant_id)

            # Add slice image references (10 slices per variant)
            slice_keys = [
                f"predictions/slices/{variant_id}/slice_{i:02d}.png"
                for i in range(10)
            ]
            item["slice_images_json"] = json.dumps(slice_keys)

            batch.put_item(Item=item)
            count += 1

    logger.info(f"Seeded {count} variants into {VARIANT_CACHE_TABLE}")


def handler(event, context):
    """CloudFormation Custom Resource handler."""
    request_type = event.get("RequestType", "Create")
    logger.info(f"DynamoSeeder invoked: RequestType={request_type}")

    if request_type == "Delete":
        # Don't delete data on stack deletion — tables have DESTROY policy
        return {"Status": "SUCCESS", "PhysicalResourceId": "dynamo-seeder"}

    try:
        seed_cost_params()
        seed_external_cost()
        seed_variant_cache()

        return {
            "Status": "SUCCESS",
            "PhysicalResourceId": "dynamo-seeder",
            "Data": {
                "CostParamsCount": str(len(COST_PARAMS_DATA)),
                "ExternalCostCount": str(len(EXTERNAL_COST_DATA)),
                "Message": "All DynamoDB tables seeded successfully",
            },
        }
    except Exception as e:
        logger.error(f"Seeding failed: {e}")
        raise
