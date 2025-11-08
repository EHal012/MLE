import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from pyspark.sql import SparkSession

import utils.data_processing_bronze_table as bronze
import utils.data_processing_silver_table as silver
import utils.data_processing_gold_table as gold
import utils.eda_analysis as eda

# Pipeline configuration
SNAPSHOT_START = "2023-01-01"
SNAPSHOT_END = "2023-12-01"
LABEL_STORE_DIR = "datamart/gold/label_store"

# Directory structure for medallion architecture
DATAMART_PATHS = {
    "bronze": {
        "loan": "datamart/bronze/loan_daily/",
        "clickstream": "datamart/bronze/feature_clickstream/",
        "attributes": "datamart/bronze/feature_attributes/",
        "financials": "datamart/bronze/feature_financials/",
    },
    "silver": {
        "loan": "datamart/silver/loan_daily/",
        "clickstream": "datamart/silver/feature_clickstream/",
        "attributes": "datamart/silver/feature_attributes/",
        "financials": "datamart/silver/feature_financials/",
    },
    "gold": {
        "feature_store": "datamart/gold/feature_store/",
    }
}

def create_directory_structure():
    """
    Create all required directories for the data pipeline
    """
    all_paths = []
    
    # Collect bronze and silver paths
    for layer in ["bronze", "silver"]:
        all_paths.extend(DATAMART_PATHS[layer].values())
    
    # Add gold paths
    all_paths.extend(DATAMART_PATHS["gold"].values())
    
    # Add label store
    all_paths.append(LABEL_STORE_DIR)
    
    # Create directories
    for directory in all_paths:
        os.makedirs(directory, exist_ok=True)
        
    print(f"[SETUP] Created {len(all_paths)} directories")

def generate_monthly_snapshots(start_date: str, end_date: str):
    """
    Generate list of first-of-month dates between start and end
    Different implementation: Using relativedelta
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Start from first day of start month
    current_date = datetime(start_dt.year, start_dt.month, 1)
    
    snapshot_dates = []
    while current_date <= end_dt:
        snapshot_dates.append(current_date.strftime("%Y-%m-%d"))
        # Move to next month using relativedelta
        current_date = current_date + relativedelta(months=1)
    
    return snapshot_dates

def process_bronze_layer(snapshot_dates, spark):
    """
    Process all bronze layer tables for all snapshot dates
    """
    print(f"\n{'='*70}")
    print("BRONZE LAYER PROCESSING")
    print(f"{'='*70}")
    
    total_dates = len(snapshot_dates)
    bronze_paths = DATAMART_PATHS["bronze"]
    
    for idx, snapshot_date in enumerate(snapshot_dates, 1):
        print(f"\n[BRONZE] Processing date {idx}/{total_dates}: {snapshot_date}")
        
        # Process loan data
        bronze.process_bronze_loan_daily(
            snapshot_date_str=snapshot_date,
            bronze_lms_directory=bronze_paths["loan"],
            spark=spark
        )
        
        # Process clickstream features
        bronze.process_bronze_feature_clickstream(
            snapshot_date_str=snapshot_date,
            bronze_clickstream_directory=bronze_paths["clickstream"],
            spark=spark
        )
        
        # Process attribute features
        bronze.process_bronze_feature_attributes(
            snapshot_date_str=snapshot_date,
            bronze_attributes_directory=bronze_paths["attributes"],
            spark=spark
        )
        
        # Process financial features
        bronze.process_bronze_feature_financials(
            snapshot_date_str=snapshot_date,
            bronze_financials_directory=bronze_paths["financials"],
            spark=spark
        )
    
    print(f"\n[BRONZE] ✓ Completed processing {total_dates} snapshots")

def process_silver_layer(snapshot_dates, spark):
    """
    Process all silver layer tables for all snapshot dates
    """
    print(f"\n{'='*70}")
    print("SILVER LAYER PROCESSING")
    print(f"{'='*70}")
    
    total_dates = len(snapshot_dates)
    bronze_paths = DATAMART_PATHS["bronze"]
    silver_paths = DATAMART_PATHS["silver"]
    
    for idx, snapshot_date in enumerate(snapshot_dates, 1):
        print(f"\n[SILVER] Processing date {idx}/{total_dates}: {snapshot_date}")
        
        # Process loan data with feature engineering
        silver.process_silver_table(
            snapshot_date_str=snapshot_date,
            bronze_lms_directory=bronze_paths["loan"],
            silver_loan_daily_directory=silver_paths["loan"],
            spark=spark
        )
        
        # Process and clean clickstream features
        silver.process_silver_feature_clickstream(
            snapshot_date_str=snapshot_date,
            bronze_dir=bronze_paths["clickstream"],
            silver_dir=silver_paths["clickstream"],
            spark=spark
        )
        
        # Process and validate attribute features
        silver.process_silver_feature_attributes(
            snapshot_date_str=snapshot_date,
            bronze_dir=bronze_paths["attributes"],
            silver_dir=silver_paths["attributes"],
            spark=spark
        )
        
        # Process and parse financial features
        silver.process_silver_feature_financials(
            snapshot_date_str=snapshot_date,
            bronze_dir=bronze_paths["financials"],
            silver_dir=silver_paths["financials"],
            spark=spark
        )
    
    print(f"\n[SILVER] ✓ Completed processing {total_dates} snapshots")

def process_gold_layer(snapshot_dates, spark):
    """
    Build gold feature store by joining silver tables with labels
    """
    print(f"\n{'='*70}")
    print("GOLD LAYER PROCESSING")
    print(f"{'='*70}")
    
    total_dates = len(snapshot_dates)
    silver_paths = DATAMART_PATHS["silver"]
    gold_path = DATAMART_PATHS["gold"]["feature_store"]
    
    processed_count = 0
    skipped_count = 0
    
    for idx, snapshot_date in enumerate(snapshot_dates, 1):
        print(f"\n[GOLD] Processing date {idx}/{total_dates}: {snapshot_date}")
        
        # Check if label file exists
        date_key = snapshot_date.replace("-", "_")
        label_filename = f"gold_label_store_{date_key}.parquet"
        label_path = os.path.join(LABEL_STORE_DIR, label_filename)
        
        if not os.path.exists(label_path):
            print(f"[GOLD] ⚠️ Label file not found: {label_filename}")
            print(f"[GOLD] Skipping {snapshot_date}")
            skipped_count += 1
            continue
        
        # Build feature store
        gold.build_gold_feature_store(
            snapshot_date_str=snapshot_date,
            silver_clickstream_dir=silver_paths["clickstream"],
            silver_attributes_dir=silver_paths["attributes"],
            silver_financials_dir=silver_paths["financials"],
            label_store_dir=LABEL_STORE_DIR,
            gold_feature_store_dir=gold_path,
            spark=spark
        )
        
        processed_count += 1
    
    print(f"\n[GOLD] ✓ Completed: {processed_count} feature stores built")
    if skipped_count > 0:
        print(f"[GOLD] ⚠️ Skipped: {skipped_count} dates (missing labels)")

def run_pipeline():
    """
    Main pipeline execution
    """
    print("\n" + "="*70)
    print("DATA PROCESSING PIPELINE - MEDALLION ARCHITECTURE")
    print("="*70)
    
    # Initialize Spark session
    print("\n[SETUP] Initializing Spark session...")
    spark_session = (
        SparkSession.builder
        .appName("data_pipeline_medallion")
        .master("local[*]")
        .getOrCreate()
    )
    spark_session.sparkContext.setLogLevel("ERROR")
    print("[SETUP] ✓ Spark session initialized")
    
    # Create directory structure
    print("\n[SETUP] Creating directory structure...")
    create_directory_structure()
    
    # Generate snapshot dates
    print("\n[SETUP] Generating snapshot dates...")
    snapshot_list = generate_monthly_snapshots(SNAPSHOT_START, SNAPSHOT_END)
    print(f"[SETUP] ✓ Generated {len(snapshot_list)} monthly snapshots")
    print(f"[SETUP] Date range: {snapshot_list[0]} to {snapshot_list[-1]}")
    
    # Execute pipeline layers
    try:
        # Bronze layer
        process_bronze_layer(snapshot_list, spark_session)
        
        # Silver layer
        process_silver_layer(snapshot_list, spark_session)
        
        # Gold layer
        process_gold_layer(snapshot_list, spark_session)
        
        # Optional: Run EDA on first snapshot
        print(f"\n{'='*70}")
        print("EXPLORATORY DATA ANALYSIS (Optional)")
        print(f"{'='*70}")
        
        try:
            first_snapshot = snapshot_list[0]
            print(f"\n[EDA] Running analysis on {first_snapshot}...")
            eda.run_basic_eda(
                spark=spark_session,
                datamart_dir="datamart",
                snapshot_date=first_snapshot
            )
        except Exception as e:
            print(f"[EDA] Could not complete EDA: {e}")
        
        # Pipeline completion
        print(f"\n{'='*70}")
        print("✓ PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"  Bronze files: {len(snapshot_list) * 4}")
        print(f"  Silver files: {len(snapshot_list) * 4}")
        print(f"  Gold files: {len(snapshot_list)}")
        print(f"  Total files: {len(snapshot_list) * 9}")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        raise
    
    finally:
        # Stop Spark
        print("[CLEANUP] Stopping Spark session...")
        spark_session.stop()
        print("[CLEANUP] ✓ Done\n")

if __name__ == "__main__":
    run_pipeline()