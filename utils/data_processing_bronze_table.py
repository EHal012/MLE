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
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from pyspark.sql import DataFrame

def _normalize_snapshot_date(df: DataFrame, colname: str = "snapshot_date") -> DataFrame:
    """
    Standardize snapshot date column using regex replacement and date parsing
    Different implementation: Using multiple when conditions instead of regexp_replace
    """
    date_str = F.col(colname).cast("string")
    
    # Replace different separators with standard format
    date_str = F.when(date_str.contains("/"), F.regexp_replace(date_str, "/", "-"))\
                .when(date_str.contains("."), F.regexp_replace(date_str, r"\.", "-"))\
                .otherwise(date_str)
    
    # Try parsing with different formats
    parsed_date = F.coalesce(
        F.to_date(date_str, "yyyy-M-d"),
        F.to_date(date_str, "yyyy-MM-dd"),
        F.to_date(date_str, "M-d-yyyy"),
        F.to_date(date_str, "MM-dd-yyyy")
    )
    
    return df.withColumn(colname, parsed_date)

def _filter_by_snapshot(df: DataFrame, snapshot_date_str: str) -> DataFrame:
    """
    Filter records matching the target snapshot date
    Different implementation: Parse date differently
    """
    target_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    date_literal = F.lit(target_date.date())
    
    filtered = df.where(F.col("snapshot_date") == date_literal)
    return filtered

def _ensure_dirs(path: str):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[Bronze] Created directory: {path}")

def _save_parquet(df: DataFrame, out_dir: str, name_prefix: str, snapshot_date_str: str):
    """
    Save DataFrame to parquet file
    Different implementation: Build path step by step
    """
    _ensure_dirs(out_dir)
    
    # Build filename with date formatting
    date_suffix = snapshot_date_str.replace("-", "_")
    filename = f"{name_prefix}_{date_suffix}.parquet"
    
    # Combine path
    full_path = os.path.join(out_dir, filename)
    
    # Write with overwrite mode
    df.write.mode("overwrite").parquet(full_path)
    
    record_count = df.count()
    print(f"[Bronze] Saved {record_count} records to {full_path}")

# ---- Bronze: loan_daily ----
def process_bronze_loan_daily(snapshot_date_str: str,
                              bronze_lms_directory: str,
                              spark: pyspark.sql.SparkSession) -> DataFrame:
    """
    Process raw loan daily data into bronze layer
    Different implementation: More explicit column handling
    """
    source_path = "data/lms_loan_daily.csv"
    
    # Load CSV with schema inference
    raw_df = spark.read.option("header", "true")\
                      .option("inferSchema", "true")\
                      .csv(source_path)
    
    print(f"[Bronze] Loaded loan_daily: {raw_df.count()} rows")
    
    # Normalize date column
    df_normalized = _normalize_snapshot_date(raw_df, "snapshot_date")
    
    # Apply snapshot filter
    df_filtered = _filter_by_snapshot(df_normalized, snapshot_date_str)
    
    # Save to bronze
    _save_parquet(df_filtered, bronze_lms_directory, "bronze_loan_daily", snapshot_date_str)
    
    return df_filtered

# ---- Bronze: feature_clickstream ----
def process_bronze_feature_clickstream(snapshot_date_str: str,
                                       bronze_clickstream_directory: str,
                                       spark: pyspark.sql.SparkSession) -> DataFrame:
    """
    Process raw clickstream features into bronze layer
    Different implementation: Explicit column rename checking
    """
    source_path = "data/feature_clickstream.csv"
    
    # Load data
    raw_df = spark.read.csv(source_path, header=True, inferSchema=True)
    
    print(f"[Bronze] Loaded clickstream: {raw_df.count()} rows")
    
    # Check and rename Customer_ID variants
    current_cols = raw_df.columns
    for col_name in current_cols:
        if col_name.lower().replace("_", "") == "customerid":
            if col_name != "Customer_ID":
                raw_df = raw_df.withColumnRenamed(col_name, "Customer_ID")
                print(f"[Bronze] Renamed {col_name} to Customer_ID")
    
    # Process dates
    df_normalized = _normalize_snapshot_date(raw_df, "snapshot_date")
    
    # Filter by snapshot
    df_filtered = _filter_by_snapshot(df_normalized, snapshot_date_str)
    
    # Save
    _save_parquet(df_filtered, bronze_clickstream_directory, 
                 "bronze_feature_clickstream", snapshot_date_str)
    
    return df_filtered

# ---- Bronze: feature_attributes ----
def process_bronze_feature_attributes(snapshot_date_str: str,
                                      bronze_attributes_directory: str,
                                      spark: pyspark.sql.SparkSession) -> DataFrame:
    """
    Process raw attribute features into bronze layer
    Different implementation: Using builder pattern for CSV reading
    """
    source_path = "data/features_attributes.csv"
    
    # Load with explicit options
    raw_df = (spark.read
              .format("csv")
              .option("header", "true")
              .option("inferSchema", "true")
              .load(source_path))
    
    print(f"[Bronze] Loaded attributes: {raw_df.count()} rows")
    
    # Rename Customer_ID if needed
    if "customer_id" in [c.lower() for c in raw_df.columns]:
        for old_col in raw_df.columns:
            if old_col.lower() == "customer_id":
                raw_df = raw_df.withColumnRenamed(old_col, "Customer_ID")
    
    # Process dates
    df_normalized = _normalize_snapshot_date(raw_df, "snapshot_date")
    
    # Filter
    df_filtered = _filter_by_snapshot(df_normalized, snapshot_date_str)
    
    # Save
    _save_parquet(df_filtered, bronze_attributes_directory, 
                 "bronze_feature_attributes", snapshot_date_str)
    
    return df_filtered

# ---- Bronze: feature_financials ----
def process_bronze_feature_financials(snapshot_date_str: str,
                                      bronze_financials_directory: str,
                                      spark: pyspark.sql.SparkSession) -> DataFrame:
    """
    Process raw financial features into bronze layer
    Different implementation: Two-step filtering approach
    """
    source_path = "data/features_financials.csv"
    
    # Read CSV
    raw_df = spark.read.csv(source_path, header=True, inferSchema=True)
    
    print(f"[Bronze] Loaded financials: {raw_df.count()} rows")
    
    # Standardize Customer_ID column
    column_mapping = {c: "Customer_ID" for c in raw_df.columns 
                     if c.lower().replace("_", "") == "customerid"}
    
    for old_name, new_name in column_mapping.items():
        if old_name != new_name:
            raw_df = raw_df.withColumnRenamed(old_name, new_name)
    
    # Date normalization
    df_normalized = _normalize_snapshot_date(raw_df, "snapshot_date")
    
    # Two-step filter: First remove nulls, then filter by date
    df_clean = df_normalized.filter(F.col("snapshot_date").isNotNull())
    df_filtered = _filter_by_snapshot(df_clean, snapshot_date_str)
    
    # Save
    _save_parquet(df_filtered, bronze_financials_directory, 
                 "bronze_feature_financials", snapshot_date_str)
    
    return df_filtered