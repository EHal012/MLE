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

import os
import pyspark.sql.functions as F

def build_gold_feature_store(
    snapshot_date_str: str,
    silver_clickstream_dir: str,
    silver_attributes_dir: str,
    silver_financials_dir: str,
    label_store_dir: str,
    gold_feature_store_dir: str,
    spark
):
    """
    Build gold feature store
    Different implementation: More modular with helper functions
    """
    print(f"[Gold] Building feature store for {snapshot_date_str}")
    
    # Format date key
    date_key = snapshot_date_str.replace("-", "_")

    # Build file paths using string formatting
    clickstream_path = f"{silver_clickstream_dir}/silver_feature_clickstream_{date_key}.parquet"
    attributes_path = f"{silver_attributes_dir}/silver_feature_attributes_{date_key}.parquet"
    financials_path = f"{silver_financials_dir}/silver_feature_financials_{date_key}.parquet"

    # Load silver tables
    clickstream_df = spark.read.parquet(clickstream_path)
    attributes_df = spark.read.parquet(attributes_path)
    financials_df = spark.read.parquet(financials_path)
    
    print(f"[Gold] Loaded clickstream: {clickstream_df.count()} rows")
    print(f"[Gold] Loaded attributes: {attributes_df.count()} rows")
    print(f"[Gold] Loaded financials: {financials_df.count()} rows")

    # Helper function for standardization
    def standardize_dataframe(input_df):
        """Standardize Customer_ID and snapshot_date"""
        standardized = input_df.withColumn(
            "Customer_ID", 
            F.col("Customer_ID").cast("string")
        )
        standardized = standardized.withColumn(
            "snapshot_date",
            F.to_date("snapshot_date")
        )
        return standardized

    # Apply standardization to all tables
    clickstream_df = standardize_dataframe(clickstream_df)
    attributes_df = standardize_dataframe(attributes_df)
    financials_df = standardize_dataframe(financials_df)

    # Feature Engineering Section
    print("[Gold] Starting feature engineering...")
    
    # 1) Clickstream aggregation: sum all fe_ columns
    fe_column_names = [c for c in clickstream_df.columns if c.startswith("fe_")]
    
    if len(fe_column_names) > 0:
        print(f"[Gold] Found {len(fe_column_names)} clickstream features")
        
        # Build sum expression incrementally
        sum_expression = F.col(fe_column_names[0])
        for feature_col in fe_column_names[1:]:
            sum_expression = sum_expression + F.col(feature_col)
        
        # Add aggregated feature
        clickstream_df = clickstream_df.withColumn(
            "click_sum",
            sum_expression.cast("double")
        )
        print("[Gold] Created click_sum feature")

    # 2) Financial feature: ensure Annual_Income is double
    if "Annual_Income" in financials_df.columns:
        financials_df = financials_df.withColumn(
            "Annual_Income",
            F.col("Annual_Income").cast("double")
        )
        print("[Gold] Casted Annual_Income to double")

    # 3) Load label data
    label_filename = f"gold_label_store_{date_key}.parquet"
    label_filepath = os.path.join(label_store_dir, label_filename)
    
    print(f"[Gold] Loading labels from {label_filepath}")
    
    # Read and prepare labels
    labels_df = spark.read.parquet(label_filepath)
    labels_df = labels_df.withColumn("Customer_ID", F.col("Customer_ID").cast("string"))
    labels_df = labels_df.withColumn("snapshot_date", F.to_date("snapshot_date"))
    
    # Select only required label columns
    labels_df = labels_df.select(
        "Customer_ID",
        "snapshot_date", 
        "label",
        "label_def"
    )
    
    print(f"[Gold] Loaded labels: {labels_df.count()} rows")

    # 4) Join all tables together
    print("[Gold] Joining tables...")
    
    # Start with labels as base
    merged_df = labels_df
    
    # Join clickstream
    merged_df = merged_df.join(
        clickstream_df,
        on=["Customer_ID", "snapshot_date"],
        how="left"
    )
    
    # Join attributes
    merged_df = merged_df.join(
        attributes_df,
        on=["Customer_ID", "snapshot_date"],
        how="left"
    )
    
    # Join financials
    merged_df = merged_df.join(
        financials_df,
        on=["Customer_ID", "snapshot_date"],
        how="left"
    )
    
    print(f"[Gold] Merged data: {merged_df.count()} rows")

    # 5) Derived feature: income per click
    has_income = "Annual_Income" in merged_df.columns
    has_clicks = "click_sum" in merged_df.columns
    
    if has_income and has_clicks:
        print("[Gold] Creating income_per_click feature")
        
        # Calculate ratio with null handling
        income_per_click_expr = F.when(
            (F.col("click_sum").isNotNull()) & (F.col("click_sum") != 0),
            F.col("Annual_Income") / F.col("click_sum")
        ).otherwise(None)
        
        merged_df = merged_df.withColumn(
            "income_per_click",
            income_per_click_expr
        )

    # 6) Data quality: remove duplicates
    initial_count = merged_df.count()
    merged_df = merged_df.dropDuplicates(["Customer_ID", "snapshot_date"])
    final_count = merged_df.count()
    
    if initial_count != final_count:
        print(f"[Gold] Removed {initial_count - final_count} duplicates")

    # 7) Save gold feature store
    os.makedirs(gold_feature_store_dir, exist_ok=True)
    
    output_filename = f"gold_feature_store_{date_key}.parquet"
    output_filepath = os.path.join(gold_feature_store_dir, output_filename)
    
    # Write to parquet
    merged_df.write.mode("overwrite").parquet(output_filepath)
    
    print(f"[Gold] ✓ Saved feature store: {output_filepath}")
    print(f"[Gold] Final row count: {final_count}")
    print(f"[Gold] Total columns: {len(merged_df.columns)}")

    return merged_df