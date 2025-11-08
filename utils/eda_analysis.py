"""
Simple Exploratory Data Analysis (EDA)
Basic data profiling and quality checks for the pipeline
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pyspark
import pyspark.sql.functions as F
from pyspark.sql import DataFrame


def profile_data_quality(df: DataFrame, table_name: str):
    """
    Generate basic data quality profile
    """
    print(f"\n{'='*60}")
    print(f"Data Quality Profile: {table_name}")
    print(f"{'='*60}")
    
    # Basic stats
    row_count = df.count()
    col_count = len(df.columns)
    
    print(f"\nDataset Shape:")
    print(f"  Rows: {row_count:,}")
    print(f"  Columns: {col_count}")
    
    # Missing values analysis
    print(f"\nMissing Values Check:")
    missing_data = []
    
    for column in df.columns:
        null_count = df.filter(F.col(column).isNull()).count()
        if null_count > 0:
            missing_pct = (null_count / row_count) * 100
            missing_data.append({
                'Column': column,
                'Missing': null_count,
                'Percentage': f"{missing_pct:.2f}%"
            })
    
    if missing_data:
        for item in missing_data[:10]:  # Show top 10
            print(f"  {item['Column']}: {item['Missing']} ({item['Percentage']})")
    else:
        print("  No missing values detected")
    
    # Duplicate check
    if 'Customer_ID' in df.columns and 'snapshot_date' in df.columns:
        distinct_keys = df.select("Customer_ID", "snapshot_date").distinct().count()
        duplicates = row_count - distinct_keys
        print(f"\nDuplicate Records: {duplicates}")
    
    return row_count, col_count


def analyze_numeric_features(df: DataFrame, table_name: str):
    """
    Analyze numeric columns
    """
    print(f"\n{'='*60}")
    print(f"Numeric Features Analysis: {table_name}")
    print(f"{'='*60}\n")
    
    # Get numeric columns
    numeric_cols = [c for c, t in df.dtypes 
                   if t in ['int', 'bigint', 'float', 'double', 'decimal']]
    
    if not numeric_cols:
        print("No numeric columns found")
        return
    
    # Show summary statistics for first few columns
    print("Summary Statistics (first 5 numeric columns):")
    sample_cols = numeric_cols[:5]
    
    for col_name in sample_cols:
        stats = df.select(col_name).summary("count", "mean", "min", "max")
        print(f"\n{col_name}:")
        stats.show()


def analyze_label_distribution(feature_store: DataFrame):
    """
    Analyze target label distribution
    """
    print(f"\n{'='*60}")
    print(f"Label Distribution Analysis")
    print(f"{'='*60}\n")
    
    if 'label' not in feature_store.columns:
        print("Label column not found")
        return
    
    # Count by label
    print("Label Counts:")
    label_counts = feature_store.groupBy("label").count().orderBy("label")
    label_counts.show()
    
    # Calculate class balance
    total = feature_store.count()
    label_stats = feature_store.groupBy("label").count().collect()
    
    print("\nClass Balance:")
    for row in label_stats:
        label_val = row['label']
        count = row['count']
        percentage = (count / total) * 100
        print(f"  Label {label_val}: {count} ({percentage:.2f}%)")


def run_basic_eda(spark, datamart_dir="datamart", snapshot_date="2023-01-01"):
    """
    Run basic EDA on pipeline outputs
    """
    print("\n" + "="*60)
    print("RUNNING BASIC EDA")
    print("="*60)
    
    date_key = snapshot_date.replace("-", "_")
    
    # Analyze Bronze layer
    print("\n### BRONZE LAYER ANALYSIS ###")
    try:
        bronze_loan = spark.read.parquet(
            f"{datamart_dir}/bronze/loan_daily/bronze_loan_daily_{date_key}.parquet"
        )
        profile_data_quality(bronze_loan, "Bronze Loan Daily")
    except Exception as e:
        print(f"Could not analyze bronze loan: {e}")
    
    # Analyze Silver layer
    print("\n### SILVER LAYER ANALYSIS ###")
    try:
        silver_loan = spark.read.parquet(
            f"{datamart_dir}/silver/loan_daily/silver_loan_daily_{date_key}.parquet"
        )
        profile_data_quality(silver_loan, "Silver Loan Daily")
        
        # Check new columns added in silver
        print("\nSilver Layer New Columns:")
        silver_specific = ['mob', 'dpd', 'installments_missed', 'first_missed_date']
        for col in silver_specific:
            if col in silver_loan.columns:
                print(f"  ✓ {col}")
    except Exception as e:
        print(f"Could not analyze silver loan: {e}")
    
    # Analyze Gold layer
    print("\n### GOLD LAYER ANALYSIS ###")
    try:
        gold_features = spark.read.parquet(
            f"{datamart_dir}/gold/feature_store/gold_feature_store_{date_key}.parquet"
        )
        rows, cols = profile_data_quality(gold_features, "Gold Feature Store")
        
        # Analyze features
        print("\nFeature Categories:")
        
        fe_cols = [c for c in gold_features.columns if c.startswith("fe_")]
        print(f"  Clickstream features (fe_*): {len(fe_cols)}")
        
        if 'click_sum' in gold_features.columns:
            print(f"  ✓ Aggregated feature: click_sum")
        
        if 'income_per_click' in gold_features.columns:
            print(f"  ✓ Derived feature: income_per_click")
        
        # Label analysis
        analyze_label_distribution(gold_features)
        
    except Exception as e:
        print(f"Could not analyze gold features: {e}")
    
    print("\n" + "="*60)
    print("EDA COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    """
    Standalone EDA execution
    Usage: Can be imported or run directly after pipeline completion
    """
    print("EDA module loaded. Use run_basic_eda(spark) to analyze pipeline outputs.")