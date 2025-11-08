import os, re
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

from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DoubleType, DateType


def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    """
    Process loan daily data for silver layer
    Different implementation: More modular column processing
    """
    # Parse snapshot date
    snapshot_dt = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # Build bronze file path
    date_key = snapshot_date_str.replace('-', '_')
    bronze_file = f"bronze_loan_daily_{date_key}.parquet"
    input_path = os.path.join(bronze_lms_directory, bronze_file)
    
    # Load bronze data
    df = spark.read.parquet(input_path)
    print(f'[Silver] Loaded from: {input_path}, rows: {df.count()}')

    # Define schema mapping
    type_mappings = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }
    
    # Apply type casting in batches
    for col_name, target_type in type_mappings.items():
        if col_name in df.columns:
            df = df.withColumn(col_name, col(col_name).cast(target_type))

    # Feature 1: Month on Book (MOB)
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # Feature 2: Calculate installments missed
    missed_calc = F.when(col("due_amt") > 0, 
                         F.ceil(col("overdue_amt") / col("due_amt")))\
                   .otherwise(F.lit(0))
    df = df.withColumn("installments_missed", missed_calc.cast(IntegerType()))
    
    # Handle null values
    df = df.fillna({"installments_missed": 0})
    
    # Feature 3: First missed payment date
    first_miss_expr = F.when(col("installments_missed") > 0,
                             F.add_months(col("snapshot_date"), 
                                         -1 * col("installments_missed")))
    df = df.withColumn("first_missed_date", first_miss_expr.cast(DateType()))
    
    # Feature 4: Days Past Due (DPD)
    dpd_expr = F.when(col("overdue_amt") > 0.0,
                     F.datediff(col("snapshot_date"), col("first_missed_date")))\
                .otherwise(F.lit(0))
    df = df.withColumn("dpd", dpd_expr.cast(IntegerType()))

    # Save silver table
    silver_file = f"silver_loan_daily_{date_key}.parquet"
    output_path = silver_loan_daily_directory + silver_file
    
    df.write.mode("overwrite").parquet(output_path)
    print(f'[Silver] Saved to: {output_path}')
    
    return df

# Problematic character patterns to remove
CLEANUP_PATTERNS = ["Ã‚", "Ã¯Â»Â¿", "\uFEFF", "\u200B", "\u200C", "\u200D", "\u00A0"]

def _coerce_to_date(df: DataFrame, colname="snapshot_date") -> DataFrame:
    """
    Flexible date parsing
    Different implementation: Using case-when logic
    """
    str_col = F.col(colname).cast("string")
    
    # Try multiple formats sequentially
    date_result = F.to_date(str_col, "yyyy-MM-dd")
    
    # Add fallback formats
    fallback_formats = ["yyyy-M-d", "yyyy/MM/dd", "yyyy/M/d"]
    for fmt in fallback_formats:
        date_result = F.when(date_result.isNull(), F.to_date(str_col, fmt))\
                       .otherwise(date_result)
    
    return df.withColumn(colname, date_result)

def _standardize_id_date(df: DataFrame) -> DataFrame:
    """
    Standardize ID and date columns
    Different implementation: Dictionary-based approach
    """
    # Customer ID standardization
    id_variants = {
        "customer_id": "Customer_ID",
        "CustomerId": "Customer_ID", 
        "customerId": "Customer_ID"
    }
    
    for old_name, new_name in id_variants.items():
        if old_name in df.columns and old_name != new_name:
            df = df.withColumnRenamed(old_name, new_name)
            break
    
    # Date column standardization
    date_variants = {
        "Snapshot_Date": "snapshot_date",
        "date": "snapshot_date",
        "Date": "snapshot_date"
    }
    
    for old_name, new_name in date_variants.items():
        if old_name in df.columns and old_name != new_name:
            df = df.withColumnRenamed(old_name, new_name)
            break
    
    # Type casting
    if "Customer_ID" in df.columns:
        df = df.withColumn("Customer_ID", F.col("Customer_ID").cast(StringType()))
    
    if "snapshot_date" in df.columns:
        df = _coerce_to_date(df, "snapshot_date")
    
    return df

def _strip_garbage_text(df: DataFrame, cols) -> DataFrame:
    """
    Remove problematic characters from text
    Different implementation: Iterative cleaning
    """
    unicode_ctrl = r"[\p{C}]"
    
    for col_name in cols:
        text_val = F.col(col_name).cast("string")
        
        # Step 1: Remove control characters
        text_val = F.regexp_replace(text_val, unicode_ctrl, "")
        
        # Step 2: Clean known garbage patterns
        for pattern in CLEANUP_PATTERNS:
            escaped = re.escape(pattern)
            text_val = F.regexp_replace(text_val, escaped, " ")
        
        # Step 3: Normalize whitespace
        text_val = F.regexp_replace(text_val, r"\s+", " ")
        text_val = F.trim(text_val)
        
        # Step 4: Empty string to null
        text_val = F.when(F.length(text_val) == 0, None).otherwise(text_val)
        
        df = df.withColumn(col_name, text_val)
    
    return df

def _normalize_underscores_and_suffix(df: DataFrame, cols) -> DataFrame:
    """
    Clean up underscore patterns
    Different implementation: Sequential replacements
    """
    for col_name in cols:
        text_val = F.col(col_name).cast("string")
        
        # Replace multiple underscores
        text_val = F.regexp_replace(text_val, r"_+", "_")
        
        # Remove leading underscores
        text_val = F.regexp_replace(text_val, r"^_+", "")
        
        # Remove trailing underscores
        text_val = F.regexp_replace(text_val, r"_+$", "")
        
        # Remove _p suffix
        text_val = F.regexp_replace(text_val, r"_p$", "")
        
        # Convert empty to null
        text_val = F.when(F.length(text_val) == 0, None).otherwise(text_val)
        
        df = df.withColumn(col_name, text_val)
    
    return df

def _coerce_numeric_from_text(df: DataFrame, cols, to="double") -> DataFrame:
    """
    Convert text to numeric
    Different implementation: Pipeline approach
    """
    target_type = DoubleType() if to == "double" else IntegerType()
    
    for col_name in cols:
        num_val = F.col(col_name).cast("string")
        
        # Remove currency and percentage symbols
        num_val = F.regexp_replace(num_val, r"[$%,]", "")
        
        # Remove all whitespace
        num_val = F.regexp_replace(num_val, r"\s+", "")
        
        # Handle empty strings
        num_val = F.when(F.length(num_val) == 0, None).otherwise(num_val)
        
        # Cast to target type
        num_val = num_val.cast(target_type)
        
        df = df.withColumn(col_name, num_val)
    
    return df

def _dedup(df: DataFrame) -> DataFrame:
    """Remove duplicates based on key columns"""
    before = df.count()
    df_dedup = df.dropDuplicates(["Customer_ID", "snapshot_date"])
    after = df_dedup.count()
    
    if before != after:
        print(f"[Silver] Removed {before - after} duplicates")
    
    return df_dedup

def _save(df: DataFrame, outdir: str, name_prefix: str, date_str: str):
    """
    Save silver table
    Different implementation: Explicit path construction
    """
    # Ensure directory
    os.makedirs(outdir, exist_ok=True)
    
    # Build filename
    date_suffix = date_str.replace('-','_')
    filename = f"{name_prefix}_{date_suffix}.parquet"
    
    # Full path
    fullpath = os.path.join(outdir, filename)
    
    # Write
    df.write.mode("overwrite").parquet(fullpath)
    print(f"[Silver] Saved: {fullpath}")

# ----- 2.1 clickstream -----
def process_silver_feature_clickstream(snapshot_date_str, bronze_dir, silver_dir, spark):
    """
    Process clickstream silver layer
    Different implementation: Separate text and numeric processing
    """
    # Load bronze
    date_key = snapshot_date_str.replace('-','_')
    infile = f"bronze_feature_clickstream_{date_key}.parquet"
    inpath = os.path.join(bronze_dir, infile)
    
    df = spark.read.parquet(inpath)
    print(f"[Silver] Processing clickstream: {df.count()} rows")
    
    # Standardize columns
    df = _standardize_id_date(df)

    # Get text columns
    text_columns = [name for name, dtype in df.dtypes if dtype == "string"]
    
    # Clean text
    df = _strip_garbage_text(df, text_columns)
    df = _normalize_underscores_and_suffix(df, text_columns)

    # Get feature columns
    feature_columns = [name for name in df.columns if name.startswith("fe_")]
    
    # Convert features to numeric
    df = _coerce_numeric_from_text(df, feature_columns, to="double")

    # Remove duplicates
    df = _dedup(df)
    
    # Save
    _save(df, silver_dir, "silver_feature_clickstream", snapshot_date_str)
    
    return df

# ----- 2.2 attributes -----
def process_silver_feature_attributes(snapshot_date_str, bronze_dir, silver_dir, spark):
    """
    Process attributes silver layer
    Different implementation: Pattern matching for numeric fields
    """
    # Load bronze
    date_key = snapshot_date_str.replace('-','_')
    infile = f"bronze_feature_attributes_{date_key}.parquet"
    inpath = os.path.join(bronze_dir, infile)
    
    df = spark.read.parquet(inpath)
    print(f"[Silver] Processing attributes: {df.count()} rows")
    
    # Standardize
    df = _standardize_id_date(df)

    # Clean text columns
    text_columns = [name for name, dtype in df.dtypes if dtype == "string"]
    df = _strip_garbage_text(df, text_columns)
    df = _normalize_underscores_and_suffix(df, text_columns)

    # Find numeric-like columns by pattern
    numeric_pattern = r"(?i)(age|num|count|score)$"
    potential_numeric = [name for name in df.columns if re.search(numeric_pattern, name)]
    
    # Convert to numeric
    df = _coerce_numeric_from_text(df, potential_numeric, to="double")

    # Deduplicate
    df = _dedup(df)
    
    # Save
    _save(df, silver_dir, "silver_feature_attributes", snapshot_date_str)
    
    return df

# ----- 2.3 financials -----
def process_silver_feature_financials(snapshot_date_str, bronze_dir, silver_dir, spark):
    """
    Process financials silver layer
    Different implementation: Comprehensive pattern matching
    """
    # Load bronze
    date_key = snapshot_date_str.replace('-','_')
    infile = f"bronze_feature_financials_{date_key}.parquet"
    inpath = os.path.join(bronze_dir, infile)
    
    df = spark.read.parquet(inpath)
    print(f"[Silver] Processing financials: {df.count()} rows")
    
    # Standardize
    df = _standardize_id_date(df)

    # Clean text
    text_columns = [name for name, dtype in df.dtypes if dtype == "string"]
    df = _strip_garbage_text(df, text_columns)
    df = _normalize_underscores_and_suffix(df, text_columns)

    # Pattern for financial numeric fields
    finance_pattern = r"(?i)(amount|amt|income|expense|spend|balance|ratio|rate|limit|num|count|score)$"
    potential_numeric = [name for name in df.columns if re.search(finance_pattern, name)]
    
    # Convert to numeric
    df = _coerce_numeric_from_text(df, potential_numeric, to="double")

    # Deduplicate
    df = _dedup(df)
    
    # Save
    _save(df, silver_dir, "silver_feature_financials", snapshot_date_str)
    
    return df