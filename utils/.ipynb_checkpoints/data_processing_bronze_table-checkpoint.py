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
    return (
        df
        .withColumn(colname, F.regexp_replace(F.col(colname).cast("string"), r"[./]", "-"))
        .withColumn(colname, F.to_date(F.col(colname), "yyyy-M-d"))
    )

def _filter_by_snapshot(df: DataFrame, snapshot_date_str: str) -> DataFrame:
    snap = datetime.strptime(snapshot_date_str, "%Y-%m-%d").date()
    return df.filter(F.col("snapshot_date") == F.lit(snap))

def _ensure_dirs(path: str):
    os.makedirs(path, exist_ok=True)

def _save_parquet(df: DataFrame, out_dir: str, name_prefix: str, snapshot_date_str: str):
    _ensure_dirs(out_dir)
    fname = f"{name_prefix}_{snapshot_date_str.replace('-', '_')}.parquet"
    df.write.mode("overwrite").parquet(os.path.join(out_dir, fname))

# ---- Bronze: loan_daily ----
def process_bronze_loan_daily(snapshot_date_str: str,
                              bronze_lms_directory: str,
                              spark: pyspark.sql.SparkSession) -> DataFrame:
    src = "data/lms_loan_daily.csv"
    df = spark.read.csv(src, header=True, inferSchema=True)
    df = _normalize_snapshot_date(df, "snapshot_date")
    df = _filter_by_snapshot(df, snapshot_date_str)
    _save_parquet(df, bronze_lms_directory, "bronze_loan_daily", snapshot_date_str)
    return df

# ---- Bronze: feature_clickstream ----
def process_bronze_feature_clickstream(snapshot_date_str: str,
                                       bronze_clickstream_directory: str,
                                       spark: pyspark.sql.SparkSession) -> DataFrame:
    src = "data/feature_clickstream.csv"
    df = spark.read.csv(src, header=True, inferSchema=True)
    df = _normalize_snapshot_date(df, "snapshot_date")
    df = df.withColumnRenamed("Customer_ID", "Customer_ID")
    df = _filter_by_snapshot(df, snapshot_date_str)
    _save_parquet(df, bronze_clickstream_directory, "bronze_feature_clickstream", snapshot_date_str)
    return df

# ---- Bronze: feature_attributes ----
def process_bronze_feature_attributes(snapshot_date_str: str,
                                      bronze_attributes_directory: str,
                                      spark: pyspark.sql.SparkSession) -> DataFrame:
    src = "data/features_attributes.csv"
    df = spark.read.csv(src, header=True, inferSchema=True)
    df = _normalize_snapshot_date(df, "snapshot_date")
    df = df.withColumnRenamed("Customer_ID", "Customer_ID")
    df = _filter_by_snapshot(df, snapshot_date_str)
    _save_parquet(df, bronze_attributes_directory, "bronze_feature_attributes", snapshot_date_str)
    return df

# ---- Bronze: feature_financials ----
def process_bronze_feature_financials(snapshot_date_str: str,
                                      bronze_financials_directory: str,
                                      spark: pyspark.sql.SparkSession) -> DataFrame:
    src = "data/features_financials.csv"
    df = spark.read.csv(src, header=True, inferSchema=True)
    df = _normalize_snapshot_date(df, "snapshot_date")
    df = df.withColumnRenamed("Customer_ID", "Customer_ID")
    df = _filter_by_snapshot(df, snapshot_date_str)
    _save_parquet(df, bronze_financials_directory, "bronze_feature_financials", snapshot_date_str)
    return df
