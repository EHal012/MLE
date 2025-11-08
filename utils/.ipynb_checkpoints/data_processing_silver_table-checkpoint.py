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
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-', '_') + ".parquet"
    filepath = os.path.join(bronze_lms_directory, partition_name)
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    column_type_map = {
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
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    return df


# =========================
# 2) 新增：三个 feature 表的 Silver 清洗
# =========================

# 常见“乱码/脏字符”
GARBAGE_CHARS = ["Â", "ï»¿", "\uFEFF", "\u200B", "\u200C", "\u200D", "\u00A0"]  # BOM/零宽/NBSP

def _coerce_to_date(df: DataFrame, colname="snapshot_date") -> DataFrame:
    c = F.col(colname).cast("string")
    return df.withColumn(
        colname,
        F.coalesce(
            F.to_date(c, "yyyy-MM-dd"),
            F.to_date(c, "yyyy-M-d"),
            F.to_date(c, "yyyy/MM/dd"),
            F.to_date(c, "yyyy/M/d"),
        )
    )

def _standardize_id_date(df: DataFrame) -> DataFrame:
    if "Customer_ID" not in df.columns:
        for cand in ["customer_id", "CustomerId", "customerId"]:
            if cand in df.columns:
                df = df.withColumnRenamed(cand, "Customer_ID")
                break
    if "snapshot_date" not in df.columns:
        for cand in ["Snapshot_Date", "date", "Date"]:
            if cand in df.columns:
                df = df.withColumnRenamed(cand, "snapshot_date")
                break
    df = df.withColumn("Customer_ID", F.col("Customer_ID").cast(StringType()))
    df = _coerce_to_date(df, "snapshot_date")
    return df

def _strip_garbage_text(df: DataFrame, cols) -> DataFrame:
    control = r"[\p{C}]"
    for c in cols:
        x = F.col(c).cast("string")
        x = F.regexp_replace(x, control, "")
        for g in GARBAGE_CHARS:
            x = F.regexp_replace(x, re.escape(g), " ")
        x = F.regexp_replace(x, r"\s+", " ")
        x = F.trim(x)
        x = F.when(F.length(x) == 0, None).otherwise(x)
        df = df.withColumn(c, x)
    return df

def _normalize_underscores_and_suffix(df: DataFrame, cols) -> DataFrame:
    for c in cols:
        x = F.col(c).cast("string")
        x = F.regexp_replace(x, r"_+", "_")
        x = F.regexp_replace(x, r"^_+|_+$", "")
        x = F.regexp_replace(x, r"_p$", "")
        df = df.withColumn(c, F.when(F.length(x) == 0, None).otherwise(x))
    return df

def _coerce_numeric_from_text(df: DataFrame, cols, to="double") -> DataFrame:
    for c in cols:
        x = F.col(c).cast("string")
        x = F.regexp_replace(x, r"[%$,]", "")
        x = F.regexp_replace(x, r"\s+", "")
        x = F.when(F.length(x) == 0, None).otherwise(x)
        df = df.withColumn(c, x.cast(DoubleType() if to == "double" else IntegerType()))
    return df

def _dedup(df: DataFrame) -> DataFrame:
    return df.dropDuplicates(["Customer_ID", "snapshot_date"])

def _save(df: DataFrame, outdir: str, name_prefix: str, date_str: str):
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{name_prefix}_{date_str.replace('-','_')}.parquet")
    df.write.mode("overwrite").parquet(out)
    print("saved:", out)

# ----- 2.1 clickstream -----
def process_silver_feature_clickstream(snapshot_date_str, bronze_dir, silver_dir, spark):
    inpath = os.path.join(bronze_dir, f"bronze_feature_clickstream_{snapshot_date_str.replace('-','_')}.parquet")
    df = spark.read.parquet(inpath)
    df = _standardize_id_date(df)

    # 文本清洗（包含 Customer_ID 在内的所有 string 列）
    text_cols = [c for c, t in df.dtypes if t == "string"]
    df = _strip_garbage_text(df, text_cols)
    df = _normalize_underscores_and_suffix(df, text_cols)

    # fe_* 数值列转 double（不做派生）
    fe_cols = [c for c in df.columns if c.startswith("fe_")]
    df = _coerce_numeric_from_text(df, fe_cols, to="double")

    df = _dedup(df)
    _save(df, silver_dir, "silver_feature_clickstream", snapshot_date_str)
    return df

# ----- 2.2 attributes -----
def process_silver_feature_attributes(snapshot_date_str, bronze_dir, silver_dir, spark):
    inpath = os.path.join(bronze_dir, f"bronze_feature_attributes_{snapshot_date_str.replace('-','_')}.parquet")
    df = spark.read.parquet(inpath)
    df = _standardize_id_date(df)

    # 文本清洗（Name/Occupation 等可能含乱码/下划线）
    text_cols = [c for c, t in df.dtypes if t == "string"]
    df = _strip_garbage_text(df, text_cols)
    df = _normalize_underscores_and_suffix(df, text_cols)

    # 可能的数值列（如 Age）统一为数字
    maybe_numeric = [c for c in df.columns if re.search(r"(?i)(age|num|count|score)$", c)]
    df = _coerce_numeric_from_text(df, maybe_numeric, to="double")

    df = _dedup(df)
    _save(df, silver_dir, "silver_feature_attributes", snapshot_date_str)
    return df

# ----- 2.3 financials -----
def process_silver_feature_financials(snapshot_date_str, bronze_dir, silver_dir, spark):
    inpath = os.path.join(bronze_dir, f"bronze_feature_financials_{snapshot_date_str.replace('-','_')}.parquet")
    df = spark.read.parquet(inpath)
    df = _standardize_id_date(df)

    # 文本列清洗（去乱码、下划线规范、去 _p）
    text_cols = [c for c, t in df.dtypes if t == "string"]
    df = _strip_garbage_text(df, text_cols)
    df = _normalize_underscores_and_suffix(df, text_cols)

    # 财务数值列：金额/比率/计数等转 double
    maybe_numeric = [c for c in df.columns if re.search(r"(?i)(amount|amt|income|expense|spend|balance|ratio|rate|limit|num|count|score)$", c)]
    df = _coerce_numeric_from_text(df, maybe_numeric, to="double")

    df = _dedup(df)
    _save(df, silver_dir, "silver_feature_financials", snapshot_date_str)
    return df
