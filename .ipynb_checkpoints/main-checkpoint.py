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

SNAPSHOT_START = "2023-01-01"
SNAPSHOT_END   = "2023-12-01"   # 包含这个月
# Lab2 的标签你已复制到 Assign1/datamart/gold/label_store/
LABEL_STORE_DIR = "datamart/gold/label_store"

# 数据湖目录
BRONZE_DIRS = {
    "loan":        "datamart/bronze/loan_daily/",
    "clickstream": "datamart/bronze/feature_clickstream/",
    "attributes":  "datamart/bronze/feature_attributes/",
    "financials":  "datamart/bronze/feature_financials/",
}
SILVER_DIRS = {
    "loan":        "datamart/silver/loan_daily/",
    "clickstream": "datamart/silver/feature_clickstream/",
    "attributes":  "datamart/silver/feature_attributes/",
    "financials":  "datamart/silver/feature_financials/",
}
GOLD_DIRS = {
    "feature_store": "datamart/gold/feature_store/",
}

def ensure_dirs():
    for p in list(BRONZE_DIRS.values()) + list(SILVER_DIRS.values()) + list(GOLD_DIRS.values()) + [LABEL_STORE_DIR]:
        os.makedirs(p, exist_ok=True)

def first_of_month_dates(start_str: str, end_str: str):
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end   = datetime.strptime(end_str, "%Y-%m-%d")
    cur = datetime(start.year, start.month, 1)
    out = []
    while cur <= end:
        out.append(cur.strftime("%Y-%m-%d"))
        # next month
        if cur.month == 12:
            cur = datetime(cur.year + 1, 1, 1)
        else:
            cur = datetime(cur.year, cur.month + 1, 1)
    return out

if __name__ == "__main__":
    # Spark
    spark: SparkSession = (
        SparkSession.builder
        .appName("assign1_pipeline")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    ensure_dirs()
    dates = first_of_month_dates(SNAPSHOT_START, SNAPSHOT_END)
    print(f"[INFO] Months to process: {dates}")

    # -------- Bronze backfill --------
    for d in dates:
        # 1) loan_daily（Lab2 同源）
        bronze.process_bronze_loan_daily(
            snapshot_date_str=d,
            bronze_lms_directory=BRONZE_DIRS["loan"],
            spark=spark
        )
        # 2) 三个新增 feature
        bronze.process_bronze_feature_clickstream(
            snapshot_date_str=d,
            bronze_clickstream_directory=BRONZE_DIRS["clickstream"],
            spark=spark
        )
        bronze.process_bronze_feature_attributes(
            snapshot_date_str=d,
            bronze_attributes_directory=BRONZE_DIRS["attributes"],
            spark=spark
        )
        bronze.process_bronze_feature_financials(
            snapshot_date_str=d,
            bronze_financials_directory=BRONZE_DIRS["financials"],
            spark=spark
        )

    print("[INFO] Bronze completed.")

    # -------- Silver backfill --------
    for d in dates:
        # 保留你原来的 loan_daily（含 mob/dpd 的版本）
        silver.process_silver_table(
            snapshot_date_str=d,
            bronze_lms_directory=BRONZE_DIRS["loan"],
            silver_loan_daily_directory=SILVER_DIRS["loan"],
            spark=spark
        )
        # 新增三张 feature 的清洗（不做派生）
        silver.process_silver_feature_clickstream(
            snapshot_date_str=d,
            bronze_dir=BRONZE_DIRS["clickstream"],
            silver_dir=SILVER_DIRS["clickstream"],
            spark=spark
        )
        silver.process_silver_feature_attributes(
            snapshot_date_str=d,
            bronze_dir=BRONZE_DIRS["attributes"],
            silver_dir=SILVER_DIRS["attributes"],
            spark=spark
        )
        silver.process_silver_feature_financials(
            snapshot_date_str=d,
            bronze_dir=BRONZE_DIRS["financials"],
            silver_dir=SILVER_DIRS["financials"],
            spark=spark
        )

    print("[INFO] Silver completed.")

    # -------- Gold: build feature store using Lab2 labels --------
    for d in dates:
        key = d.replace("-", "_")
        # labels 必须已复制到 Assign1/datamart/gold/label_store 下
        label_file = os.path.join(LABEL_STORE_DIR, f"gold_label_store_{key}.parquet")
        if not os.path.exists(label_file):
            print(f"[WARN] Label file not found for {d}: {label_file}. Skip this month.")
            continue

        gold.build_gold_feature_store(
            snapshot_date_str=d,
            silver_clickstream_dir=SILVER_DIRS["clickstream"],
            silver_attributes_dir=SILVER_DIRS["attributes"],
            silver_financials_dir=SILVER_DIRS["financials"],
            label_store_dir=LABEL_STORE_DIR,                  # 直接复用 Lab2 标签（你已复制过来）
            gold_feature_store_dir=GOLD_DIRS["feature_store"],
            spark=spark
        )

    print("[INFO] Gold completed. All done ✅")
    spark.stop()
