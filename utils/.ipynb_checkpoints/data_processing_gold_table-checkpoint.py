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
    label_store_dir: str,               # ← Assign1/datamart/gold/label_store
    gold_feature_store_dir: str,        # ← 输出目录：Assign1/datamart/gold/feature_store
    spark
):
    key = snapshot_date_str.replace("-", "_")

    # 1) 读取三张 Silver 特征
    cs_path   = os.path.join(silver_clickstream_dir, f"silver_feature_clickstream_{key}.parquet")
    attr_path = os.path.join(silver_attributes_dir,  f"silver_feature_attributes_{key}.parquet")
    fin_path  = os.path.join(silver_financials_dir,  f"silver_feature_financials_{key}.parquet")

    cs   = spark.read.parquet(cs_path)
    attr = spark.read.parquet(attr_path)
    fin  = spark.read.parquet(fin_path)

    def prep(df):
        return (df
            .withColumn("Customer_ID", F.col("Customer_ID").cast("string"))
            .withColumn("snapshot_date", F.to_date("snapshot_date"))
        )

    cs, attr, fin = map(prep, [cs, attr, fin])

    # 2) Gold 层的规则型特征工程
    # 2.1 clickstream：把 fe_* 做一个当期总和（仅用本表当前快照）
    fe_cols = [c for c in cs.columns if c.startswith("fe_")]
    if fe_cols:
        expr_sum = None
        for c in fe_cols:
            expr_sum = F.col(c) if expr_sum is None else expr_sum + F.col(c)
        cs = cs.withColumn("click_sum", expr_sum.cast("double"))

    # 2.2 financials × clickstream 的交互：收入/点击（0 除保护）
    if "Annual_Income" in fin.columns:
        fin = fin.withColumn("Annual_Income", F.col("Annual_Income").cast("double"))
    # 交互项稍后在 join 后也行，这里演示在 fin 侧先不做

    # 3) 读取标签（Lab2 的 label_store，已复制到 Assign1）
    lbl_path = os.path.join(label_store_dir, f"gold_label_store_{key}.parquet")
    labels = (spark.read.parquet(lbl_path)
        .withColumn("Customer_ID", F.col("Customer_ID").cast("string"))
        .withColumn("snapshot_date", F.to_date("snapshot_date"))
        .select("Customer_ID", "snapshot_date", "label", "label_def")
    )

    # 4) 合并（建议 left：保留所有有标签的样本）
    joined = (labels
        .join(cs,   on=["Customer_ID", "snapshot_date"], how="left")
        .join(attr, on=["Customer_ID", "snapshot_date"], how="left")
        .join(fin,  on=["Customer_ID", "snapshot_date"], how="left")
    )

    # 5) 在合并后的大表上做交互（示例：收入/点击，0 除保护）
    if "Annual_Income" in joined.columns and "click_sum" in joined.columns:
        joined = joined.withColumn(
            "income_per_click",
            F.when((F.col("click_sum").isNotNull()) & (F.col("click_sum") != 0),
                   F.col("Annual_Income") / F.col("click_sum"))
             .otherwise(None)
        )

    # 去重 + 基本对齐
    joined = joined.dropDuplicates(["Customer_ID", "snapshot_date"])

    # 6) 写出 Gold Feature Store
    os.makedirs(gold_feature_store_dir, exist_ok=True)
    out_path = os.path.join(gold_feature_store_dir, f"gold_feature_store_{key}.parquet")
    joined.write.mode("overwrite").parquet(out_path)
    print("Saved feature store:", out_path)

    return joined