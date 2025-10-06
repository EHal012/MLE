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

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table





##load data
# Clickstream features
feature_clickstream_df = spark.read.option("header", True).csv("data/feature_clickstream.csv")

# Customer attributes
features_attributes_df = spark.read.option("header", True).csv("data/features_attributes.csv")

# Financial features
features_financials_df = spark.read.option("header", True).csv("data/features_financials.csv")





##define functions for EDA
import os
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import *

class FeatureEDA:
    """
    filter + EDA
    """
    
    def __init__(self, spark):
        self.spark = spark
        
    def load_features(self, data_folder):
        """
        load
        """
        print("="*80)
        print("STEP 1: Loading Feature Datasets")
        print("="*80)
        
        # clickstream
        clickstream_path = os.path.join(data_folder, "feature_clickstream.csv")
        print(f"\nLoading: {clickstream_path}")
        df_click = self.spark.read.option("header", "true").csv(clickstream_path)
        df_click = df_click.withColumn("snapshot_date", 
                                       F.to_date(F.col("snapshot_date"), "yyyy/M/d"))
        print(f"  ✓ Clickstream: {df_click.count()} rows, {len(df_click.columns)} columns")
        print(f"  Columns: {df_click.columns}")
        
        # attributes
        attributes_path = os.path.join(data_folder, "features_attributes.csv")
        print(f"\nLoading: {attributes_path}")
        df_attr = self.spark.read.option("header", "true").csv(attributes_path)
        df_attr = df_attr.withColumn("snapshot_date", 
                                     F.to_date(F.col("snapshot_date"), "yyyy/M/d"))
        print(f"  ✓ Attributes: {df_attr.count()} rows, {len(df_attr.columns)} columns")
        print(f"  Columns: {df_attr.columns}")
        
        # financials
        financials_path = os.path.join(data_folder, "features_financials.csv")
        print(f"\nLoading: {financials_path}")
        df_fin = self.spark.read.option("header", "true").csv(financials_path)
        df_fin = df_fin.withColumn("snapshot_date", 
                                   F.to_date(F.col("snapshot_date"), "yyyy/M/d"))
        print(f"  ✓ Financials: {df_fin.count()} rows, {len(df_fin.columns)} columns")
        print(f"  Columns: {df_fin.columns}")
        
        return df_click, df_attr, df_fin
    
    def define_feature_categories(self, df_click, df_attr, df_fin):
        """
        Classify
        """
        print("\n" + "="*80)
        print("STEP 2: Categorizing Features by Business Logic")
        print("="*80)
        
       
        feature_categories = {
            '❌ EXCLUDE - Identity/Name': [
                'Name', 'SSN' 
            ],
            
            '❌ EXCLUDE - Data Leakage': [
                'Num_of_Delayed_Payment',  
                'Delay_from_due_date',     
                'Payment_of_Min_Amount',    
            ],
            
            '✅ KEEP - Demographic': [
                'Age',          
                'Occupation',   
            ],
            
            '✅ KEEP - Financial Capacity': [
                'Annual_Income',           
                'Monthly_Inhand_Salary',   
                'Monthly_Balance',         
                'Amount_invested_monthly', 
            ],
            
            '✅ KEEP - Credit Behavior': [
                'Num_Bank_Accounts',      
                'Num_Credit_Card',       
                'Interest_Rate',          
                'Num_of_Loan',            
                'Type_of_Loan',           
                'Changed_Credit_Limit',   
                'Num_Credit_Inquiries',  
                'Credit_Mix',            
                'Outstanding_Debt',       
                'Credit_Utilization_Ratio', 
                'Credit_History_Age',     
            ],
            
            '✅ KEEP - Payment Behavior': [
                'Total_EMI_per_month',    
                'Payment_Behaviour',       
            ],
            
            '✅ KEEP - Clickstream': []  
        }
        
        # clickstream
        clickstream_features = [col for col in df_click.columns 
                               if col.startswith('fe_')]
        feature_categories['✅ KEEP - Clickstream'] = clickstream_features
        
        
        keep_features = []
        exclude_features = []
        
        print("\n📋 Feature Categories:")
        for category, features in feature_categories.items():
            print(f"\n{category} ({len(features)} features):")
            for feat in features:
                print(f"  - {feat}")
            
            if category.startswith('✅'):
                keep_features.extend(features)
            else:
                exclude_features.extend(features)
        
        print(f"\n{'='*80}")
        print(f"✅ Total features to KEEP: {len(keep_features)}")
        print(f"❌ Total features to EXCLUDE: {len(exclude_features)}")
        print(f"{'='*80}")
        
        return keep_features, exclude_features, feature_categories
    
    def comprehensive_eda(self, df_click, df_attr, df_fin, keep_features):
        """
        EDA
        """
        print("\n" + "="*80)
        print("STEP 3: Comprehensive EDA for Selected Features")
        print("="*80)
        
        eda_results = {}
        
        
        all_dfs = {
            'clickstream': df_click,
            'attributes': df_attr,
            'financials': df_fin
        }
        
        for dataset_name, df in all_dfs.items():
            print(f"\n{'='*80}")
            print(f"📊 Analyzing {dataset_name.upper()}")
            print(f"{'='*80}")
            
            
            dataset_features = [col for col in df.columns 
                              if col in keep_features]
            
            if len(dataset_features) == 0:
                print(f"  No features to analyze in this dataset")
                continue
            
            dataset_results = []
            
            for idx, col in enumerate(dataset_features, 1):
                print(f"\n[{idx}/{len(dataset_features)}] {col}")
                print("-" * 60)
                
                feature_info = {
                    'feature_name': col,
                    'dataset': dataset_name
                }
                
                
                total = df.count()
                null_count = df.filter(F.col(col).isNull() | (F.col(col) == "")).count()
                null_rate = (null_count / total) * 100
                unique_count = df.select(col).distinct().count()
                
                feature_info['total_records'] = total
                feature_info['null_count'] = null_count
                feature_info['null_rate'] = f"{null_rate:.2f}%"
                feature_info['unique_count'] = unique_count
                
               
                samples = df.select(col).limit(10).toPandas()[col].tolist()
                feature_info['sample_values'] = samples
                
                print(f"  Total records: {total:,}")
                print(f"  Missing: {null_count:,} ({null_rate:.2f}%)")
                print(f"  Unique values: {unique_count:,}")
                print(f"  Sample values: {samples[:5]}")
                
               
                if len(samples) > 0:
                    sample_non_null = df.filter(F.col(col).isNotNull() & (F.col(col) != "")) \
                                       .select(col).limit(100).toPandas()[col]
                    
                    if len(sample_non_null) > 0:
                       
                        cleaned = sample_non_null.astype(str).str.replace('[^0-9.-]', '', regex=True)
                        numeric_count = cleaned.str.match(r'^-?\d*\.?\d+$').sum()
                        is_numeric = (numeric_count / len(sample_non_null)) > 0.7
                        
                        if is_numeric:
                            feature_info['data_type'] = 'numeric'
                            
                            
                            df_clean = df.withColumn(
                                f"{col}_clean",
                                F.regexp_replace(F.col(col), "[^0-9.-]", "")
                            )
                            df_clean = df_clean.withColumn(
                                f"{col}_clean",
                                F.when(F.col(f"{col}_clean").rlike("^-?[0-9]*\\.?[0-9]+$"), 
                                      F.col(f"{col}_clean").cast(DoubleType()))
                                .otherwise(None)
                            )
                            
                            stats = df_clean.select(
                                F.min(f"{col}_clean").alias('min'),
                                F.max(f"{col}_clean").alias('max'),
                                F.mean(f"{col}_clean").alias('mean'),
                                F.stddev(f"{col}_clean").alias('stddev'),
                                F.expr(f"percentile_approx(`{col}_clean`, 0.25)").alias('q25'),
                                F.expr(f"percentile_approx(`{col}_clean`, 0.50)").alias('median'),
                                F.expr(f"percentile_approx(`{col}_clean`, 0.75)").alias('q75')
                            ).collect()[0]
                            
                            feature_info['statistics'] = {
                                'min': float(stats['min']) if stats['min'] is not None else None,
                                'max': float(stats['max']) if stats['max'] is not None else None,
                                'mean': float(stats['mean']) if stats['mean'] is not None else None,
                                'stddev': float(stats['stddev']) if stats['stddev'] is not None else None,
                                'q25': float(stats['q25']) if stats['q25'] is not None else None,
                                'median': float(stats['median']) if stats['median'] is not None else None,
                                'q75': float(stats['q75']) if stats['q75'] is not None else None
                            }
                            
                            print(f"  Type: NUMERIC")
                            if stats['min'] is not None:
                                print(f"    Min: {stats['min']:.2f}")
                                print(f"    Max: {stats['max']:.2f}")
                                print(f"    Mean: {stats['mean']:.2f}")
                                print(f"    Median: {stats['median']:.2f}")
                                print(f"    Std: {stats['stddev']:.2f}")
                                
                                
                                if null_rate > 30:
                                    print(f"  ⚠️  WARNING: High missing rate!")
                                if stats['stddev'] and stats['stddev'] < 0.01:
                                    print(f"  ⚠️  WARNING: Very low variance!")
                            
                        else:
                            feature_info['data_type'] = 'categorical'
                            
                           
                            top_values = df.groupBy(col).count() \
                                          .orderBy(F.desc("count")) \
                                          .limit(10) \
                                          .toPandas()
                            
                            feature_info['top_values'] = top_values.to_dict('records')
                            
                            print(f"  Type: CATEGORICAL")
                            print(f"  Top 10 values:")
                            for _, row in top_values.iterrows():
                                pct = (row['count'] / total) * 100
                                val = row[col] if row[col] else "[NULL/EMPTY]"
                                print(f"    {val}: {row['count']:,} ({pct:.1f}%)")
                            
                            
                            if unique_count > 100:
                                print(f"  ⚠️  WARNING: High cardinality!")
                            if null_rate > 30:
                                print(f"  ⚠️  WARNING: High missing rate!")
                
                dataset_results.append(feature_info)
            
            eda_results[dataset_name] = dataset_results
        
        return eda_results
    
    def generate_summary(self, eda_results, keep_features, exclude_features, feature_categories):
        """
        EDA report
        """
        print("\n" + "="*80)
        print("STEP 4: EDA Summary")
        print("="*80)
        
        
        all_features = []
        for dataset in eda_results.values():
            all_features.extend(dataset)
        
        numeric_features = [f for f in all_features if f.get('data_type') == 'numeric']
        categorical_features = [f for f in all_features if f.get('data_type') == 'categorical']
        
        print(f"\n📊 Feature Type Distribution:")
        print(f"  Total selected features: {len(keep_features)}")
        print(f"  Numeric features: {len(numeric_features)}")
        print(f"  Categorical features: {len(categorical_features)}")
        print(f"  Excluded features: {len(exclude_features)}")
        
        
        print(f"\n⚠️  Data Quality Issues:")
        high_missing = [f for f in all_features 
                       if float(f['null_rate'].rstrip('%')) > 30]
        print(f"  High missing rate (>30%): {len(high_missing)}")
        if high_missing:
            for f in high_missing:
                print(f"    - {f['feature_name']}: {f['null_rate']}")
        
        high_cardinality = [f for f in categorical_features 
                           if f['unique_count'] > 100]
        print(f"  High cardinality categorical (>100 unique): {len(high_cardinality)}")
        if high_cardinality:
            for f in high_cardinality:
                print(f"    - {f['feature_name']}: {f['unique_count']:,} unique values")
        
        
        print(f"\n📋 Feature Distribution by Category:")
        for category, features in feature_categories.items():
            if category.startswith('✅'):
                print(f"  {category}: {len(features)}")
        
        summary = {
            'total_selected': len(keep_features),
            'total_excluded': len(exclude_features),
            'numeric_count': len(numeric_features),
            'categorical_count': len(categorical_features),
            'high_missing_count': len(high_missing),
            'high_cardinality_count': len(high_cardinality),
            'feature_categories': feature_categories
        }
        
        return summary
    
    def save_results(self, keep_features, exclude_features, eda_results, summary):
        """
        save report
        """
        print("\n" + "="*80)
        print("STEP 5: Saving Results")
        print("="*80)
        
       
        with open("selected_features.txt", "w") as f:
            f.write("# Selected Features for Modeling\n")
            f.write(f"# Total: {len(keep_features)} features\n\n")
            for feat in keep_features:
                f.write(f"{feat}\n")
        print("✓ Saved: selected_features.txt")
        
        
        with open("excluded_features.txt", "w") as f:
            f.write("# Excluded Features\n")
            f.write(f"# Total: {len(exclude_features)} features\n\n")
            for feat in exclude_features:
                f.write(f"{feat}\n")
        print("✓ Saved: excluded_features.txt")
        
        #EDA report
        import json
        with open("eda_report.json", "w") as f:
            json.dump(eda_results, f, indent=2, default=str)
        print("✓ Saved: eda_report.json")
        
        
        with open("eda_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print("✓ Saved: eda_summary.json")
        
        print("\n✅ All results saved successfully!")
    
    def run_pipeline(self, data_folder):
        """
        pipeline
        """
        # Step 1
        df_click, df_attr, df_fin = self.load_features(data_folder)
        
        # Step 2
        keep_features, exclude_features, feature_categories = \
            self.define_feature_categories(df_click, df_attr, df_fin)
        
        # Step 3
        eda_results = self.comprehensive_eda(df_click, df_attr, df_fin, keep_features)
        
        # Step 4
        summary = self.generate_summary(eda_results, keep_features, exclude_features, feature_categories)
        
        # Step 5
        self.save_results(keep_features, exclude_features, eda_results, summary)
        
        return keep_features, eda_results, summary



if __name__ == "__main__":
    

    analyzer = FeatureEDA(spark)
    

    keep_features, eda_results, summary = analyzer.run_pipeline(
        data_folder="data"
    )
    
    print("\n" + "="*80)
    print("🎉 Pipeline Completed!")
    print("="*80)
    print(f"\n📁 Generated Files:")
    print(f"  1. selected_features.txt - List of features to use")
    print(f"  2. excluded_features.txt - List of excluded features")
    print(f"  3. eda_report.json - Detailed EDA for each feature")
    print(f"  4. eda_summary.json - Summary statistics")









    
##Bronze to Sliver

#  features_attributes 
features_attributes_clean = features_attributes.drop(columns=['Name', 'SSN'], errors='ignore')

#  features_financials 
features_financials_clean = features_financials.drop(
    columns=['Num_of_Delayed_Payment', 'Delay_from_due_date', 'Payment_of_Min_Amount'], errors='ignore'
)

# feature_clickstream 
feature_clickstream_clean = feature_clickstream.copy()

# === Clean numeric-like strings before converting ===
numeric_cols = [
    "Monthly_Balance",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Age"  # from attributes
]

for col in numeric_cols:
    if col in features_financials_clean.columns:
        # remove underscores, commas, and any non-numeric characters except . and -
        features_financials_clean[col] = (
            features_financials_clean[col]
            .astype(str)
            .str.replace(r"[^0-9\.\-]", "", regex=True)
        )
        features_financials_clean[col] = pd.to_numeric(features_financials_clean[col], errors='coerce')

    elif col in features_attributes_clean.columns:
        features_attributes_clean[col] = (
            features_attributes_clean[col]
            .astype(str)
            .str.replace(r"[^0-9\.\-]", "", regex=True)
        )
        features_attributes_clean[col] = pd.to_numeric(features_attributes_clean[col], errors='coerce')

# === 1. Clean numeric columns ===

# Monthly_Balance: replace extreme negative values with median
median_balance = features_financials_clean['Monthly_Balance'].median()
features_financials_clean.loc[features_financials_clean['Monthly_Balance'] < -1e10, 'Monthly_Balance'] = median_balance

# Age: replace invalid values (<0 or >120) with median
median_age = features_attributes_clean['Age'][(features_attributes_clean['Age'] >= 0) & (features_attributes_clean['Age'] <= 120)].median()
features_attributes_clean.loc[(features_attributes_clean['Age'] < 0) | (features_attributes_clean['Age'] > 120), 'Age'] = median_age

# Annual_Income: clip using IQR
q1 = features_financials_clean['Annual_Income'].quantile(0.25)
q3 = features_financials_clean['Annual_Income'].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
features_financials_clean['Annual_Income'] = features_financials_clean['Annual_Income'].clip(lower, upper)

# Num_Bank_Accounts: fix negative, clip extreme
median_accounts = features_financials_clean['Num_Bank_Accounts'][features_financials_clean['Num_Bank_Accounts'] >= 0].median()
features_financials_clean.loc[features_financials_clean['Num_Bank_Accounts'] < 0, 'Num_Bank_Accounts'] = median_accounts

q1 = features_financials_clean['Num_Bank_Accounts'].quantile(0.25)
q3 = features_financials_clean['Num_Bank_Accounts'].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
features_financials_clean['Num_Bank_Accounts'] = features_financials_clean['Num_Bank_Accounts'].clip(lower, upper)

# === 2. Clean categorical columns ===

# Credit_Mix: replace "_" with "Unknown"
features_financials_clean['Credit_Mix'] = features_financials_clean['Credit_Mix'].replace("_", "Unknown")

# Payment_Behaviour: replace "!@9#%8" with "Unknown"
features_financials_clean['Payment_Behaviour'] = features_financials_clean['Payment_Behaviour'].replace("!@9#%8", "Unknown")

# Type_of_Loan: fill NaN with "Unknown"
features_financials_clean['Type_of_Loan'] = features_financials_clean['Type_of_Loan'].fillna("Unknown")

# Fill NaN with median after cleaning
features_financials_clean['Annual_Income'].fillna(
    features_financials_clean['Annual_Income'].median(), inplace=True
)






##Merge dataset

# Step 1: Attributes + Financials
df_merged = pd.merge(
    features_attributes_clean,
    features_financials_clean,
    on='Customer_ID',
    how='left'  
)

# Step 2:  Clickstream
df_merged = pd.merge(
    df_merged,
    feature_clickstream_clean,
    on='Customer_ID',  
    how='left'
)


print(df_merged.shape)
print(df_merged.head())







#store silver dataset
silver_dataset = df_merged.copy()

silver_dataset.to_csv("datamart/silver/feature_store/silver_dataset.csv", index=False)



##Silver to Gold

from sklearn.preprocessing import LabelEncoder


categorical_cols = ['Occupation', 'Credit_Mix', 'Payment_Behaviour', 'Type_of_Loan']

#  LabelEncoder
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()

    df_merged[col] = df_merged[col].fillna('Missing')

    df_merged[col] = le.fit_transform(df_merged[col])

    label_encoders[col] = le

print(df_merged[categorical_cols].head())



##Store Gold dataset
gold_dataset = df_merged.copy()

gold_dataset.to_csv("datamart/gold/feature_store/gold_dataset.csv", index=False)
