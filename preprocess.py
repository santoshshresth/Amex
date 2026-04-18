import polars as pl

print("1. Initializing Lazy Pipeline...")

# The AMEX categorical columns
categorical_cols = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']

print("2. Scanning CSVs...")
train_query = pl.scan_csv("train_data.csv", null_values=[''])
labels_query = pl.scan_csv("train_labels.csv")

print("3. Merging and Cleaning...")
# Join the training data with the labels based on 'customer_ID'
query = train_query.join(labels_query, on="customer_ID", how="left")

# FIX 1: Get column names the fast, 'Lazy' way to clear the Performance Warning
actual_columns = query.collect_schema().names()

# Apply our memory-saving downcasts
query = (
    query
    # FIX 2: Cast to String first, THEN to Categorical to clear the InvalidOperationError
    .with_columns([
        pl.col(c).cast(pl.String).cast(pl.Categorical) for c in categorical_cols if c in actual_columns
    ])
    .with_columns([
        pl.col(pl.Float64).cast(pl.Float32)
    ])
)

print("4. Executing Pipeline and Streaming to Parquet...")
print("   (Your CPU will work hard here as it streams the 50 GB file in chunks...)")

# Execute the join, clean it, and save it straight to Parquet
query.sink_parquet("train_merged_cleaned.parquet")

print("✅ Success! Your data is merged, cleaned, and saved as a Parquet file.")