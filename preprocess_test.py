import polars as pl

print("1. Initializing Lazy Pipeline for Test Data...")

categorical_cols = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']

print("2. Scanning test_data.csv...")
query = pl.scan_csv("test_data.csv", null_values=[''])

# Get column names the fast way to clear the warning
actual_columns = query.collect_schema().names()

print("3. Cleaning and Downcasting...")
query = (
    query
    # FIX: Cast to String first, then Categorical
    .with_columns([
        pl.col(c).cast(pl.String).cast(pl.Categorical) for c in categorical_cols if c in actual_columns
    ])
    .with_columns([
        pl.col(pl.Float64).cast(pl.Float32)
    ])
)

print("4. Streaming to Parquet... (This will take a moment for the 33 GB file)")
query.sink_parquet("test_data_cleaned.parquet")

print("✅ Success! Test data converted to Parquet.")