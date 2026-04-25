import cudf
import cupy as cp
import gc
import warnings
from sklearn.experimental import enable_iterative_imputer  # Must be imported to enable IterativeImputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

def treatment_missing_values_gpu(df):
    print("[INIT] Starting Advanced Missing Value Treatment...")

    # ---------------------------------------------------------
    # 1. CONDITIONAL MISSINGNESS (Feature-linked gaps)
    # ---------------------------------------------------------
    print("[STEP 1] Applying conditional treatment for feature-linked gaps...")
    # Example logic: If a user has no active credit cards (Active_Cards == 0), 
    # their 'Credit_Limit' might be NaN. This isn't random; it's conditional.
    # We fill these specific cases with 0 or a designated category.
    
    # (Note: These are conceptual AMEX column examples. Nishant will need to map the real ones)
    if 'Active_Cards' in df.columns and 'Credit_Limit' in df.columns:
        # Convert to pandas temporarily if complex masking fails in cuDF, 
        # but basic masking works in cuDF:
        mask = (df['Active_Cards'] == 0) & (df['Credit_Limit'].isnull())
        df.loc[mask, 'Credit_Limit'] = 0

    # ---------------------------------------------------------
    # 2. MICE-BASED LOGIC (For Random Missingness)
    # ---------------------------------------------------------
    print("[STEP 2] Applying MICE for random missingness...")
    
    # We must separate numeric columns for MICE.
    # Categoricals should be encoded before this step, or imputed using mode.
    numeric_cols = [col for col in df.columns if df[col].dtype in ['float32', 'float64', 'int32', 'int64']]
    
    # Convert cuDF to Pandas/NumPy temporarily for sklearn compatibility, 
    # BUT we use a GPU-accelerated XGBoost estimator to do the heavy lifting.
    df_numeric = df[numeric_cols].to_pandas()

    # The Secret Weapon: XGBoost on GPU acting as the MICE estimator.
    # Standard MICE uses Linear Regression, which is slow. XGBoost on GPU is lightning fast.
    gpu_estimator = XGBRegressor(
        tree_method='hist', 
        device='cuda',      # Forces XGBoost to use the GPU
        n_estimators=50, 
        random_state=42
    )

    mice_imputer = IterativeImputer(
        estimator=gpu_estimator,
        max_iter=5,         # 5 iterations is usually sufficient for stability
        tol=1e-3,           # Tolerance for stability checks
        random_state=42,
        verbose=1           # Prints progress
    )

    # Fit and transform the data
    imputed_array = mice_imputer.fit_transform(df_numeric)
    
    # Convert back to GPU cuDF dataframe
    df_imputed_numeric = cudf.DataFrame(imputed_array, columns=numeric_cols)
    
    # Overwrite original numeric columns with imputed columns
    for col in numeric_cols:
        df[col] = df_imputed_numeric[col]

    # ---------------------------------------------------------
    # 3. STABILITY CHECKS
    # ---------------------------------------------------------
    print("[STEP 3] Running imputation stability checks...")
    # Check if any NaNs slipped through the MICE process
    remaining_nans = df.isnull().sum().sum()
    if remaining_nans > 0:
        print(f"[WARNING] {remaining_nans} missing values remain. Applying fallback median fill.")
        df = df.fillna(df.median())
    else:
        print("[SUCCESS] All missing values resolved with stability.")

    return df

# --- ASSUMING YOUR FUNCTIONS ARE DEFINED HERE ---
# def process_data_gpu(df, is_amex=True): ...
# def treatment_missing_values_gpu(df): ...

if __name__ == "__main__":
    print("=== EVOASTRA MASTER DATA PIPELINE ===")
    
    # 1. Define file paths
    amex_path = 'data/train_data.parquet'
    gmsc_path = 'data/cs-training.csv'
    
    # 2. Load the Raw Data directly to the GPU
    print("\n--- LOADING DATA ---")
    try:
        amex_raw = cudf.read_parquet(amex_path)
        print(f"AMEX loaded successfully. Shape: {amex_raw.shape}")
    except Exception as e:
        print(f"Error loading AMEX data: {e}")
        
    # 3. Execute Phase 1: Aggregation, Categorical Encoding, and Outliers
    # (Assuming you updated process_data_gpu to skip its basic imputation step)
    print("\n--- RUNNING BASE PREPARATION ---")
    amex_base = process_data_gpu(amex_raw, is_amex=True)
    
    # Free up GPU memory from the raw dataset immediately
    del amex_raw
    gc.collect()
    
    # 4. Execute Phase 2: Advanced MICE Imputation
    print("\n--- RUNNING MICE IMPUTATION ---")
    amex_final = treatment_missing_values_gpu(amex_base)
    
    print("\n[SUCCESS] Pipeline execution complete. Ready for ML Modeling.")
    print(f"Final AMEX Shape: {amex_final.shape}")
    
    # Optional: Save the pristine data back to disk so ML teams don't have to rerun this
    # amex_final.to_parquet('data/train_data_CLEANED.parquet')