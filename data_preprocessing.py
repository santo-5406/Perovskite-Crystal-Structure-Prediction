import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
import os

def impute_missing_values(df):
    """
    Imputes missing values in numerical columns with median and categorical with mode.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df

def engineer_domain_features(df):
    """
    Calculates domain-specific features for perovskite structures:
    1. Goldschmidt tolerance factor: t = (r_A + r_X) / (sqrt(2) * (r_B + r_X))
    2. Ionic radius ratios: r_A / r_B, r_B / r_X
    3. Electronegativity differences: |EN_A - EN_X|, |EN_B - EN_X|, |EN_A - EN_B|
    """
    df = df.copy()
    
    # Required columns for physics calculation
    req_radius = ['r_A', 'r_B', 'r_X']
    req_en = ['electronegativity_A', 'electronegativity_B', 'electronegativity_X']
    
    # Check if we have the physical columns
    has_radius = all(col in df.columns for col in req_radius)
    has_en = all(col in df.columns for col in req_en)
    
    if has_radius:
        # Goldschmidt tolerance factor
        df['goldschmidt_tolerance_factor'] = (df['r_A'] + df['r_X']) / (np.sqrt(2) * (df['r_B'] + df['r_X']))
        # Ionic radius ratios
        df['radius_ratio_AB'] = df['r_A'] / df['r_B']
        df['radius_ratio_BX'] = df['r_B'] / df['r_X']
    else:
        print("Warning: Missing required radius columns (r_A, r_B, r_X). Goldschmidt factor and radius ratios skipped.")
        
    if has_en:
        # Electronegativity differences
        df['en_diff_AX'] = (df['electronegativity_A'] - df['electronegativity_X']).abs()
        df['en_diff_BX'] = (df['electronegativity_B'] - df['electronegativity_X']).abs()
        df['en_diff_AB'] = (df['electronegativity_A'] - df['electronegativity_B']).abs()
    else:
        print("Warning: Missing electronegativity columns. Electronegativity differences skipped.")
        
    return df

def handle_imbalance_ros(X_train, y_train):
    """
    Balances the training set using Random Over-Sampling (ROS).
    """
    # Combine back to access target
    df_train = pd.DataFrame(X_train)
    df_train['target'] = y_train
    
    # Find size of majority class
    class_counts = df_train['target'].value_counts()
    max_size = class_counts.max()
    
    resampled_classes = []
    for cls, group in df_train.groupby('target'):
        if len(group) < max_size:
            oversampled = resample(group, replace=True, n_samples=max_size, random_state=42)
            resampled_classes.append(oversampled)
        else:
            resampled_classes.append(group)
            
    df_resampled = pd.concat(resampled_classes)
    
    # Shuffle
    df_resampled = df_resampled.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df_resampled.drop(columns=['target']), df_resampled['target']

def prepare_pipeline(csv_path, target_col, is_classification=True, handle_imbalance=True):
    """
    Complete end-to-end preprocessing pipeline:
    1. Load data
    2. Impute missing values
    3. Engineer domain features
    4. Encode input categorical features
    5. Encode target label (if classification)
    6. Train-test split (80/20)
    7. Handle class imbalance on training set (if classification & enabled)
    8. Scale numeric features
    
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder, feature_names
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # 1. Impute missing values
    df = impute_missing_values(df)
    
    # 2. Domain-specific features
    df = engineer_domain_features(df)
    
    # Separate features and target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset. Available: {list(df.columns)}")
        
    y = df[target_col]
    
    # Drop target and unnecessary columns like formula
    drop_cols = [target_col]
    if 'formula' in df.columns:
        drop_cols.append('formula')
    X = df.drop(columns=drop_cols)
    
    # 3. Categorical encoding of inputs
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        # Convert boolean dummies to floats/ints
        bool_cols = X.select_dtypes(include=['bool']).columns
        X[bool_cols] = X[bool_cols].astype(float)
        
    # Ensure all inputs are numeric now
    X = X.astype(float)
    feature_names = X.columns.tolist()
    
    # 4. Target encoding (if classification)
    label_encoder = None
    if is_classification:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    else:
        # For regression, check if target is numeric
        y = pd.to_numeric(y, errors='coerce')
        # Drop rows where target is NaN (if any failed conversion)
        non_nan_mask = ~np.isnan(y)
        X = X[non_nan_mask]
        y = y[non_nan_mask]
        y = y.values
        
    # 5. Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.20, random_state=42, stratify=y if is_classification else None
    )
    
    # 6. Handle class imbalance (on training data only)
    if is_classification and handle_imbalance:
        X_train, y_train = handle_imbalance_ros(X_train, y_train)
        # Convert back to numpy array if returned as df
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
            
    # 7. Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder, feature_names

if __name__ == '__main__':
    # Test preprocessor
    print("Testing data preprocessing...")
    try:
        X_train, X_test, y_train, y_test, scaler, le, fnames = prepare_pipeline(
            "perovskite_dataset.csv", "crystal_structure", is_classification=True
        )
        print("Classification preprocessing successful!")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"Number of engineered features: {len(fnames)}")
        print(f"Features: {fnames}")
        
        # Test regression preprocessing
        X_train_r, X_test_r, y_train_r, y_test_r, scaler_r, le_r, fnames_r = prepare_pipeline(
            "perovskite_dataset.csv", "lattice_parameter", is_classification=False
        )
        print("\nRegression preprocessing successful!")
        print(f"X_train_r shape: {X_train_r.shape}, y_train_r shape: {y_train_r.shape}")
    except Exception as e:
        print(f"Preprocessing test failed: {e}")
