import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(filepath):
    """
    Loads a CSV dataset from the specified file path.
    
    Parameters:
    filepath (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: The loaded dataset.
    """
    return pd.read_csv(filepath)

def create_stratified_split(df, target_col, test_size=0.2, random_state=42, n_quantiles=10):
    """
    Splits the dataframe into train and validation sets, stratified.
    Uses quantiles to create categorical bins for stratification.
    
    Parameters:
    df (pd.DataFrame): The dataset to split.
    target_col (str): The name of the target column to stratify on.
    test_size (float): The proportion of the dataset to include in the validation split.
    random_state (int): Controls the shuffling applied to the data before applying the split.
    n_quantiles (int): The number of quantile bins to create for stratification.
    
    Returns:
    tuple: (train_df, val_df) containing the split dataframes.
    """
    # duplicates='drop' handles cases where there might be many identical values 
    # clustering at a specific quantile boundary.
    stratify_bins = pd.qcut(df[target_col], q=n_quantiles, labels=False, duplicates='drop')
    
    # Perform the split using the bins for stratification
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=stratify_bins
    )
    
    return train_df, val_df

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def build_cpg_preprocessing_pipeline():
    """
    Builds a pipeline to handle missing values and scale features.
    """
    pipeline = Pipeline(steps=[
        # Impute missing MCAR values with the median of that feature
        ('imputer', SimpleImputer(strategy='median')),
        
        # Scale features
        ('scaler', StandardScaler())
    ])
    
    return pipeline

import pandas as pd

def separate_features_target(df, target_col='age', meta_cols=['sample_id', 'ethnicity', 'sex']):
    """
    Separates the dataframe into feature matrix X (CpG sites), 
    target vector y (Age), and a metadata dataframe.
    
    Parameters:
    df (pd.DataFrame): The input dataframe.
    target_col (str): The name of the target variable to predict.
    meta_cols (list): A list of column names that contain metadata (not predictive features).
    
    Returns:
    tuple: (X, y, metadata) where X contains only the CpG features, 
           y is the target, and metadata contains the demographic info.
    """
    # Extract the target variable
    y = df[target_col].copy()
    
    # 2. Extract the metadata
    # We keep this just in case we want to analyze errors across different sexes/ethnicities later
    metadata = df[meta_cols].copy()
    
    # 3. Extract the features (CpG sites) by dropping the target and metadata columns
    cols_to_drop = [target_col] + meta_cols
    X = df.drop(columns=cols_to_drop)
    
    return X, y, metadata