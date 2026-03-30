import pandas as pd
import numpy as np
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
 
    stratify_bins = pd.qcut(df[target_col], q=n_quantiles, labels=False, duplicates='drop')
    
    # Perform the split
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
    
    # Extract the metadata
    metadata = df[meta_cols].copy()
    
    # Extract the features (CpG sites) by dropping the target and metadata columns
    cols_to_drop = [target_col] + meta_cols
    X = df.drop(columns=cols_to_drop)
    
    return X, y, metadata

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

def bootstrap_evaluation(y_true, y_pred, n_bootstraps=1000, seed=42):
    """
    Evaluates predictions using bootstrap resampling to generate 95% Confidence Intervals.
    """
    # Set the seed
    np.random.seed(seed)
    
    # Convert inputs to numpy arrays for easier indexing
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    n_samples = len(y_true_np)
    
    # Lists to store the metrics for each bootstrap iteration
    rmses, maes, r2s, pearsons = [], [], [], []
    
    for _ in range(n_bootstraps):
        # Resample indices with replacement
        indices = np.random.choice(np.arange(n_samples), size=n_samples, replace=True)
        
        # Extract the resampled pairs of actual and predicted ages
        y_true_boot = y_true_np[indices]
        y_pred_boot = y_pred_np[indices]
        
        # Calculate and store metrics for this specific bootstrap sample
        rmses.append(np.sqrt(mean_squared_error(y_true_boot, y_pred_boot)))
        maes.append(mean_absolute_error(y_true_boot, y_pred_boot))
        r2s.append(r2_score(y_true_boot, y_pred_boot))
        
        r, _ = pearsonr(y_true_boot, y_pred_boot)
        pearsons.append(r)
        
    # Calculate the Mean and 95% Confidence Intervals (2.5th and 97.5th percentiles)
    results = {}
    metric_names = ['RMSE', 'MAE', 'R²', 'Pearson r']
    metric_lists = [rmses, maes, r2s, pearsons]
    
    for name, values in zip(metric_names, metric_lists):
        mean_val = np.mean(values)
        lower_ci = np.percentile(values, 2.5)
        upper_ci = np.percentile(values, 97.5)
        results[name] = (mean_val, lower_ci, upper_ci)
        
    return results

# This function is an enhanced version of the previous bootstrap_evaluation, which now also returns the standard deviation of the metrics across the bootstrap samples.
def bootstrap_evaluation_final(y_true, y_pred, n_bootstraps=1000, seed=42):
    np.random.seed(seed)
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    n_samples = len(y_true_np)
    
    rmses, maes, r2s, pearsons = [], [], [], []
    
    for _ in range(n_bootstraps):
        indices = np.random.choice(np.arange(n_samples), size=n_samples, replace=True)
        y_true_boot, y_pred_boot = y_true_np[indices], y_pred_np[indices]
        
        rmses.append(np.sqrt(mean_squared_error(y_true_boot, y_pred_boot)))
        maes.append(mean_absolute_error(y_true_boot, y_pred_boot))
        r2s.append(r2_score(y_true_boot, y_pred_boot))
        
        r, _ = pearsonr(y_true_boot, y_pred_boot)
        pearsons.append(r)
        
    results = {}
    metric_names = ['RMSE', 'MAE', 'R²', 'Pearson r']
    metric_lists = [rmses, maes, r2s, pearsons]
    
    for name, values in zip(metric_names, metric_lists):
        results[name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'lower_ci': np.percentile(values, 2.5),
            'upper_ci': np.percentile(values, 97.5)
        }
        
    return results

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from collections import Counter

def stability_selection(X_scaled, y, feature_names, n_iterations=50, subsample_fraction=0.8, top_k=200, threshold=0.5, seed=42):
    """
    Implements stability selection using Spearman correlation to find robust features.
    """
    np.random.seed(seed)
    n_samples, n_features = X_scaled.shape
    subsample_size = int(n_samples * subsample_fraction)
    
    # Counter to track how many times each feature index is selected in the top 200
    selection_counts = Counter()
    y_array = np.array(y)
    
    print(f"Running Stability Selection ({n_iterations} iterations)...")
    
    for i in range(n_iterations):
        # Draw subsample (80% of data, WITHOUT replacement)
        indices = np.random.choice(n_samples, size=subsample_size, replace=False)
        X_sub = X_scaled[indices, :]
        y_sub = y_array[indices]
        
        # Calculate absolute Spearman correlation for each feature with age
        # We use absolute correlation because strongly negative correlations 
        # (hypomethylation with age) are just as biologically important as positive ones.
        correlations = np.zeros(n_features)
        for j in range(n_features):
            corr, _ = spearmanr(X_sub[:, j], y_sub)
            correlations[j] = np.abs(corr)
            
        # Rank all features and keep the indices of the top 200
        # np.argsort sorts ascending, so we take the last `top_k` elements
        top_indices = np.argsort(correlations)[-top_k:]
        
        # Update our counts with the features selected in this iteration
        selection_counts.update(top_indices)
        
    # Identify stable features (selected in > 50% of iterations)
    threshold_count = n_iterations * threshold
    stable_indices = [idx for idx, count in selection_counts.items() if count > threshold_count]
    
    # Map indices back to actual CpG feature names
    stable_feature_names = [feature_names[idx] for idx in stable_indices]
    
    return stable_feature_names, selection_counts