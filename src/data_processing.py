import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def optimize_memory(df):
    """
    Optimizes memory usage by converting float64 to float32 and int64 to int32.
    This fulfills the specific project requirement for memory management.
    """
    start_mem = df.memory_usage().sum() / 1024
    print(f"Memory before optimization: {start_mem:.2f} KB")

    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
            
    end_mem = df.memory_usage().sum() / 1024
    print(f"Memory after optimization: {end_mem:.2f} KB")
    return df

def preprocess(df):
    """
    Separates features and target, splits the data, and applies SMOTE
    safely ONLY to the training data to prevent data leakage.
    """
    X = df.drop('DEATH_EVENT', axis=1)
    y = df['DEATH_EVENT']
    
    # CRITICAL FIX: Split the data BEFORE applying SMOTE
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply SMOTE only to the training set to balance the classes
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    return X_train_resampled, X_test, y_train_resampled, y_test