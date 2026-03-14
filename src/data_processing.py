import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def load_data(filepath):
    df = pd.read_excel(filepath)
    return df

def optimize_memory(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

def preprocess(df):
    X = df.drop('DEATH_EVENT', axis=1)
    y = df['DEATH_EVENT']
    
    # Corriger le déséquilibre avec SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Séparer train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test