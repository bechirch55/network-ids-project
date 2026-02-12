import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data_multiclass(df, label_column=None):
    if not label_column:
        str_cols = df.select_dtypes(include='object').columns
        label_column = str_cols[-1] if len(str_cols) > 0 else None

    if label_column not in df.columns:
        raise ValueError("No label column found! Please specify label_column.")

    df[label_column] = df[label_column].str.strip().str.lower()
    X = df.drop(columns=[label_column])
    y = df[label_column]

    cat_cols = X.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols)

    ohe = OneHotEncoder(sparse_output=False)
    y_encoded = ohe.fit_transform(y.values.reshape(-1,1))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save metadata for future predictions
    np.save("models/feature_columns.npy", X.columns.to_numpy())
    joblib.dump(scaler, "models/scaler.pkl")

    return X_scaled, y_encoded

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def preprocess_data_for_prediction(df):
    # 1. Load feature columns
    feature_columns = np.load("models/feature_columns.npy", allow_pickle=True)

    # 2. Load the actual saved scaler
    scaler = joblib.load("models/scaler.pkl")

    # 3. Handle categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols)

    # 4. Align columns with training data
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    df = df[feature_columns]

    # 5. Scale using the TRAINED parameters (transform, not fit)
    X_scaled = scaler.transform(df)
    return X_scaled