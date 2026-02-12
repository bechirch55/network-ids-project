import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
from src import preprocess

MODEL_PATH = "models/ids_model.h5"
FEATURES_PATH = "models/feature_columns.npy"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
model = load_model(MODEL_PATH)

if os.path.exists(LABEL_ENCODER_PATH):
    encoder = joblib.load(LABEL_ENCODER_PATH)
    if hasattr(encoder, 'classes_'):
        label_classes = encoder.classes_
    else:
        label_classes = encoder
else:
    
    print(f"Warning: {LABEL_ENCODER_PATH} not found. Using default labels.")
    label_classes = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R'] 



def predict_labels(filepath):
    """
    Reads a CSV, predicts labels, and returns a LIST of label strings.
    Called by app.py.
    """
    
    df = pd.read_csv(filepath) 
    
    X = preprocess.preprocess_data_for_prediction(df.copy())
    y_pred_probs = model.predict(X)
    y_pred = np.argmax(y_pred_probs, axis=1)
    predicted_labels = [label_classes[i] for i in y_pred]
    return predicted_labels