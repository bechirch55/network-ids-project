import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from src.preprocess import load_data, preprocess_data_multiclass, split_data
import numpy as np

os.makedirs("models", exist_ok=True)

df = load_data("data/train.csv")
X, y = preprocess_data_multiclass(df)
X_train, X_val, y_train, y_val = split_data(X, y)
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
model = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dense(32, activation='relu'),
    Dense(output_dim, activation='softmax')
])

model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
model.save("models/ids_model.h5")
print("Training finished, model saved!")
