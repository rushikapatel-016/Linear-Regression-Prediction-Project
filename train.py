
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Project folder paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "clothing_demand_dataset.csv"
MODEL_PATH = BASE_DIR / "clothing_demand_model.pkl"

# Load dataset
data = pd.read_csv(DATA_PATH)

# Features (X)
X = data[
    [
        "price",
        "discount_percent",
        "previous_sales",
        "advertising_spend",
        "season_demand_index"
    ]
]

# Target (y)
y = data["units_sold"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test results
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

# Save the trained model
joblib.dump(model, MODEL_PATH)

print("Clothing Demand Predictor")
print("-------------------------")
print("Model trained successfully!")
print("Total records:", len(data))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 3))
print("Model saved as clothing_demand_model.pkl")
