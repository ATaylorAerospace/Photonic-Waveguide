"""Train XGBoost regression models on the waveguide dataset."""
import os
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def train_loss_model(dataset_path: str = "data/SiN_Photonic_Waveguide_Loss_Efficiency.csv"):
    """Train XGBoost model for propagation loss prediction."""
    df = pd.read_csv(dataset_path)
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    target = "propagation_loss_dB_cm"
    if target not in df.columns:
        print(f"Target column '{target}' not found. Available: {list(df.columns)}")
        return
    X = df.drop(columns=[target]).select_dtypes(include="number")
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    print(f"XGBoost R² score: {model.score(X_test, y_test):.4f}")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgboost_loss_model.joblib")
    print("Model saved to models/xgboost_loss_model.joblib")


if __name__ == "__main__":
    train_loss_model()
