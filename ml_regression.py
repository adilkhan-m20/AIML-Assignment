import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import Bunch


# ----------------------------
# Load dataset
# ----------------------------
def load_dataset():
    """
    Try to load the California Housing dataset.
    If it fails (no internet), fallback to the Diabetes dataset.
    Returns a sklearn Bunch object with data, target, and metadata.
    """
    try:
        from sklearn.datasets import fetch_california_housing
        ds = fetch_california_housing(as_frame=True)
        name = "California Housing"
        X = ds.frame.drop(columns=[ds.target_names[0]])
        y = ds.frame[ds.target_names[0]]
        feature_names = list(X.columns)
        return Bunch(data=X, target=y, feature_names=feature_names, name=name)
    except Exception:
        from sklearn.datasets import load_diabetes
        ds = load_diabetes(as_frame=True)
        name = "Diabetes (fallback)"
        X = ds.frame.drop(columns=["target"])
        y = ds.frame["target"]
        feature_names = list(X.columns)
        return Bunch(data=X, target=y, feature_names=feature_names, name=name)


# ----------------------------
# Quick Exploratory Data Analysis
# ----------------------------
def quick_eda(X, y, name):
    """Prints dataset summary and saves a histogram of the target variable."""
    print(f"Dataset: {name} | Samples = {len(X)} | Features = {len(X.columns)}")
    print("Head:\n", X.head(3))
    print("Target stats: mean = {:.3f}, std = {:.3f}".format(y.mean(), y.std()))

    # Plot target distribution
    plt.figure()
    y.hist(bins=30)
    plt.title(f"{name} - Target Distribution")
    plt.xlabel("Target Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("target_hist.png")
    plt.close()
    print("Saved: target_hist.png")


# ----------------------------
# Build, Train, and Evaluate Linear Regression Model
# ----------------------------
def build_and_eval(X, y, feature_names):
    """Builds a preprocessing + Linear Regression pipeline and evaluates it."""
    # Define preprocessing
    num_features = list(feature_names)
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), num_features)],
        remainder="drop"
    )

    # Define pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)

    # Evaluate metrics
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

    # Plot True vs Predicted
    plt.figure()
    plt.scatter(y_test, preds, s=10)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Linear Regression: True vs Predicted")
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    plt.plot(lims, lims, 'r--')
    plt.tight_layout()
    plt.savefig("true_vs_pred.png")
    plt.close()
    print("Saved: true_vs_pred.png")

    # ----------------------------
    # Save Model Card (metadata)
    # ----------------------------
    card = {
        "model": "LinearRegression",
        "dataset": "California Housing (fallback: Diabetes)",
        "task": "Tabular Regression",
        "preprocessing": "StandardScaler on all numeric features",
        "target": "MedianHouseValue (or Diabetes target)",
        "metrics": {"RMSE": rmse, "MAE": mae, "R2": r2},
        "intended_use": "Introductory ML coursework; not for real-estate decisions",
        "limitations": [
            "Linear model — cannot capture feature interactions or nonlinearity.",
            "No feature engineering; sensitive to outliers and collinearity."
        ],
        "owner": "Student",
    }

    with open("model_card.json", "w") as f:
        json.dump(card, f, indent=2)

    print("Saved: model_card.json")
    return rmse, mae, r2


# ----------------------------
# Main Script
# ----------------------------
if __name__ == "__main__":
    ds = load_dataset()
    quick_eda(ds.data, ds.target, ds.name)
    build_and_eval(ds.data, ds.target, ds.feature_names)
