from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features.csv"
MODEL_PATH = BASE_DIR / "models" / "rf_fold1.pkl"
FIGURE_PATH = BASE_DIR / "figures" / "shap_summary.png"
TARGET = "downgrade_flag"
DROP_COLUMNS = ["Company name", "Region", "Country", "NACE code"]
VAL_YEAR = 2018


def load_validation_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(FEATURES_PATH)
    df = df.loc[df["year"] == VAL_YEAR].copy()
    df = df.drop(columns=DROP_COLUMNS)

    X_val = df.drop(columns=[TARGET, "year"])
    y_val = df[TARGET]

    finite_mask = np.isfinite(X_val).all(axis=1)
    X_val = X_val.loc[finite_mask].reset_index(drop=True)
    y_val = y_val.loc[finite_mask].reset_index(drop=True)
    return X_val, y_val


def main() -> None:
    model = joblib.load(MODEL_PATH)
    model.n_jobs = 1
    X_val, _ = load_validation_data()

    if len(model.feature_importances_) != X_val.shape[1]:
        raise ValueError(
            f"Feature count mismatch: model expects {len(model.feature_importances_)} "
            f"features but validation data has {X_val.shape[1]} columns."
        )

    importance = (
        pd.Series(model.feature_importances_, index=X_val.columns)
        .sort_values(ascending=False)
    )
    top_15 = importance.head(15).sort_values(ascending=True)

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    plt.barh(top_15.index, top_15.values, color="#2f6c8f")
    plt.xlabel("Feature Importance")
    plt.title("Random Forest Top 15 Feature Importances")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved feature-importance chart to: {FIGURE_PATH.resolve()}")
    print("Top 5 features by importance:")
    for feature, value in importance.head(5).items():
        print(f"{feature}: {value:.6f}")


if __name__ == "__main__":
    main()
