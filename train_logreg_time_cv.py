from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features.csv"
FIGURE_PATH = BASE_DIR / "figures" / "roc_logreg.png"
TARGET = "downgrade_flag"
DROP_BEFORE_MODELING = ["Company name", "Region", "Country", "NACE code"]
FOLDS = [
    {"label": "Fold 1", "train_years": [2015, 2016, 2017], "val_year": 2018},
    {"label": "Fold 2", "train_years": [2015, 2016, 2017, 2018], "val_year": 2019},
]


def main() -> None:
    df = pd.read_csv(FEATURES_PATH)
    df = df.drop(columns=DROP_BEFORE_MODELING)

    plt.figure(figsize=(8, 6))
    aucs = []

    for fold in FOLDS:
        train_mask = df["year"].isin(fold["train_years"])
        val_mask = df["year"] == fold["val_year"]

        train_df = df.loc[train_mask].copy()
        val_df = df.loc[val_mask].copy()

        y_train = train_df[TARGET]
        y_val = val_df[TARGET]

        X_train = train_df.drop(columns=[TARGET, "year"])
        X_val = val_df.drop(columns=[TARGET, "year"])

        train_finite_mask = np.isfinite(X_train).all(axis=1)
        val_finite_mask = np.isfinite(X_val).all(axis=1)
        X_train = X_train.loc[train_finite_mask]
        y_train = y_train.loc[train_finite_mask]
        X_val = X_val.loc[val_finite_mask]
        y_val = y_val.loc[val_finite_mask]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
        )
        model.fit(X_train_scaled, y_train)

        val_probs = model.predict_proba(X_val_scaled)[:, 1]
        fold_auc = roc_auc_score(y_val, val_probs)
        aucs.append(fold_auc)

        fpr, tpr, _ = roc_curve(y_val, val_probs)
        plt.plot(fpr, tpr, label=f"{fold['label']} (AUROC = {fold_auc:.4f})")

        print(
            f"{fold['label']}: train_years={fold['train_years']}, "
            f"val_year={fold['val_year']}, AUROC={fold_auc:.4f}"
        )

    mean_auc = sum(aucs) / len(aucs)
    print(f"Mean AUROC: {mean_auc:.4f}")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Logistic Regression ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_PATH, dpi=150)
    plt.close()

    print(f"Saved ROC figure to: {FIGURE_PATH.resolve()}")


if __name__ == "__main__":
    main()
