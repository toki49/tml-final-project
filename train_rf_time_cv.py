from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features.csv"
FIGURE_PATH = BASE_DIR / "figures" / "model_comparison.png"
MODELS_DIR = BASE_DIR / "models"
TARGET = "downgrade_flag"
DROP_BEFORE_MODELING = ["Company name", "Region", "Country", "NACE code"]
LOGREG_AUCS = {"Fold 1": 0.6688, "Fold 2": 0.6487}
FOLDS = [
    {
        "label": "Fold 1",
        "train_years": [2015, 2016, 2017],
        "val_year": 2018,
        "model_path": MODELS_DIR / "rf_fold1.pkl",
    },
    {
        "label": "Fold 2",
        "train_years": [2015, 2016, 2017, 2018],
        "val_year": 2019,
        "model_path": MODELS_DIR / "rf_fold2.pkl",
    },
]


def main() -> None:
    df = pd.read_csv(FEATURES_PATH)
    df = df.drop(columns=DROP_BEFORE_MODELING)

    rf_aucs = {}
    best_fold = None
    best_auc = float("-inf")
    best_confusion = None

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

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

        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_scaled, y_train)

        val_probs = model.predict_proba(X_val_scaled)[:, 1]
        fold_auc = roc_auc_score(y_val, val_probs)
        rf_aucs[fold["label"]] = fold_auc

        print(
            f"{fold['label']}: train_years={fold['train_years']}, "
            f"val_year={fold['val_year']}, AUROC={fold_auc:.4f}"
        )

        joblib.dump(model, fold["model_path"])

        if fold_auc > best_auc:
            best_auc = fold_auc
            best_fold = fold["label"]
            best_predictions = (val_probs >= 0.3).astype(int)
            best_confusion = confusion_matrix(y_val, best_predictions)

    mean_auc = sum(rf_aucs.values()) / len(rf_aucs)
    print(f"Mean AUROC: {mean_auc:.4f}")

    labels = [fold["label"] for fold in FOLDS]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 6))
    plt.bar(x - width / 2, [LOGREG_AUCS[label] for label in labels], width, label="LogReg")
    plt.bar(x + width / 2, [rf_aucs[label] for label in labels], width, label="RandomForest")
    plt.xticks(x, labels)
    plt.ylabel("AUROC")
    plt.ylim(0, 1)
    plt.title("Model AUROC Comparison by Fold")
    plt.legend()
    plt.tight_layout()

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_PATH, dpi=150)
    plt.close()

    print(f"Saved comparison chart to: {FIGURE_PATH.resolve()}")
    print(
        f"Saved models to: {(MODELS_DIR / 'rf_fold1.pkl').resolve()} "
        f"and {(MODELS_DIR / 'rf_fold2.pkl').resolve()}"
    )

    print(f"Best RandomForest fold: {best_fold} (AUROC={best_auc:.4f})")
    print("Confusion matrix at threshold=0.3:")
    print(best_confusion)


if __name__ == "__main__":
    main()
