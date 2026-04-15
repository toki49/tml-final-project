from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features.csv"
MODEL_PATH = BASE_DIR / "models" / "rf_fold1.pkl"
TARGET = "downgrade_flag"
DROP_COLUMNS = ["Company name", "Region", "Country", "NACE code"]
TRAIN_YEARS = [2015, 2016, 2017]
DEMO_YEAR = 2018


def ensure_artifact(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found at {path}. Run the training pipeline first to create it."
        )


def load_reference_features() -> pd.DataFrame:
    ensure_artifact(FEATURES_PATH, "Feature dataset")
    return pd.read_csv(FEATURES_PATH)


def get_model_feature_names() -> list[str]:
    df = load_reference_features().drop(columns=DROP_COLUMNS)
    train_df = df.loc[df["year"].isin(TRAIN_YEARS)].copy()
    X_train = train_df.drop(columns=[TARGET, "year"])
    finite_mask = np.isfinite(X_train).all(axis=1)
    X_train = X_train.loc[finite_mask].reset_index(drop=True)
    return X_train.columns.tolist()


def fit_reference_scaler() -> tuple[StandardScaler, list[str]]:
    feature_names = get_model_feature_names()
    df = load_reference_features().drop(columns=DROP_COLUMNS)
    train_df = df.loc[df["year"].isin(TRAIN_YEARS)].copy()
    X_train = train_df[feature_names]
    finite_mask = np.isfinite(X_train).all(axis=1)
    X_train = X_train.loc[finite_mask].reset_index(drop=True)

    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler, feature_names


def load_reference_model():
    ensure_artifact(MODEL_PATH, "Random Forest model")
    model = joblib.load(MODEL_PATH)
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    return model


def read_uploaded_table(name: str, data: bytes) -> pd.DataFrame:
    suffix = Path(name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(BytesIO(data))
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(BytesIO(data))
    raise ValueError("Unsupported file type. Upload a CSV or Excel file.")


def load_demo_input(sample_size: int = 200) -> pd.DataFrame:
    df = load_reference_features()
    demo = df.loc[df["year"] == DEMO_YEAR].copy()
    return demo.head(sample_size).reset_index(drop=True)


def template_input(sample_size: int = 20) -> pd.DataFrame:
    return load_demo_input(sample_size=sample_size)


def score_feature_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if df.empty:
        raise ValueError("The uploaded table is empty.")

    scaler, feature_names = fit_reference_scaler()
    model = load_reference_model()

    missing_columns = [col for col in feature_names if col not in df.columns]
    if missing_columns:
        preview = ", ".join(missing_columns[:10])
        raise ValueError(
            "The uploaded file is missing required model features. "
            f"First missing columns: {preview}"
        )

    metadata = df.copy()
    X = df[feature_names].copy()
    finite_mask = np.isfinite(X).all(axis=1)

    scored_rows = metadata.loc[finite_mask].reset_index(drop=True)
    X_valid = X.loc[finite_mask].reset_index(drop=True)
    X_scaled = scaler.transform(X_valid)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    scored_rows["downgrade_probability"] = probabilities
    scored_rows["predicted_downgrade_0_5"] = (probabilities >= 0.5).astype(int)

    summary = {
        "input_rows": int(len(df)),
        "scored_rows": int(len(scored_rows)),
        "dropped_non_finite_rows": int((~finite_mask).sum()),
        "model_feature_count": int(len(feature_names)),
    }
    return scored_rows, summary
