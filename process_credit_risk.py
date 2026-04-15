import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SOURCE_PATH = BASE_DIR / "data" / "raw" / "credit-risk.xlsx"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "credit_panel.csv"
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features.csv"
YEARS = list(range(2015, 2021))
METRICS = ["Turnover", "EBIT", "PLTax", "MScore", "Leverage", "ROE", "TAsset"]
ID_COLUMNS = ["Company name", "Region", "Country", "NACE code", "Sector 1"]
M_SCORE_ORDER = {
    "AAA": 1,
    "AA": 2,
    "A": 3,
    "BBB": 4,
    "BB": 5,
    "B": 6,
    "CCC": 7,
    "CC": 8,
    "C": 9,
    "D": 10,
}


def metric_year_columns(columns: list[str]) -> list[str]:
    pattern = re.compile(rf"^({'|'.join(METRICS)})\.(\d{{4}})$")
    return [col for col in columns if pattern.match(col)]


def validate_source_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            "Raw dataset not found. Expected file at: "
            f"{path}. Place credit-risk.xlsx in data/raw/ and rerun the script."
        )


def load_sheets(path: Path) -> pd.DataFrame:
    workbook = pd.ExcelFile(path)
    data_sheets = [name for name in workbook.sheet_names if re.match(r"^\d+k-\d+k$", name)]
    frames = [pd.read_excel(path, sheet_name=sheet_name) for sheet_name in data_sheets]
    return pd.concat(frames, ignore_index=True)


def reshape_to_long(wide_df: pd.DataFrame) -> pd.DataFrame:
    use_columns = ID_COLUMNS + metric_year_columns(wide_df.columns.tolist())
    df = wide_df[use_columns].copy()

    long_frames = []
    for year in YEARS:
        year_frame = df[ID_COLUMNS].copy()
        year_frame["year"] = year
        for metric in METRICS:
            year_frame[metric] = df[f"{metric}.{year}"]
        long_frames.append(year_frame)

    long_df = pd.concat(long_frames, ignore_index=True)
    return long_df.sort_values(["Company name", "year"]).reset_index(drop=True)


def add_target(long_df: pd.DataFrame) -> pd.DataFrame:
    panel = long_df.copy()
    panel["MScore"] = panel["MScore"].astype(str).str.strip().replace({"nan": pd.NA})
    panel["mscore_num"] = panel["MScore"].map(M_SCORE_ORDER)
    panel["next_MScore"] = panel.groupby("Company name")["MScore"].shift(-1)
    panel["next_mscore_num"] = panel.groupby("Company name")["mscore_num"].shift(-1)
    panel = panel[panel["next_mscore_num"].notna()].copy()
    panel["downgrade_flag"] = (
        (panel["next_mscore_num"] - panel["mscore_num"]) > 0.5
    ).astype(int)
    return panel


def build_features() -> None:
    panel = pd.read_csv(OUTPUT_PATH)
    panel = panel.sort_values(["Company name", "year"]).reset_index(drop=True)

    company_group = panel.groupby("Company name")
    panel["ebit_margin"] = panel["EBIT"] / panel["Turnover"]
    panel["return_on_assets"] = panel["EBIT"] / panel["TAsset"]
    panel["asset_growth"] = company_group["TAsset"].pct_change(fill_method=None)
    panel["revenue_growth"] = company_group["Turnover"].pct_change(fill_method=None)
    panel["lagged_mscore_num"] = company_group["mscore_num"].shift(1)
    panel["MScore_trend"] = panel["mscore_num"] - panel["lagged_mscore_num"]

    panel = pd.get_dummies(panel, columns=["Sector 1"], drop_first=True)

    engineered_features = [
        "ebit_margin",
        "return_on_assets",
        "asset_growth",
        "revenue_growth",
        "lagged_mscore_num",
        "MScore_trend",
    ]
    sector_columns = [col for col in panel.columns if col.startswith("Sector 1_")]
    panel = panel.dropna(subset=engineered_features + sector_columns)
    panel = panel.drop(columns=["next_MScore", "next_mscore_num", "MScore"])

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(FEATURES_PATH, index=False)

    print(f"Feature matrix shape: {panel.shape}")
    print("Column names:")
    for col in panel.columns:
        print(col)
    print()
    print(f"Saved engineered features to: {FEATURES_PATH.resolve()}")


def main() -> None:
    validate_source_file(SOURCE_PATH)
    wide_df = load_sheets(SOURCE_PATH)
    long_df = reshape_to_long(wide_df)
    panel = add_target(long_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUTPUT_PATH, index=False)

    print("Class balance of downgrade_flag:")
    print(panel["downgrade_flag"].value_counts(dropna=False).sort_index())
    print()
    print("Sample of resulting DataFrame:")
    print(panel.head(10).to_string(index=False))
    print()
    print(f"Saved processed panel to: {OUTPUT_PATH.resolve()}")

    build_features()


if __name__ == "__main__":
    main()
