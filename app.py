from io import StringIO

import pandas as pd
import streamlit as st

from prediction_utils import (
    load_demo_input,
    read_uploaded_table,
    score_feature_dataframe,
    template_input,
)


st.set_page_config(page_title="Credit Risk Predictor", layout="wide")

st.title("European Credit Downgrade Predictor")
st.write(
    "Upload a feature-engineered CSV or Excel file with the same schema as "
    "`data/processed/features.csv`, and the app will score downgrade risk with "
    "the best Random Forest model."
)

with st.sidebar:
    st.header("How To Use")
    st.markdown(
        "1. Run the training pipeline so `features.csv` and `rf_fold1.pkl` exist.\n"
        "2. Upload a CSV or Excel file with the same feature columns as `features.csv`.\n"
        "3. Review the predicted downgrade probabilities and download the results."
    )

template_csv = template_input(20).to_csv(index=False)
st.download_button(
    "Download Sample Input Template",
    data=template_csv,
    file_name="sample_features_template.csv",
    mime="text/csv",
)

source = st.radio(
    "Choose a scoring source",
    ["Use bundled 2018 demo sample", "Upload CSV or Excel"],
    horizontal=True,
)

input_df = None

if source == "Use bundled 2018 demo sample":
    input_df = load_demo_input(200)
else:
    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
    )
    if uploaded_file is not None:
        input_df = read_uploaded_table(uploaded_file.name, uploaded_file.getvalue())

if input_df is not None:
    st.subheader("Input Preview")
    st.dataframe(input_df.head(20), use_container_width=True)

    try:
        scored_df, summary = score_feature_dataframe(input_df)
    except Exception as exc:
        st.error(str(exc))
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Input Rows", summary["input_rows"])
        col2.metric("Scored Rows", summary["scored_rows"])
        col3.metric("Dropped Non-finite Rows", summary["dropped_non_finite_rows"])

        st.subheader("Prediction Results")
        display_columns = [
            col
            for col in [
                "Company name",
                "Country",
                "Region",
                "year",
                "downgrade_probability",
                "predicted_downgrade_0_5",
            ]
            if col in scored_df.columns
        ]
        if display_columns:
            st.dataframe(
                scored_df[display_columns]
                .sort_values("downgrade_probability", ascending=False)
                .head(50),
                use_container_width=True,
            )
        else:
            st.dataframe(
                scored_df.sort_values("downgrade_probability", ascending=False).head(50),
                use_container_width=True,
            )

        risk_stats = pd.Series(scored_df["downgrade_probability"]).describe()
        st.subheader("Probability Summary")
        st.dataframe(risk_stats.to_frame(name="value"), use_container_width=True)

        csv_buffer = StringIO()
        scored_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "Download Predictions CSV",
            data=csv_buffer.getvalue(),
            file_name="credit_risk_predictions.csv",
            mime="text/csv",
        )
else:
    st.info("Choose a source above to load sample data or upload your own file.")
