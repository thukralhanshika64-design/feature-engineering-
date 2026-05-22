import numpy as np
import streamlit as st
import pandas as pd

from feature_selection import (
    correlation_filter,
    generate_synthetic_dataset,
    model_selection,
    variance_filter,
)


def prepare_uploaded_data(df: pd.DataFrame, target_column: str):
    """Keep only numeric feature columns and fill missing values."""
    y = df[target_column].copy()
    X = df.drop(columns=[target_column])

    # Convert string numeric columns to actual numeric values when possible.
    X_converted = X.apply(lambda col: pd.to_numeric(col, errors="coerce") if col.dtype == object else col)

    # Drop columns that are still non-numeric after conversion.
    X_numeric = X_converted.select_dtypes(include=[np.number]).copy()
    dropped_non_numeric = X.columns.difference(X_numeric.columns).tolist()

    # Drop any columns that became all-NaN after conversion.
    all_nan_cols = X_numeric.columns[X_numeric.isna().all()].tolist()
    if all_nan_cols:
        X_numeric = X_numeric.drop(columns=all_nan_cols)
        dropped_non_numeric.extend(all_nan_cols)

    if X_numeric.empty:
        raise ValueError(
            "Uploaded CSV has no numeric feature columns after cleaning. Please upload a dataset with numeric inputs."
        )

    missing_count = int(X_numeric.isna().sum().sum())
    if missing_count > 0:
        X_numeric = X_numeric.fillna(X_numeric.median())

    return X_numeric, y, dropped_non_numeric, missing_count

st.set_page_config(page_title="Feature Selection App", layout="wide")
st.title("Feature Selection for High-Dimensional Data")
st.markdown(
    "Use this app to filter low-variance features, remove highly correlated columns, and select the strongest features with a model-based approach."
)

with st.sidebar:
    st.header("Configuration")
    data_source = st.radio(
        "Dataset source",
        ["Example synthetic dataset", "Upload CSV"],
        index=0,
    )

    if data_source == "Example synthetic dataset":
        n_samples = st.number_input("Samples", min_value=100, max_value=10000, value=1000, step=100)
        n_features = st.number_input("Features", min_value=50, max_value=1000, value=500, step=50)
        n_informative = st.number_input("Informative features", min_value=1, max_value=200, value=20, step=1)
        n_redundant = st.number_input("Redundant features", min_value=0, max_value=200, value=50, step=1)
        random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        target_column = None

    st.markdown("---")
    st.subheader("Selection settings")
    variance_threshold = st.slider("Variance threshold", min_value=0.0, max_value=0.10, value=0.01, step=0.005)
    correlation_threshold = st.slider("Correlation threshold", min_value=0.70, max_value=0.99, value=0.90, step=0.01)
    importance_threshold = st.selectbox(
        "Model importance threshold",
        ["median", "mean"],
        index=0,
    )
    n_estimators = st.slider("Random forest trees", min_value=10, max_value=500, value=100, step=10)

if data_source == "Example synthetic dataset":
    df, y = generate_synthetic_dataset(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        random_state=random_state,
    )
    X = df
    target_name = "target"
else:
    if uploaded_file is None:
        st.warning("Upload a CSV file to continue or choose the example dataset.")
        st.stop()

    df = pd.read_csv(uploaded_file)
    if df.empty:
        st.error("Uploaded file is empty. Please upload a valid CSV.")
        st.stop()

    target_column = st.selectbox("Choose target column", options=df.columns)
    if target_column is None:
        st.warning("Select a target column to continue.")
        st.stop()

    target_name = target_column
    try:
        X, y, dropped_non_numeric, missing_count = prepare_uploaded_data(df, target_column)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if dropped_non_numeric:
        st.warning(
            "The following non-numeric columns were removed before feature selection: "
            + ", ".join(dropped_non_numeric)
        )

st.sidebar.markdown("---")
st.sidebar.write("Selected features will be generated based on the current settings.")

st.header("Input data summary")
col1, col2 = st.columns(2)
col1.metric("Rows", X.shape[0])
col1.metric("Initial features", X.shape[1])
col2.metric("Target values", len(y.unique()) if hasattr(y, 'unique') else len(y))

st.subheader("Preview of the first 5 rows")
st.dataframe(pd.concat([X.head(), y.head()], axis=1))

if data_source == "Upload CSV":
    st.subheader("Upload cleanup summary")
    st.write(f"Numeric feature columns after cleaning: {X.shape[1]}")
    st.write(f"Target column: {target_name}")
        st.write(f"Missing numeric values filled: {missing_count}")
    else:
        st.info("All uploaded feature columns were numeric or converted successfully.")

with st.spinner("Applying feature selection steps..."):
    try:
        df_variance, variance_cols = variance_filter(X, threshold=variance_threshold)
        df_correlation, dropped_corr = correlation_filter(df_variance, threshold=correlation_threshold)
        X_final, selected_features = model_selection(
            df_correlation,
            y,
            threshold=importance_threshold,
            n_estimators=n_estimators,
        )
    except Exception as exc:
        st.error(f"Feature selection failed: {exc}")
        st.stop()

st.header("Feature selection results")
step_cols = st.columns(3)
step_cols[0].metric("After variance filter", len(variance_cols))
step_cols[1].metric("After correlation filter", df_correlation.shape[1])
step_cols[2].metric("Final selected features", X_final.shape[1])

st.subheader("Dropped correlation features")
if len(dropped_corr) == 0:
    st.info("No highly correlated features were dropped at the current threshold.")
else:
    st.write(f"Dropped {len(dropped_corr)} features due to correlation above {correlation_threshold}.")
    with st.expander("View dropped features"):
        st.write(dropped_corr)

st.subheader("Selected features")
if len(selected_features) == 0:
    st.warning("No features were selected by the model. Try lowering the thresholds or increasing the number of trees.")
else:
    st.write(f"Selected {len(selected_features)} features after model-based selection.")
    with st.expander("View selected feature names"):
        st.write(selected_features)

selected_df = pd.DataFrame(X_final, columns=selected_features)
selected_df[target_name] = y.reset_index(drop=True)

st.download_button(
    label="Download selected feature dataset",
    data=selected_df.to_csv(index=False).encode("utf-8"),
    file_name="selected_features.csv",
    mime="text/csv",
)

st.markdown("---")
st.subheader("How this app works")
st.markdown(
    "1. Variance filter removes nearly constant columns.\n"
    "2. Correlation filter removes redundant features with strong pairwise correlation.\n"
    "3. Model-based selection uses a random forest to choose the most important features."
)
