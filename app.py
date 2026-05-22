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
    X_converted = X.apply(
        lambda col: pd.to_numeric(col, errors="coerce") if col.dtype == object else col
    )

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
            "Uploaded CSV has no numeric feature columns after cleaning. "
            "Please upload a dataset with numeric inputs."
        )

    missing_count = int(X_numeric.isna().sum().sum())
    if missing_count > 0:
        X_numeric = X_numeric.fillna(X_numeric.median())

    return X_numeric, y, dropped_non_numeric, missing_count


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Feature Selection App", layout="wide")
st.title("Feature Selection for High-Dimensional Data")
st.markdown(
    "Filter low-variance features, remove highly correlated columns, and select "
    "the strongest features with a model-based approach."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    data_source = st.radio(
        "Dataset source",
        ["Example synthetic dataset", "Upload CSV"],
        index=0,
    )

    # ── Data source settings ──────────────────────────────────────────────────
    if data_source == "Example synthetic dataset":
        n_samples = st.number_input(
            "Samples", min_value=100, max_value=10_000, value=1_000, step=100
        )
        n_features = st.number_input(
            "Features", min_value=50, max_value=1_000, value=500, step=50
        )
        n_informative = st.number_input(
            "Informative features", min_value=1, max_value=200, value=20, step=1
        )
        n_redundant = st.number_input(
            "Redundant features", min_value=0, max_value=200, value=50, step=1
        )
        random_state = st.number_input(
            "Random seed", min_value=0, max_value=9_999, value=42, step=1
        )
        uploaded_file = None
        target_column = None

    else:  # Upload CSV
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        target_column = None  # resolved below once file is loaded

    st.markdown("---")

    # ── Feature-selection settings ────────────────────────────────────────────
    st.subheader("Selection settings")
    variance_threshold = st.slider(
        "Variance threshold", min_value=0.0, max_value=0.10, value=0.01, step=0.005
    )
    correlation_threshold = st.slider(
        "Correlation threshold", min_value=0.70, max_value=0.99, value=0.90, step=0.01
    )
    importance_threshold = st.selectbox(
        "Model importance threshold", ["median", "mean"], index=0
    )
    n_estimators = st.slider(
        "Random forest trees", min_value=10, max_value=500, value=100, step=10
    )

# ── Data loading ──────────────────────────────────────────────────────────────
dropped_non_numeric: list = []
missing_count: int = 0

if data_source == "Example synthetic dataset":
    df, y = generate_synthetic_dataset(
        n_samples=int(n_samples),
        n_features=int(n_features),
        n_informative=int(n_informative),
        n_redundant=int(n_redundant),
        random_state=int(random_state),
    )
    X = df
    target_name = "target"

else:  # Upload CSV
    if uploaded_file is None:
        st.info("Upload a CSV file using the sidebar, or switch to the example dataset.")
        st.stop()

    df = pd.read_csv(uploaded_file)
    if df.empty:
        st.error("Uploaded file is empty. Please upload a valid CSV.")
        st.stop()

    # ── Target column picker lives in the SIDEBAR after upload ────────────────
    # Placing it in the sidebar keeps the main body clean and avoids layout jumps.
    with st.sidebar:
        st.markdown("---")
        target_column = st.selectbox(
            "Choose target column", options=df.columns, key="target_col"
        )

    target_name = target_column
    try:
        X, y, dropped_non_numeric, missing_count = prepare_uploaded_data(df, target_column)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if dropped_non_numeric:
        st.warning(
            "Non-numeric columns removed before feature selection: "
            + ", ".join(dropped_non_numeric)
        )

# ── Input data summary ────────────────────────────────────────────────────────
st.header("Input data summary")
col1, col2 = st.columns(2)
col1.metric("Rows", X.shape[0])
col1.metric("Initial features", X.shape[1])

# y.unique() works for pd.Series; fall back for numpy arrays
n_unique_targets = len(pd.Series(y).unique())
col2.metric("Unique target values", n_unique_targets)

st.subheader("Preview of the first 5 rows")
st.dataframe(pd.concat([X.head(), pd.Series(y, name=target_name).head()], axis=1))

if data_source == "Upload CSV":
    st.subheader("Upload cleanup summary")
    st.write(f"Numeric feature columns after cleaning: **{X.shape[1]}**")
    st.write(f"Target column: **{target_name}**")
    st.write(f"Missing numeric values filled with median: **{missing_count}**")
    if dropped_non_numeric:
        with st.expander("View removed non-numeric columns"):
            st.write(dropped_non_numeric)
    else:
        st.success("All feature columns were numeric or converted successfully.")

# ── Feature selection pipeline ────────────────────────────────────────────────
with st.spinner("Applying feature selection steps — this may take a moment on large datasets…"):
    try:
        df_variance, variance_cols = variance_filter(X, threshold=variance_threshold)

        # Warn user if correlation filtering will be slow
        if df_variance.shape[1] > 500:
            st.info(
                f"Computing pairwise correlations for {df_variance.shape[1]} features. "
                "This may take a moment."
            )

        df_correlation, dropped_corr = correlation_filter(
            df_variance, threshold=correlation_threshold
        )

        # FIX: model_selection now returns 4 values (added task_type + importances)
        X_final, selected_features, task_type, importances = model_selection(
            df_correlation,
            pd.Series(y),
            threshold=importance_threshold,
            n_estimators=int(n_estimators),
        )

    except Exception as exc:
        st.error(f"Feature selection failed: {exc}")
        st.stop()

# ── Results ───────────────────────────────────────────────────────────────────
st.header("Feature selection results")

task_label = "🔵 Classification" if task_type == "classification" else "🟠 Regression"
st.caption(f"Detected task type: {task_label}")

step_cols = st.columns(3)
step_cols[0].metric("After variance filter", len(variance_cols))
step_cols[1].metric("After correlation filter", df_correlation.shape[1])
step_cols[2].metric("Final selected features", X_final.shape[1])

# Dropped correlated features
st.subheader("Dropped correlation features")
if len(dropped_corr) == 0:
    st.info("No highly correlated features were dropped at the current threshold.")
else:
    st.write(f"Dropped **{len(dropped_corr)}** features with correlation above {correlation_threshold}.")
    with st.expander("View dropped feature names"):
        st.write(dropped_corr)

# Selected features
st.subheader("Selected features")
if len(selected_features) == 0:
    st.warning(
        "No features were selected. Try lowering the thresholds or increasing the number of trees."
    )
else:
    st.write(f"Selected **{len(selected_features)}** features after model-based selection.")
    with st.expander("View selected feature names"):
        st.write(selected_features)

# ── Feature importance chart ──────────────────────────────────────────────────
st.subheader("Feature importances (top 20)")

top_n = min(20, len(importances))
top_importances = importances.head(top_n).reset_index()
top_importances.columns = ["Feature", "Importance"]

# Highlight selected features
top_importances["Selected"] = top_importances["Feature"].isin(selected_features)
st.dataframe(
    top_importances.style.format({"Importance": "{:.4f}"}).apply(
        lambda row: [
            "background-color: #d4edda" if row["Selected"] else "" for _ in row
        ],
        axis=1,
    ),
    use_container_width=True,
)

st.bar_chart(top_importances.set_index("Feature")["Importance"])

# ── Download ──────────────────────────────────────────────────────────────────
selected_df = X_final.copy()
selected_df[target_name] = pd.Series(y).reset_index(drop=True)

st.download_button(
    label="⬇ Download selected feature dataset (CSV)",
    data=selected_df.to_csv(index=False).encode("utf-8"),
    file_name="selected_features.csv",
    mime="text/csv",
)

# ── How it works ──────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("How this app works")
st.markdown(
    """
1. **Variance filter** — removes nearly constant columns (low signal).
2. **Correlation filter** — removes redundant features with strong pairwise correlation.
3. **Model-based selection** — fits a Random Forest and keeps features above the importance threshold.
   - Classification targets → `RandomForestClassifier`
   - Continuous regression targets → `RandomForestRegressor` *(auto-detected)*
"""
)
