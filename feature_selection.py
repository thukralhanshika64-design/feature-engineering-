import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, VarianceThreshold


def generate_synthetic_dataset(
    n_samples: int = 1000,
    n_features: int = 500,
    n_informative: int = 20,
    n_redundant: int = 50,
    random_state: int = 42,
):
    """Generate a synthetic classification dataset with many columns."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        random_state=random_state,
        shuffle=False,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    return df, pd.Series(y, name="target")


def variance_filter(df: pd.DataFrame, threshold: float = 0.01):
    """Drop features with variance below the given threshold."""
    selector = VarianceThreshold(threshold=threshold)
    selector.fit_transform(df)
    selected_columns = df.columns[selector.get_support()].tolist()
    return df[selected_columns], selected_columns


def correlation_filter(df: pd.DataFrame, threshold: float = 0.90):
    """Drop features that are strongly correlated with another feature.

    Uses the upper triangle of the absolute correlation matrix. For each
    pair that exceeds the threshold the *later* column (higher index) is
    dropped, keeping the first occurrence.
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=to_drop), to_drop


def _is_regression_target(y: pd.Series) -> bool:
    """Return True if the target looks like a continuous regression target."""
    if not pd.api.types.is_numeric_dtype(y):
        return False
    n_unique = y.nunique()
    # Treat as regression when there are many unique numeric values
    return n_unique > 20 and (n_unique / len(y)) > 0.05


def model_selection(
    df: pd.DataFrame,
    target: pd.Series,
    threshold: str = "median",
    n_estimators: int = 100,
    random_state: int = 42,
):
    """Select features using a tree-based model importance threshold.

    Automatically chooses RandomForestClassifier for classification targets
    and RandomForestRegressor for continuous regression targets.

    Returns
    -------
    X_final : pd.DataFrame
        Dataset containing only the selected features.
    selected_columns : list[str]
        Names of the selected features.
    task_type : str
        Either ``'classification'`` or ``'regression'``.
    importances : pd.Series
        Feature importances for all features that survived the correlation
        filter, indexed by feature name.
    """
    if _is_regression_target(target):
        estimator = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        task_type = "regression"
    else:
        estimator = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        task_type = "classification"

    selector = SelectFromModel(estimator=estimator, threshold=threshold)
    selector.fit(df, target)

    selected_columns = df.columns[selector.get_support()].tolist()
    X_final = pd.DataFrame(selector.transform(df), columns=selected_columns)

    # Extract importances from the fitted estimator inside the selector
    importances = pd.Series(
        selector.estimator_.feature_importances_,
        index=df.columns,
        name="importance",
    ).sort_values(ascending=False)

    return X_final, selected_columns, task_type, importances
