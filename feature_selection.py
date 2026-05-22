import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
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
    """Drop features with low variance."""
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(df)
    selected_columns = df.columns[selector.get_support()].tolist()
    return df[selected_columns], selected_columns


def correlation_filter(df: pd.DataFrame, threshold: float = 0.90):
    """Drop features that are strongly correlated with another feature."""
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop), to_drop


def model_selection(
    df: pd.DataFrame,
    target: pd.Series,
    threshold: str = "median",
    n_estimators: int = 100,
    random_state: int = 42,
):
    """Select features using a tree-based model importance threshold."""
    estimator = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    selector = SelectFromModel(estimator=estimator, threshold=threshold)
    selector.fit(df, target)
    selected_columns = df.columns[selector.get_support()].tolist()
    X_final = pd.DataFrame(selector.transform(df), columns=selected_columns)
    return X_final, selected_columns
