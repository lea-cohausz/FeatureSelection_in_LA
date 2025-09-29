import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics, model_selection, preprocessing
from mlxtend.feature_selection import SequentialFeatureSelector
from unfairness_metrics import CombinedMetric  # Ensure this is available in your path


def fairness_aware_feature_selection(
        X: pd.DataFrame,
        y: pd.DataFrame,
        protected_column: str,
        unfairness_metric: str,
        model,
        unfairness_weight: float = 1.0,
        iterations: int = 10,
        threshold: float = 0.5
):
    """
    Performs fairness-aware feature selection using a custom unfairness metric.

    Parameters:
        X (pd.DataFrame): Features (excluding the label).
        y (pd.DataFrame): Label variable (single column dataframe).
        protected_column (str): Name of the protected attribute in X.
        unfairness_metric (str): Metric name from CombinedMetric.
        model: Scikit-learn compatible model (e.g., LogisticRegression).
        unfairness_weight (float): Weight for unfairness in combined metric.
        iterations (int): Number of cross-validated feature selection runs.
        threshold (float): Minimum selection frequency to include a feature.

    Returns:
        selected_features (np.ndarray): Array of selected feature names.
    """
    X = X.copy()
    y = y.squeeze()  # Convert to Series if needed
    protected_groups = X[protected_column]
    X = X.drop(columns=[protected_column])

    feature_names = X.columns.to_numpy()

    # Scale features
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)

    feature_selection_counts = np.zeros(X.shape[1])

    for i in tqdm(range(iterations), desc='Feature selection runs'):
        kf = model_selection.KFold(n_splits=4, shuffle=True, random_state=i)

        fairness_metric = CombinedMetric(
            accuracy_metric_func=metrics.accuracy_score,
            protected_groups=protected_groups,
            unfairness_metric=unfairness_metric,
            unfairness_weight=unfairness_weight
        )

        scorer = metrics.make_scorer(fairness_metric)

        sfs = SequentialFeatureSelector(
            model,
            k_features='best',
            scoring=scorer,
            cv=kf,
            n_jobs=-1
        )
        sfs.fit(X_scaled, y)

        for idx in sfs.k_feature_idx_:
            feature_selection_counts[idx] += 1

    feature_selection_freq = feature_selection_counts / iterations
    selected_indices = np.where(feature_selection_freq >= threshold)[0]
    selected_features = feature_names[selected_indices]

    #print("Selected feature indices:", selected_indices)
    #print("Selected feature names:", selected_features)

    return selected_features
