
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from boruta import BorutaPy
from mlxtend.feature_selection import SequentialFeatureSelector
from graph_learning import *
import networkx as nx
from PyImpetus import PPIMBC
from aif360.algorithms.preprocessing import DisparateImpactRemover
from fairlearn.preprocessing import CorrelationRemover
from aif360.datasets import BinaryLabelDataset

#import lightgbm as lgb
#from tabpfn import TabPFNClassifier
from scipy import stats
#from GRANDE import GRANDE
from sklearn.feature_selection import SelectKBest, f_classif, SelectPercentile, mutual_info_classif, VarianceThreshold, RFE, chi2, SelectFromModel
from scipy import stats

def data_summary(df: pd.DataFrame, target_column: str):
    df_without_target = df.drop(columns=[target_column])

    num_features = df_without_target.shape[1]
    categorical_features = df_without_target.select_dtypes(include=['object', 'category']).columns
    num_categorical_features = len(categorical_features)

    if num_categorical_features == 0:
        int_columns = df_without_target.select_dtypes(include=['int']).columns.tolist()
        num_categorical_features = len(int_columns)

    num_categorical_features_with_10_or_more_categories = sum(
        df_without_target[cat].nunique() > 10 for cat in categorical_features)

    target_classes = df[target_column].nunique()

    num_samples = df.shape[0]

    return {
        'num_features': num_features,
        'num_categorical_features': num_categorical_features,
        'num_categorical_features_with_10_or_more_categories': num_categorical_features_with_10_or_more_categories,
        'num_target_classes': target_classes,
        'num_samples': num_samples
    }



def evaluate_models_with_cv(data, n_splits=5, random_state=42):
    X = data.drop(columns=['target'], axis=1)
    y = data["target"]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    params_grande = {
        'depth': 5,  # tree depth
        'n_estimators': 2048,  # number of estimators / trees

        'learning_rate_weights': 0.005,  # learning rate for leaf weights
        'learning_rate_index': 0.01,  # learning rate for split indices
        'learning_rate_values': 0.01,  # learning rate for split values
        'learning_rate_leaf': 0.01,  # learning rate for leafs (logits)

        'optimizer': 'adam',  # optimizer
        'cosine_decay_steps': 0,  # decay steps for lr schedule (CosineDecayRestarts)

        'loss': 'crossentropy',
        # loss function (default 'crossentropy' for binary & multi-class classification and 'mse' for regression)
        'focal_loss': False,  # use focal loss {True, False}
        'temperature': 0.0,  # temperature for stochastic re-weighted GD (0.0, 1.0)

        'from_logits': True,  # use logits for weighting {True, False}
        'use_class_weights': True,  # use class weights for training {True, False}

        'dropout': 0.0,
        # dropout rate (here, dropout randomly disables individual estimators of the ensemble during training)

        'selected_variables': 0.8,  # feature subset percentage (0.0, 1.0)
        'data_subset_fraction': 1.0,  # data subset percentage (0.0, 1.0)
    }

    args_grande = {
        'epochs': 200,  # number of epochs for training
        'early_stopping_epochs': 5,  # patience for early stopping (best weights are restored)
        'batch_size': 64,  # batch size for training

        'cat_idx': [],  # put list of categorical indices
        'objective': 'classification',  # objective / task {'binary', 'classification', 'regression'}

        'random_seed': random_state,
        'verbose': 0,
    }

    models = {
        "RandomForest": RandomForestClassifier(random_state=random_state),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=random_state),
        "kNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True, random_state=random_state),
        #"TabPFN": TabPFNClassifier(device='cpu'),
        #"GRANDE": GRANDE(params=params_grande, args=args_grande),
        "MLP": MLPClassifier(random_state=random_state, max_iter=300),
        "LR": LogisticRegression(random_state=random_state),
        "CatBoost": CatBoostClassifier(random_seed=random_state, verbose=False)
    }

    results = {model_name: [] for model_name in models}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for model_name, model in models.items():
            if model_name == "TabPFN":
                model.fit(X_train, y_train.astype(np.int64))
            elif model_name == "GRANDE":
                X_train_, X_val, y_train_, y_val = train_test_split(
                    X_train, y_train, test_size=0.1, random_state=42)
                model.fit(X_train=X_train_,
                          y_train=y_train_,
                          X_val=X_val,
                          y_val=y_val)
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            if model_name == "GRANDE":
                y_pred = np.argmax(y_pred, axis=1)
            f1 = f1_score(y_test, y_pred, average='macro')
            results[model_name].append(f1)
        print("Fold done")

    # Compute mean and 95% confidence interval
    summary = {}
    for model_name, scores in results.items():
        mean = np.mean(scores)
        ci = stats.t.interval(0.95, len(scores)-1, loc=mean, scale=stats.sem(scores))
        summary[model_name] = {
            "mean_macro_f1": mean,
            "95%_CI": ci
        }

    return summary



def regular_selector_fit(selector, X_train, y_train, X_test, y_test):
    selector.fit(X_train, y_train)
    X_train_fs = selector.transform(X_train)
    X_test_fs = selector.transform(X_test)
    y_train_np = y_train.values
    y_test_np = y_test.values
    return X_train_fs, y_train_np, X_test_fs, y_test_np


def select_fair_features_corr(df, target_col, sensitive_col,
                         corr_with_y_thresh=0.2, corr_with_a_thresh=0.1):
    """
    Select features that are:
    - Sufficiently correlated with the target (|corr| >= corr_with_y_thresh)
    - Not too correlated with the sensitive attribute (|corr| <= corr_with_a_thresh)

    Parameters:
    - df: pandas DataFrame containing the dataset
    - target_col: name of the target variable column
    - sensitive_col: name of the sensitive feature column
    - corr_with_y_thresh: minimum absolute correlation with y to keep feature
    - corr_with_a_thresh: maximum absolute correlation with a to keep feature

    Returns:
    - List of selected feature names
    """
    feature_cols = [col for col in df.columns if col not in [target_col, sensitive_col]]

    selected_features = []

    for col in feature_cols:
        if not np.issubdtype(df[col].dtype, np.number):
            continue  # skip non-numeric features

        corr_with_y = df[col].corr(df[target_col])
        corr_with_a = df[col].corr(df[sensitive_col])

        if (abs(corr_with_y) >= corr_with_y_thresh) and (abs(corr_with_a) <= corr_with_a_thresh):
            selected_features.append(col)

    return selected_features

def disparate_impact(X_train, X_test, y_train, y_test, sensitive_attribute, label_name='target', repair_level=1.0):
    # Convert to BinaryLabelDataset
    train_dataset = to_binary_label_dataset(X_train, y_train, label_name, sensitive_attribute)
    test_dataset = to_binary_label_dataset(X_test, y_test, label_name, sensitive_attribute)

    # Apply transformation
    dir_remover = DisparateImpactRemover(repair_level=repair_level, sensitive_attribute=sensitive_attribute)
    train_repaired = dir_remover.fit_transform(train_dataset)
    test_repaired = dir_remover.fit_transform(test_dataset)

    # Convert back to DataFrames (only features)
    X_train_repaired = pd.DataFrame(train_repaired.features, columns=X_train.columns)
    X_test_repaired = pd.DataFrame(test_repaired.features, columns=X_test.columns)

    return X_train_repaired, X_test_repaired

def correlation_remover(X_train, X_test, sensitive_feature_cols, alpha=1.0):
    corr_remover = CorrelationRemover(sensitive_feature_ids = sensitive_feature_cols, alpha=alpha)  # alpha=1.0 = full removal
    X_train_transformed = corr_remover.fit_transform(X_train)
    X_test_transformed = corr_remover.transform(X_test)
    return X_train_transformed, X_test_transformed


def correlation_remover2(X_train, X_test, sensitive_feature_cols, alpha=1.0):
    corr_remover = CorrelationRemover(sensitive_feature_ids=sensitive_feature_cols, alpha=alpha)

    # Transform without sensitive columns
    X_train_transformed = corr_remover.fit_transform(X_train)
    X_test_transformed = corr_remover.transform(X_test)

    # Create DataFrames with only the non-sensitive columns
    non_sensitive_cols = [col for col in X_train.columns if col not in sensitive_feature_cols]
    X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=non_sensitive_cols, index=X_train.index)
    X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=non_sensitive_cols, index=X_test.index)

    # Add sensitive columns back
    X_train_final = pd.concat([X_train_transformed_df, X_train[sensitive_feature_cols]], axis=1)
    X_test_final = pd.concat([X_test_transformed_df, X_test[sensitive_feature_cols]], axis=1)

    return X_train_final, X_test_final

def to_binary_label_dataset(X, y, label_name, sensitive_attribute):
    df = X.copy()
    df[label_name] = y.values  # Attach the multi-class label
    dataset = BinaryLabelDataset(
        df=df,
        label_names=[label_name],
        protected_attribute_names=[sensitive_attribute]
    )
    return dataset

def convert_to_binary_labels(y):
    """Convert multi-class labels to binary: 0 (unfavorable), 1 (favorable for 1 and 2)."""
    return y.apply(lambda label: 1 if label in [1, 2] else 0)