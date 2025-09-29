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
from graph_learningelp import *
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif, SelectPercentile, mutual_info_classif, VarianceThreshold, RFE, chi2, SelectFromModel
from fairfs import fairness_aware_feature_selection
from testing_mb_learning import HITON_MB, IAMB, BAMB, LRH
from utils import regular_selector_fit, select_fair_features_corr, disparate_impact, correlation_remover, convert_to_binary_labels
from fairlearn.metrics import demographic_parity_ratio, equalized_odds_ratio
from my_fair_lady import cade
from faircfs import fair_cfs
from iamb_based_dag import *


def experiment_elp(data, n_splits=5, random_state=42):
    X = data.drop(columns=['target'], axis=1)
    y = data["target"]
    perc20 = 20
    perc40 = 40
    perc60 = 60
    perc80 = 80
    k20 = int(0.2 * len(X.columns))
    k40 = int(0.4 * len(X.columns))
    k60 = int(0.6 * len(X.columns))
    k80 = int(0.8 * len(X.columns))


    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Feature selection methods
    feature_selectors = [
        'SelectKBest_f_classif20',
        'SelectKBest_f_classif40',
        'SelectKBest_f_classif60',
        'SelectKBest_f_classif80',
        'SelectPercentile_mutual_info20',
        'SelectPercentile_mutual_info40',
        'SelectPercentile_mutual_info60',
        'SelectPercentile_mutual_info80',
        'RFE_LogisticRegression20',
        'RFE_LogisticRegression40',
        'RFE_LogisticRegression60',
        'RFE_LogisticRegression80',
        'RFE_RF20',
        'RFE_RF40',
        'RFE_RF60',
        'RFE_RF80',
        'RFE_XGBoost20',
        'RFE_XGBoost40',
        'RFE_XGBoost60',
        'RFE_XGBoost80',
        'L1_LogisticRegression',
        'Boruta',
        'Forward_Selector_LR',
        'Forward_Selector_XGBoost',
        'Forward_Selector_RF',
        'MB', #using HC
        'Exclude_Desc', # use HC and exclude only descendants of problematic variables
        'MB_Exclude_Desc', #use only MB of target and exclude descendants of problematic variables
        'HITON',
        'IAMB',
        'mb_exdesc_iamb',
        'exdesc_iamb'
        'LRH',
        'MFL',
        'FairCFS',
        'FFS_LR', #C. Belitz method variants – non-causal
        'FFS_RF',
        'FFS_XGBoost',
        'Disparate_Impact',
        'Correlation_Remover'
    ]

    # Models
    base_models = {
        "RandomForest": RandomForestClassifier(random_state=random_state),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=random_state),
        "kNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True, random_state=random_state),
        "MLP": MLPClassifier(random_state=random_state, max_iter=300),
        "LR": LogisticRegression(random_state=random_state),
        "CatBoost": CatBoostClassifier(random_seed=random_state, verbose=False)
    }

    # Results dictionary
    results = {
        fs_name: {
            model_name: {'with_fs': [], 'without_fs': [], 'abs_diff': [], 'demographic_parity_ratio_no_fs': [], 'equalized_odds_ratio_no_fs': [], 'demographic_parity_ratio_fs': [], 'equalized_odds_ratio_fs': [], 'num_features': []}
            for model_name in base_models
        }
        for fs_name in feature_selectors
    }

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold_idx + 1}/{n_splits}...")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        X_train_sensitive = X_train["gender"] #needs to be adjusted for other datasets
        X_test_sensitive = X_test["gender"]  # needs to be adjusted for other datasets
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # First: run all models WITHOUT feature selection
        no_fs_preds = {}
        fairness_pdr = {}
        fairness_eor = {}
        for model_name, model in base_models.items():
            clf = model() if callable(model) else model
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            no_fs_preds[model_name] = (f1_score(y_test, y_pred, average='macro'))

            print(f"I ran through {model_name}")

            # Fairness Evaluation
            classes = np.unique(y_test)
            demographic_parity_values = []
            equalized_odds_values = []

            for cls in classes:
                # One-vs-rest binary labels
                y_true_binary = (y_test == cls).astype(int)
                y_pred_binary = (y_pred == cls).astype(int)

                dp_ratio = demographic_parity_ratio(y_true=y_true_binary, y_pred=y_pred_binary,
                                                    sensitive_features=X_test_sensitive)
                eo_ratio = equalized_odds_ratio(y_true=y_true_binary, y_pred=y_pred_binary,
                                                sensitive_features=X_test_sensitive)

                demographic_parity_values.append(dp_ratio)
                equalized_odds_values.append(eo_ratio)

            fairness_pdr[model_name] = np.mean(demographic_parity_values)
            fairness_eor[model_name] = np.mean(equalized_odds_values)


        # Now run all feature selection methods
        for fs_name in feature_selectors:
            if fs_name == 'SelectKBest_f_classif20':
                selector = SelectKBest(score_func=f_classif, k=k20) #adjust if necessary
                X_train_fs, y_train_np, X_test_fs, y_test_np = regular_selector_fit(selector, X_train, y_train, X_test, y_test)
            elif fs_name == 'SelectKBest_f_classif40':
                selector = SelectKBest(score_func=f_classif, k=k40)  # adjust if necessary
                X_train_fs, y_train_np, X_test_fs, y_test_np = regular_selector_fit(selector, X_train, y_train,
                                                                                    X_test, y_test)
            elif fs_name == 'SelectKBest_f_classif60':
                selector = SelectKBest(score_func=f_classif, k=k60)  # adjust if necessary
                X_train_fs, y_train_np, X_test_fs, y_test_np = regular_selector_fit(selector, X_train, y_train,
                                                                                    X_test, y_test)
            elif fs_name == 'SelectKBest_f_classif80':
                selector = SelectKBest(score_func=f_classif, k=k80)  # adjust if necessary
                X_train_fs, y_train_np, X_test_fs, y_test_np = regular_selector_fit(selector, X_train, y_train,
                                                                                    X_test, y_test)

            elif fs_name == 'SelectPercentile_mutual_info20':
                selector = SelectPercentile(score_func=mutual_info_classif, percentile=perc20)
                X_train_fs, y_train_np, X_test_fs, y_test_np = regular_selector_fit(selector, X_train, y_train, X_test,
                                                                                    y_test)
            elif fs_name == 'SelectPercentile_mutual_info40':
                selector = SelectPercentile(score_func=mutual_info_classif, percentile=perc40)
                X_train_fs, y_train_np, X_test_fs, y_test_np = regular_selector_fit(selector, X_train, y_train, X_test,
                                                                                    y_test)
            elif fs_name == 'SelectPercentile_mutual_info60':
                selector = SelectPercentile(score_func=mutual_info_classif, percentile=perc60)
                X_train_fs, y_train_np, X_test_fs, y_test_np = regular_selector_fit(selector, X_train, y_train, X_test,
                                                                                    y_test)
            elif fs_name == 'SelectPercentile_mutual_info80':
                selector = SelectPercentile(score_func=mutual_info_classif, percentile=perc80)
                X_train_fs, y_train_np, X_test_fs, y_test_np = regular_selector_fit(selector, X_train, y_train, X_test,
                                                                                    y_test)

            elif fs_name == 'RFE_LogisticRegression20':
                selector = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=k20)
                X_train_fs, y_train_np, X_test_fs, y_test_np = regular_selector_fit(selector, X_train, y_train, X_test,
                                                                                    y_test)
            elif fs_name == 'RFE_LogisticRegression40':
                selector = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=k40)
                X_train_fs, y_train_np, X_test_fs, y_test_np = regular_selector_fit(selector, X_train, y_train, X_test,
                                                                                    y_test)
            elif fs_name == 'RFE_LogisticRegression60':
                selector = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=k60)
                X_train_fs, y_train_np, X_test_fs, y_test_np = regular_selector_fit(selector, X_train, y_train, X_test,
                                                                                    y_test)
            elif fs_name == 'RFE_LogisticRegression80':
                selector = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=k80)
                X_train_fs, y_train_np, X_test_fs, y_test_np = regular_selector_fit(selector, X_train, y_train, X_test,
                                                                                    y_test)
            elif fs_name == 'RFE_RF20':
                selector = RFE(estimator=RandomForestClassifier(random_state=random_state), n_features_to_select=k20)
                X_train_fs, y_train_np, X_test_fs, y_test_np = regular_selector_fit(selector, X_train, y_train, X_test,
                                                                                    y_test)
            elif fs_name == 'RFE_RF40':
                selector = RFE(estimator=RandomForestClassifier(random_state=random_state), n_features_to_select=k40)
                X_train_fs, y_train_np, X_test_fs, y_test_np = regular_selector_fit(selector, X_train, y_train, X_test,
                                                                                    y_test)
            elif fs_name == 'RFE_RF60':
                selector = RFE(estimator=RandomForestClassifier(random_state=random_state), n_features_to_select=k60)
                X_train_fs, y_train_np, X_test_fs, y_test_np = regular_selector_fit(selector, X_train, y_train, X_test,
                                                                                    y_test)
            elif fs_name == 'RFE_RF80':
                selector = RFE(estimator=RandomForestClassifier(random_state=random_state), n_features_to_select=k80)
                X_train_fs, y_train_np, X_test_fs, y_test_np = regular_selector_fit(selector, X_train, y_train, X_test,
                                                                                    y_test)
            elif fs_name == 'RFE_XGBoost20':
                selector = RFE(estimator=XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=random_state), n_features_to_select=k20)
                X_train_fs, y_train_np, X_test_fs, y_test_np = regular_selector_fit(selector, X_train, y_train, X_test,
                                                                                    y_test)
            elif fs_name == 'RFE_XGBoost40':
                selector = RFE(estimator=XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=random_state), n_features_to_select=k40)
                X_train_fs, y_train_np, X_test_fs, y_test_np = regular_selector_fit(selector, X_train, y_train, X_test,
                                                                                    y_test)
            elif fs_name == 'RFE_XGBoost60':
                selector = RFE(estimator=XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=random_state), n_features_to_select=k60)
                X_train_fs, y_train_np, X_test_fs, y_test_np = regular_selector_fit(selector, X_train, y_train, X_test,
                                                                                    y_test)
            elif fs_name == 'RFE_XGBoost80':
                selector = RFE(estimator=XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=random_state), n_features_to_select=k80)
                X_train_fs, y_train_np, X_test_fs, y_test_np = regular_selector_fit(selector, X_train, y_train, X_test,
                                                                                    y_test)
            elif fs_name == 'L1_LogisticRegression':
                logistic = LogisticRegression(C=1, penalty='l1', solver='liblinear', random_state=7).fit(X_train, y_train)
                model = SelectFromModel(logistic, prefit=True)
                X_train_fs = model.transform(X_train)
                X_test_fs = model.transform(X_test)
                y_train_np = y_train.values
                y_test_np = y_test.values

            elif fs_name == 'Boruta':
                rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=random_state)
                boruta_selector = BorutaPy(estimator=rf, n_estimators='auto', verbose=2, random_state=random_state)
                boruta_selector.fit(X_train.values, y_train.values)
                X_train_fs = boruta_selector.transform(X_train.values)
                X_test_fs = boruta_selector.transform(X_test.values)
                y_train_np = y_train.values
                y_test_np = y_test.values
            elif fs_name == 'Forward_Selector_LR':
                ffs = SequentialFeatureSelector(LogisticRegression(max_iter=10000), k_features='best', forward=True, n_jobs=-1)
                ffs.fit(X_train, y_train)
                features = list(ffs.k_feature_names_)
                X_train_fs = X_train[features]
                X_test_fs = X_test[features]
                y_train_np = y_train.values
                y_test_np = y_test.values
            elif fs_name == 'Forward_Selector_XGBoost':
                ffs = SequentialFeatureSelector(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=random_state), k_features='best', forward=True, n_jobs=-1)
                ffs.fit(X_train, y_train)
                features = list(ffs.k_feature_names_)
                X_train_fs = X_train[features]
                X_test_fs = X_test[features]
                y_train_np = y_train.values
                y_test_np = y_test.values
            elif fs_name == 'Forward_Selector_RF':
                ffs = SequentialFeatureSelector(RandomForestClassifier(random_state=random_state), k_features='best', forward=True, n_jobs=-1)
                ffs.fit(X_train, y_train)
                features = list(ffs.k_feature_names_)
                X_train_fs = X_train[features]
                X_test_fs = X_test[features]
                y_train_np = y_train.values
                y_test_np = y_test.values
            elif fs_name == 'MB':
                data_hc = X_train.copy()
                data_hc["target"] = y_train
                data_hc.to_csv("data.csv", index=False)
                column_names = learn_graph(data_hc, "discrete") # adjust if necessary for other experiments
                if not column_names:
                    X_train_fs = X_train
                    X_test_fs = X_test
                    y_train_np = y_train.values
                    y_test_np = y_test.values
                else:
                    X_train_fs = X_train[column_names]
                    X_test_fs = X_test[column_names]
                    y_train_np = y_train.values
                    y_test_np = y_test.values
            elif fs_name == 'Exclude_Desc':
                data_hc = X_train.copy()
                data_hc["target"] = y_train
                data_hc.to_csv("data.csv", index=False)
                descendants = learn_graph_exc_desc(data_hc, "discrete")  # adjust if necessary for other experiments
                #bayes_model = BayesianModel(model.edges())  # convert DAG to BayesianModel
                #descendants = list(nx.descendants(bayes_model, "gender"))
                if not descendants:
                    X_train_fs = X_train.drop("gender", axis=1)
                    X_test_fs = X_test.drop("gender", axis=1)
                    y_train_np = y_train.values
                    y_test_np = y_test.values
                else:
                    X_train_fs = X_train.drop(descendants, axis=1)
                    X_test_fs = X_test.drop(descendants, axis=1)
                    if "gender" not in descendants:
                        X_train_fs = X_train_fs.drop(["gender"], axis=1)
                        X_test_fs = X_test_fs.drop(["gender"], axis=1)
                    y_train_np = y_train.values
                    y_test_np = y_test.values
            elif fs_name == 'MB_Exclude_Desc':
                data_hc = X_train.copy()
                data_hc["target"] = y_train
                data_hc.to_csv("data.csv", index=False)
                column_names, descendants = learn_graph_exc_desc_mb(data_hc, "discrete")  # adjust if necessary for other experiments
                #bayes_model = BayesianModel(model.edges())  # convert DAG to BayesianModel
                #descendants = list(nx.descendants(bayes_model, "gender"))
                #column_names = list(model.get_markov_blanket('target'))  # adjust for other experiments
                if column_names:
                    to_keep = list(set(column_names) - set(descendants) - set("target"))
                    X_train_fs = X_train[to_keep]
                    X_test_fs = X_test[to_keep]
                #X_train_fs = X_train_fs.drop(["gender"], axis=1)
                #X_test_fs = X_test_fs.drop(["gender"], axis=1)
                    y_train_np = y_train.values
                    y_test_np = y_test.values
                else:
                    X_train_fs = X_train.drop(descendants, axis=1)
                    X_train_fs = X_train_fs.drop(["gender"], axis=1)
                    X_test_fs = X_test.drop(descendants, axis=1)
                    X_test_fs = X_test_fs.drop(["gender"], axis=1)
                    y_train_np = y_train.values
                    y_test_np = y_test.values
            elif fs_name == 'HITON':
                mb, ci = HITON_MB(data, "target", 0.05, is_discrete=False) # adjust for other experiments
                X_train_fs = X_train[mb]
                X_test_fs = X_test[mb]
                y_train_np = y_train.values
                y_test_np = y_test.values
            elif fs_name == 'IAMB':
                mb, ci = IAMB(data, "target", 0.05, is_discrete=False) # adjust for other experiments
                X_train_fs = X_train[mb]
                X_test_fs = X_test[mb]
                y_train_np = y_train.values
                y_test_np = y_test.values
            elif fs_name == 'mb_exdesc_iamb':
                sel = mb_exdesc_iamb(data, "gender")
                X_train_fs = X_train[sel]
                X_test_fs = X_test[sel]
                y_train_np = y_train.values
                y_test_np = y_test.values
            elif fs_name == 'exdesc_iamb':
                sel = exdesc_iamb(data, "gender")
                X_train_fs = X_train[sel]
                X_test_fs = X_test[sel]
                y_train_np = y_train.values
                y_test_np = y_test.values
            elif fs_name == 'LRH':
                mb, ci = LRH(data, "target", 0.05, is_discrete=False) # adjust for other experiments
                X_train_fs = X_train[mb]
                X_test_fs = X_test[mb]
                y_train_np = y_train.values
                y_test_np = y_test.values
            elif fs_name == "MFL":
                X_train_fs, X_test_fs = my_fair_lady(X_train, X_test, ["gender"], dominant_data_type="discrete")
                y_train_np = y_train.values
                y_test_np = y_test.values
            elif fs_name == "FairCFS":
                t = fair_cfs("gender")
                t = list(t)
                X_train_fs = X_train[t]
                X_test_fs = X_test[t]
                y_train_np = y_train.values
                y_test_np = y_test.values
            elif fs_name == 'FFS_LR':
                selected_features = fairness_aware_feature_selection(
                    X=X_train,
                    y=y_train,
                    protected_column="gender",
                    unfairness_metric='statistical_parity',
                    model=LogisticRegression(max_iter=500, random_state=random_state),
                    unfairness_weight=1.0,
                    iterations=3,
                    threshold=0.5
                )
                X_train_fs = X_train[selected_features]
                X_test_fs = X_test[selected_features]
                y_train_np = y_train.values
                y_test_np = y_test.values
            elif fs_name == 'FFS_RF':
                selected_features = fairness_aware_feature_selection(
                    X=X_train,
                    y=y_train,
                    protected_column="gender",
                    unfairness_metric='statistical_parity',
                    model=RandomForestClassifier(random_state=random_state),
                    unfairness_weight=1.0,
                    iterations=10,
                    threshold=0.5
                )
                X_train_fs = X_train[selected_features]
                X_test_fs = X_test[selected_features]
                y_train_np = y_train.values
                y_test_np = y_test.values
            elif fs_name == 'FFS_XGBoost':
                selected_features = fairness_aware_feature_selection(
                    X=X_train,
                    y=y_train,
                    protected_column="gender",
                    unfairness_metric='statistical_parity',
                    model=XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=random_state),
                    unfairness_weight=1.0,
                    iterations=3,
                    threshold=0.5
                )
                X_train_fs = X_train[selected_features]
                X_test_fs = X_test[selected_features]
                y_train_np = y_train.values
                y_test_np = y_test.values

            elif fs_name == 'Disparate_Impact':
                y_train_binary = convert_to_binary_labels(y_train) #adjust for other datasets
                y_test_binary = convert_to_binary_labels(y_test) # adjust for other datasets
                X_train_fs, X_test_fs = disparate_impact(X_train, X_test, y_train_binary, y_test_binary, 'gender', label_name='target', repair_level=1.0)
                y_train_np = y_train.values
                y_test_np = y_test.values
            elif fs_name == 'Correlation_Remover':
                X_train_fs, X_test_fs = correlation_remover(X_train, X_test, ["gender"], alpha=1.0)
                y_train_np = y_train.values
                y_test_np = y_test.values

            for model_name, model in base_models.items():
                clf = model() if callable(model) else model
                clf.fit(X_train_fs, y_train_np)
                y_pred = clf.predict(X_test_fs)
                print(f"I ran through {fs_name}, {model_name}")

                f1_fs = f1_score(y_test_np, y_pred, average='macro')
                f1_no_fs = no_fs_preds[model_name]
                dpr_no_fs = fairness_pdr[model_name]
                eor_no_fs = fairness_eor[model_name]
                abs_diff = abs(f1_fs - f1_no_fs)
                dimensions = X_train_fs.shape
                rows, columns = dimensions
                num_features = columns

                # Fairness Evaluation
                classes = np.unique(y_test_np)
                demographic_parity_values = []
                equalized_odds_values = []

                for cls in classes:
                    # One-vs-rest binary labels
                    y_true_binary = (y_test_np == cls).astype(int)
                    y_pred_binary = (y_pred == cls).astype(int)
                    if np.all(y_true_binary == 0) or np.all(y_pred_binary == 0):
                        dp_ratio = 1
                        eo_ratio = 1
                    else:
                        dp_ratio = demographic_parity_ratio(y_true=y_true_binary, y_pred=y_pred_binary,
                                                            sensitive_features=X_test_sensitive)
                        eo_ratio = equalized_odds_ratio(y_true=y_true_binary, y_pred=y_pred_binary,
                                                        sensitive_features=X_test_sensitive)

                    demographic_parity_values.append(dp_ratio)
                    equalized_odds_values.append(eo_ratio)

                demographic_parity_fs = np.mean(demographic_parity_values)
                equalized_odds_fs = np.mean(equalized_odds_values)


                results[fs_name][model_name]['with_fs'].append(f1_fs)
                results[fs_name][model_name]['without_fs'].append(f1_no_fs)
                results[fs_name][model_name]['abs_diff'].append(abs_diff)
                results[fs_name][model_name]['demographic_parity_ratio_fs'].append(demographic_parity_fs)
                results[fs_name][model_name]['equalized_odds_ratio_fs'].append(equalized_odds_fs)
                results[fs_name][model_name]['demographic_parity_ratio_no_fs'].append(dpr_no_fs)
                results[fs_name][model_name]['equalized_odds_ratio_no_fs'].append(eor_no_fs)
                results[fs_name][model_name]['num_features'].append(num_features)

    # Compute summary
    summary = {
        fs_name: {
            model_name: {
                'mean_f1_with_fs': np.mean(data['with_fs']),
                'mean_f1_without_fs': np.mean(data['without_fs']),
                'mean_abs_diff': np.mean(data['abs_diff']),
                'demographic_parity_ratio_fs': np.mean(data['demographic_parity_ratio_fs']),
                'equalized_odds_ratio_fs': np.mean(data['equalized_odds_ratio_fs']),
                'demographic_parity_ratio_no_fs': np.mean(data['demographic_parity_ratio_no_fs']),
                'equalized_odds_ratio_no_fs': np.mean(data['equalized_odds_ratio_no_fs']),
                'num_features': np.mean(data['num_features'])
            }
            for model_name, data in models.items()
        }
        for fs_name, models in results.items()
    }
    return summary