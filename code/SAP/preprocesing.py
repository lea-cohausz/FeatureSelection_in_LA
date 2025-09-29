import pandas as pd
import re

def preprcoess_sap(data):
    class_recode = [0, 1, 2, 3]
    status = ["Pass", "Good", "Vg", "Best"]
    data[["esp", "tnp", "iap", "twp"]] = data[["esp", "tnp", "iap", "twp"]].replace(status, class_recode)
    income_recode = [0, 1, 2, 3, 4]
    income = ["Low", "Medium", "Am", "High", "Vh"]
    data[["fmi"]] = data[["fmi"]].replace(income, income_recode)
    family_recode = [0, 1, 2]
    family = ["Small", "Average", "Large"]
    data[["fs", "nf", "tt"]] = data[["fs", "nf", "tt"]].replace(family, family_recode)
    qualification_recode = [0, 1, 2, 3, 4, 5]
    qualification = ["Il", "Um", "10", "12", "Degree", "Pg"]
    data[["fq", "mq"]] = data[["fq", "mq"]].replace(qualification, qualification_recode)
    study_recode = [0, 1, 2]
    study = ["Poor", "Average", "Good"]
    data[["sh", "atd"]] = data[["sh", "atd"]].replace(study, study_recode)
    data.rename(columns={'esp': 'target'}, inplace=True)

    # different kinds of features requiring different pre-processing
    num_cols = data.select_dtypes(include=["int32", "int64", "float64"]).columns.tolist()
    bin_cols = [col for col in data.columns if data[col].nunique() == 2 and col not in num_cols]
    cat_cols = [col for col in data.columns if col not in num_cols and col not in bin_cols]

    # recoding of binary features
    data[bin_cols] = data[bin_cols].apply(lambda x: pd.factorize(x)[0])

    data_recoded = pd.get_dummies(data, columns=cat_cols, dtype=int)
    data_recoded.columns = [re.sub(r'\W+', '_', col) for col in data_recoded.columns]

    return data_recoded

def data_summary(df: pd.DataFrame, target_column: str):
    df_without_target = df.drop(columns=[target_column])

    num_features = df_without_target.shape[1]
    categorical_features = df_without_target.select_dtypes(include=['object', 'category']).columns
    num_categorical_features = len(categorical_features)

    #if num_categorical_features == 0:
    #    int_columns = df_without_target.select_dtypes(include=['int']).columns.tolist()
    #    num_categorical_features = len(int_columns)

    #num_categorical_features_with_10_or_more_categories = sum(
    #    df_without_target[cat].nunique() > 10 for cat in categorical_features)

    target_classes = df[target_column].nunique()

    num_samples = df.shape[0]

    return {
        'num_features': num_features,
        'num_categorical_features': num_categorical_features,
        #'num_categorical_features_with_10_or_more_categories': num_categorical_features_with_10_or_more_categories,
        'num_target_classes': target_classes,
        'num_samples': num_samples
    }