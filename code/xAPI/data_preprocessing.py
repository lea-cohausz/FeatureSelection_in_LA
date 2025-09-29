import pandas as pd
from sklearn.preprocessing import StandardScaler
import re

def preprocessing_xapi(data):
    # recode target
    class_recode = [0, 1, 2]
    status = ["L", "M", "H"]
    data["Class"] = data["Class"].replace(status, class_recode)
    data.rename(columns={'Class': 'target'}, inplace=True)

    # different kinds of features requiring different pre-processing
    num_cols = data.select_dtypes(include=["int32", "int64", "float64"]).columns.tolist()
    bin_cols = [col for col in data.columns if data[col].nunique() == 2 and col not in num_cols]
    cat_cols = [col for col in data.columns if col not in num_cols and col not in bin_cols]

    # recoding of binary features
    data[bin_cols] = data[bin_cols].apply(lambda x: pd.factorize(x)[0])

    # recoding of ordinal features
    data["StageID"] = data["StageID"].replace({"lowerlevel": 0, "MiddleSchool": 1, "HighSchool": 2})
    data["SectionID"] = data["SectionID"].replace({"A": 0, "B": 1, "C": 2})

    # recoding of rare occurrences in categorical features to "Other"
    var_others = ["NationalITy", "PlaceofBirth", "GradeID"]
    for var in var_others:
        counts = data[var].value_counts()
        data[var] = data[var].replace(counts[counts < 20].index, "Other")

    # creation of dummy encoding
    data_recoded = pd.get_dummies(data, columns=['NationalITy', 'PlaceofBirth', 'GradeID', 'Topic'], dtype=int)
    data_recoded.columns = [re.sub(r'\W+', '_', col) for col in data_recoded.columns]

    # standardization of numerical features
    #data_recoded[['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']] = StandardScaler().fit_transform(data_recoded[['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']])

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