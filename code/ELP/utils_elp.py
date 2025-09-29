import numpy as np
import pandas as pd



import xgboost as xgb

from sklearn.metrics import mean_squared_error as mse

from sklearn.model_selection import KFold


def trans_func_prifood(x): 
    if x!=x: 
        return np.nan
    elif "None" in x:
        return 0
    elif "Some" in x:
        return 1
    elif "All" in x:
        return 2
    else:
        raise ValueError(f"Invalid value encountered: {x}")
    
def trans_func_prisupport(x): 
    if x!=x: 
        return np.nan
    elif "Never" in x:
        return 0
    elif "Once" in x:
        return 1
    elif "Twice" in x:
        return 2
    elif "Three" in x:
        return 3
    elif "More than three times" in x:
        return 4
    else:
        raise ValueError(f"Invalid value encountered: {x}")
    
def trans_func_priattendance(x): 
    if x!=x: 
        return np.nan
    elif "Once" in x:
        return 1
    elif "Two" in x:
        return 2
    elif "Three" in x:
        return 3
    elif "Four" in x:
        return 4
    elif "Five" in x:
        return 5
    else:
        raise ValueError(f"Invalid value encountered: {x}")
    
def trans_func_priregistered(x): 
    if x!=x: 
        return np.nan
    elif 'Not registered' in x or 'Lapsed registration' in x:
        return 0
    elif 'In process' in x:
        return 1
    elif 'Conditionally registered' in x:
        return 2
    elif 'Fully registered' in x:
        return 3
    else:
        raise ValueError(f"Invalid value encountered: {x}")
    
def trans_func_pripraeducation(x): 
    if x!=x: 
        return np.nan
    elif 'Other' in x or 'Below Grade 12/matric' in x:
        return 0
    elif 'Matric/National Senior Certificate' in x:
        return 1
    elif 'Certificate' in x:
        return 2
    elif 'Diploma' in x:
        return 3
    elif 'Undergraduate Degree' in x:
        return 4
    elif 'Postgraduate degree' in x:
        return 5
    else:
        raise ValueError(f"Invalid value encountered: {x}")
    
def trans_func_prilangauge(x): 
    if x!=x: 
        return np.nan
    elif 'None of the children speak this language at home' in x:
        return 0
    elif 'Less than half of the children speak this language at home' in x:
        return 1
    elif 'More than half of the children speak this language at home' in x:
        return 2
    elif 'All children speak this language at home' in x:
        return 3
    else:
        raise ValueError(f"Invalid value encountered: {x}")
    
def trans_func_priparents(x): 
    if x!=x: 
        return np.nan
    elif 'No, no-one has asked me this year.' in x:
        return 0
    elif 'Other' in x:
        return 0
    elif 'Yes, but only one or two have asked me this year.' in x:
        return 1
    elif 'Yes, some parents have asked me this year.' in x:
        return 2
    elif 'Yes, most parents have asked me this year.' in x:
        return 3
    else:
        raise ValueError(f"Invalid value encountered: {x}")
    

def trans_func_prafree(x): 
    if x!=x: 
        return np.nan
    elif "None" in x:
        return 0
    elif "30 minutes or less" in x:
        return 1
    elif "Up to 1 hour" in x:
        return 2
    elif "Up to 2 hours" in x:
        return 3
    elif "Up to 3 hours" in x:
        return 4
    elif "More than 3 hours" in x:
        return 5
    else:
        raise ValueError(f"Invalid value encountered: {x}")


def trans_func_pramotivate(x): 
    if x!=x: 
        return np.nan
    elif "Disagree strongly" in x:
        return 0
    elif "Disagree a little" in x:
        return 1
    elif "Agree a little" in x:
        return 2
    elif "Agree strongly" in x:
        return 3
    else:
        raise ValueError(f"Invalid value encountered: {x}")
    
def trans_func_prajob(x): 
    if x!=x: 
        return np.nan
    elif 'Assistant practitioner' in x:
        return 0
    elif 'Practitioner/teacher' in x:
        return 1
    elif 'Supervisor' in x:
        return 2
    elif 'Principal/Matron' in x:
        return 3
    else:
        raise ValueError(f"Invalid value encountered: {x}")
    
def trans_func_prasalary(x): 
    if x!=x: 
        return np.nan
    elif 'R0' in x:
        return 0
    elif 'Less than R500 per month' in x:
        return 1
    elif 'R500 – R749' in x:
        return 2
    elif 'R750 – R999' in x:
        return 3
    elif 'R1000 – R1249' in x:
        return 4
    elif 'R1250 – R1499' in x:
        return 5
    elif 'R1500 – R1999' in x:
        return 6
    elif 'R2000 – R2499' in x:
        return 7
    elif 'R2500 – R2999' in x:
        return 8
    elif 'R3000 – R3499' in x:
        return 9
    elif 'R3500 – R3999' in x:
        return 10
    elif 'R4000 – R4449' in x:
        return 11
    elif 'R 5000 – R5999' in x:
        return 12
    elif 'More than R6000' in x:
        return 13
    else:
        raise ValueError(f"Invalid value encountered: {x}")

def trans_func_child_observe(x):
    if x!=x:
        return np.nan
    elif "Almost never" in x:
        return 0
    elif "Sometimes" in x:
        return 1
    elif "Often" in x:
        return 2
    elif "Almost always" in x:
        return 3
    else:
        raise ValueError(f"Invalid value encountered: {x}")


# Transformation function for pqa scale
def trans_func_pqa(x):
    if x != x:
        return np.nan
    elif "<b>Inadequate</b>" in x or "<b>Inadequate:</b>" in x:
        return 0
    elif "<b>Basic</b>" in x or "<b>Basic:</b>" in x:
        return 1
    elif "<b>Good</b>" in x or "<b>Good:</b>" in x:
        return 2
    else:
        raise ValueError(f"Invalid value encountered: {x}")


# Transformation function for scale ['None of the time', 'All of the time', 'A little of the time', 'Most of the time']
trans_func_teacher_social = lambda x: \
{'None of the time': 0, 'A little of the time': 1, 'Most of the time': 2, 'All of the time': 3, "nan": np.nan}[
    x] if not x != x else np.nan


# Transformation function for scale including 'Often True', 'Sometimes True', 'Not True'
def trans_func_teacher_emotional(x):
    if x!=x:
        return np.nan
    elif "Not True" in x:
        return 0
    elif "Sometimes True" in x:
        return 1
    elif "Often True" in x:
        return 2
    else:
        raise ValueError(f"Invalid value encountered: {x}")



# Transformation function for scale including 'Yes' 'No, but it exists' 'Does not exist'
def trans_func_certificate(x):
    if x!=x:
        return np.nan
    elif "Does not exist" in x:
        return 0
    elif "No, but it exists" in x:
        return 1
    elif "Yes" in x:
        return 2
    else:
        raise ValueError(f"Invalid value encountered: {x}")

def enrich_feature_description():
    df = pd.read_csv('data/Train.csv', low_memory=False)
    test = pd.read_csv('data/Test.csv', low_memory=False)
    info = pd.read_csv("data/VariableDescription.csv")
    info.index = info["Variable Name"]
    # Remove duplicates
    info = info.loc[[i not in [np.where(info.index == i)[0][1] for i in np.unique(info.index, return_counts=True)[0][
        np.unique(info.index, return_counts=True)[1] != 1]] for i in range(info.shape[0])]]

    test["target"] = np.nan
    all_data = pd.concat([df.loc[:,info.index], test.loc[:,info.index]], axis=0)

    # Check whether feature is on facility level - Yes if output is zero
    fac_group_unique = all_data.groupby("id_facility").nunique()
    fac_group_count = all_data.groupby("id_facility").count()
    unique_for_facility = [sum(fac_group_unique.loc[fac_group_count[col] != 0, col] != 1) if col != "id_facility" else 0
                           for col in info.index]
    info["Facility Level"] = np.array(unique_for_facility) == 0

    # Check whether feature is on assessment level - Yes if output is zero
    fac_group_unique = all_data.groupby("language_assessment_w2").nunique()
    fac_group_count = all_data.groupby("language_assessment_w2").count()
    unique_for_assessment = [sum(fac_group_unique.loc[fac_group_count[col] != 0, col] != 1) if col != "language_assessment_w2" else 0
                           for col in info.index]
    info["Assessment Level"] = np.array(unique_for_assessment) == 0

    info["Train NA %"] = df.loc[:,info.index].isna().sum()/df.shape[0]
    info["Test NA %"] = test.loc[:,info.index].isna().sum()/test.shape[0]
    info["N unique"] = all_data.nunique()
    info["Mode %"] = all_data.apply(lambda x: max(np.unique(x[~x.isna()].astype(str),return_counts=True)[1])/(all_data.shape[0]-x.isna().sum()), axis=0)


    return info

