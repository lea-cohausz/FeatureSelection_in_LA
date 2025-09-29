import pandas as pd
import re

def prepare_data(data):
    data = data.replace('?', None)
    data.rename(columns={'final_result': 'target'}, inplace=True)
    data = data.loc[data["target"] != "Withdrawn"]
    data["target"] = data["target"].replace({"Distinction": "Pass"})
    data = data.drop(["code_presentation", "id_student", "code_module_presentation", "code_module_student_presentation",
                      "module_presentation_length", "date_unregistration", "final_score"], axis=1)


    num_cols = data.select_dtypes(include=["int32", "int64", "float64"]).columns.tolist()
    bin_cols = [col for col in data.columns if data[col].nunique() == 2 and col not in num_cols]
    cat_cols = [col for col in data.columns if col not in num_cols and col not in bin_cols]

    # recoding of binary features
    data[bin_cols] = data[bin_cols].apply(lambda x: pd.factorize(x)[0])

    education_recode = [0, 1, 2, 3, 4]
    education = ["No Formal quals", "Lower Than A Level", "A Level or Equivalent", "HE Qualification", "Post Graduate Qualification"]
    data["highest_education"] = data["highest_education"].replace(education, education_recode)

    imd_recode = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    imd = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%',
       '60-70%', '70-80%', '80-90%', '90-100%']
    data["imd_band"] = data["imd_band"].replace(imd, imd_recode)

    age_recode = [0, 1, 2]
    age = ['0-35', '35-55', '55<=']
    data["age_band"] = data["age_band"].replace(age, age_recode)

    data["date_registration"] = pd.to_numeric(data["date_registration"])

    data_recoded = pd.get_dummies(data, columns=['code_module', 'region'], dtype=int)
    data_recoded.columns = [re.sub(r'\W+', '_', col) for col in data_recoded.columns]
    print(len(data_recoded))
    data_recoded = data_recoded.dropna(axis=0)
    print(len(data_recoded))
    return data_recoded