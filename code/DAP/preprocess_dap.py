import pandas as pd
import re

def prepare_dap(data):
    data = data.drop("Unnamed: 9", axis=1)
    identifiers = ["COD_S11", "Cod_SPro"]
    alternative_targets = ["CR_PRO", "QR_PRO", "CC_PRO", "WC_PRO", "FEP_PRO", "ENG_PRO", "G_SC", "PERCENTILE",
                           "2ND_DECILE", ]
    data = data.drop(identifiers + alternative_targets, axis=1)
    data.rename(columns={"QUARTILE": 'target'}, inplace=True)
    data.rename(columns={"GENDER": 'gender'}, inplace=True)
    data["target"] = data["target"].replace({1: 0, 2: 1, 3: 2, 4: 3})

    # different kinds of features requiring different pre-processing
    num_cols = data.select_dtypes(include=["int32", "int64", "float64"]).columns.tolist()
    bin_cols = [col for col in data.columns if data[col].nunique() == 2 and col not in num_cols]
    cat_cols = [col for col in data.columns if col not in num_cols and col not in bin_cols]

    # recoding of binary features
    data[bin_cols] = data[bin_cols].apply(lambda x: pd.factorize(x)[0])

    data["EDU_FATHER"] = data["EDU_FATHER"].replace( {"0": 0, "Not sure": 0, "Ninguno": 0, "Incomplete primary ": 1, "Complete primary ": 2, "Incomplete Secundary": 3,
         "Complete Secundary": 4, "Incomplete technical or technological": 5, "Complete technique or technology": 6, "Incomplete Professional Education": 7, "Complete professional education": 8, "Postgraduate education": 9})
    data["EDU_MOTHER"] = data["EDU_MOTHER"].replace(
        {"0": 0, "Not sure": 0, "Ninguno": 0, "Incomplete primary ": 1, "Complete primary ": 2, "Incomplete Secundary": 3,
         "Complete Secundary": 4, "Incomplete technical or technological": 5, "Complete technique or technology": 6,
         "Incomplete Professional Education": 7, "Complete professional education": 8, "Postgraduate education": 9})
    data["STRATUM"] = data["STRATUM"].replace(
        {"0": 0, "Stratum 1": 0, "Stratum 2": 1, "Stratum 3": 2, "Stratum 4": 3, "Stratum 5": 4, "Stratum 6": 5})
    data["SISBEN"] = data["SISBEN"].replace(
        {"0": 0, "Esta clasificada en otro Level del SISBEN": 0, "It is not classified by the SISBEN": 0, "Level 1": 1, "Level 2": 2, "Level 3": 3})
    data["PEOPLE_HOUSE"] = data["PEOPLE_HOUSE"].replace(
        {"0": 0, "Once": 0, "One": 0, "Two": 1,
         "Three": 3, "Four": 4, "Five": 5, "Six": 6, "Seven": 7, "Eight": 8, "Nueve": 9, "Ten": 10, "Twelve or more": 11})
    data["REVENUE"] = data["REVENUE"].replace(
        {"0": 0, "less than 1 LMMW": 0, "Between 1 and less than 2 LMMW": 1,
         "Between 2 and less than 3 LMMW": 3, "Between 3 and less than 5 LMMW": 4, "Between 5 and less than 7 LMMW": 5, "Between 7 and less than 10 LMMW": 6, "10 or more LMMW": 7})
    data["JOB"] = data["JOB"].replace(
        {"0": 0, "No": 0, "Yes, less than 20 hours per week": 1, "Yes, 20 hours or more per week": 2})
    data["SCHOOL_TYPE"] = data["SCHOOL_TYPE"].replace(
        {"Not apply": 0, "TECHNICAL": 0, "TECHNICAL/ACADEMIC": 1, "ACADEMIC": 2})

    var_others = ["OCC_FATHER", "OCC_MOTHER", "UNIVERSITY", "ACADEMIC_PROGRAM"]
    for var in var_others:
        counts = data[var].value_counts()
        data[var] = data[var].replace(counts[counts < 20].index, "Other")
    counts = data["SCHOOL_NAME"].value_counts()
    data["SCHOOL_NAME"] = data["SCHOOL_NAME"].replace(counts[counts < 50].index, "Other")

    # creation of dummy encoding
    data_recoded = pd.get_dummies(data, columns=["OCC_FATHER", "OCC_MOTHER", "SCHOOL_NAME", "UNIVERSITY", "ACADEMIC_PROGRAM"], dtype=int)
    data_recoded.columns = [re.sub(r'\W+', '_', col) for col in data_recoded.columns]

    return data_recoded
