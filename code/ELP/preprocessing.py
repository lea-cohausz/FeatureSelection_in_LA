import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import re
from utils_elp import *

def preprocess_data(df):

    ### Set Specific preprocessing

    ################## Process "pri_"-features ############################################################

    ### Drop variables
    # Drop because train values are different from test values
    df = df.drop(["pri_languageother"],axis=1)

    # Drop because info is already encoded
    encoded_elsewhere_pri = ["pri_language","pri_meal", "pri_qualification", "pri_food_type", "pri_network_type",  "pri_funding_salary", "pri_money", "pri_clinic_travel", "pri_covid_awareness", "pri_covid_precautions", "pri_records", "pri_support_provider"]
    df = df.drop(encoded_elsewhere_pri,axis=1)

    # Drop constant except nan
    df = df.drop(["pri_toys",'pri_meal_prep'],axis=1)

    # Drop because of many missings and crosstab a) between missings and non-missings and b) different values revealed no important difference
    df = df.drop(["pri_dsd_conditional", "pri_dsd_conditional_other", "pri_dsd_unregistered", "pri_dsd_unregistered_other", "pri_email_network_forum", "pri_name_network_other", "pri_staff_changes_reasons", "pri_staff_changes_reasonsother", "pri_clinic_travelother"],axis=1)


    ### Special Encodings
    df['pri_staff_employed'].loc[df['pri_staff_employed']=='I have about the same number of staff employed compared to previous years'] = 0
    df['pri_staff_employed'].loc[df['pri_staff_employed']=='I have less than half of the number of staff employed compared to previous years'] = -1
    df['pri_staff_employed'].loc[df['pri_staff_employed']=='I have more staff employed compared to previous years'] = 1
    df['pri_staff_employed'].loc[df['pri_staff_employed']=='I have more than half of the number of staff employed compared to previous years'] = 2
    df['pri_staff_employed'] = df['pri_staff_employed'].astype(float)

    df['pri_covid_staff_salaries'].loc[df['pri_covid_staff_salaries']=='Yes, their salaries have remained the same'] = 1
    df['pri_covid_staff_salaries'].loc[df['pri_covid_staff_salaries']=='No, I have not been able to pay them at all'] = 0
    df['pri_covid_staff_salaries'].loc[df['pri_covid_staff_salaries']=='No, I had to reduce their salaries'] = -1
    df['pri_covid_staff_salaries'] = df['pri_covid_staff_salaries'].astype(float)

    df['pri_fees_paid_proportion'].loc[df['pri_fees_paid_proportion']=='None of the parents'] = 0
    df['pri_fees_paid_proportion'].loc[df['pri_fees_paid_proportion']=='Only a few parents'] = 1
    df['pri_fees_paid_proportion'].loc[df['pri_fees_paid_proportion']=='About half of the parents'] = 2
    df['pri_fees_paid_proportion'].loc[df['pri_fees_paid_proportion']=='Most parrents, but not all'] = 3
    df['pri_fees_paid_proportion'].loc[df['pri_fees_paid_proportion']=='All parents'] = 4
    df['pri_fees_paid_proportion'] = df['pri_fees_paid_proportion'].astype(float)


    for col in ['pri_food_parents_breakfast', 'pri_food_parents_morning', 'pri_food_parents_lunch', 'pri_food_parents_afternoon']:
        #print(col)
        df[col] = df[col].apply(trans_func_prifood)


    for col in ["pri_support_dsd", "pri_support_dbe", "pri_support_municipality", "pri_support_ngo"]:
        #print(col)
        df[col] = df[col].apply(trans_func_prisupport)


    for col in ["pri_attendance"]:
        #print(col)
        df[col] = df[col].apply(trans_func_priattendance)


    for col in ["pri_registered_programme","pri_registered_dsd", "pri_registered_partial"]:
        #print(col)
        df[col] = df[col].apply(trans_func_priregistered)

    for col in ["pri_education", "pra_education"]:
        #print(col)
        df[col] = df[col].apply(trans_func_pripraeducation)


    for col in ["pri_same_language"]:
        #print(col)
        df[col] = df[col].apply(trans_func_prilangauge)

    for col in ["pri_parents_contact", "pri_parents_activities"]:
        #print(col)
        df[col] = df[col].apply(trans_func_priparents)

    ### Define sets of variables
    bin_cols_pri = ['pri_funding_salary_4','pri_health_1','pri_aftercare','pri_clinic_travel_4','pri_clinic_travel_1','pri_covid_awareness_2','pri_qualification_97',
    'pri_funding_97','pri_covid_awareness_3','pri_registered_health','pri_funding_none','pri_funding_salary_2','pri_support_provider_97','pri_fees_free','pri_clinic_travel_2',
    'pri_language_8','pri_language_4','pri_language_6','pri_records_0','pri_money_2',
    'pri_money_1','pri_support_provider_4','pri_meal_3','pri_covid_awareness_4','pri_covid_precautions_6','pri_money_3','pri_refrigerator','pri_language_97',
    'pri_covid_fund_received','pri_records_4','pri_support','pri_covid_precautions_3','pri_language_5','pri_firstaid','pri_meal_2','pri_food_type_4',
    'pri_funding_subsidy','pri_language_2','pri_qualification_5','pri_qualification_3','pri_food_type_5','pri_qualification_6','pri_meals','pri_covid_precautions_5','pri_holidays',
    'pri_funding_6','pri_health_5','pri_funding_7','pri_covid_awareness_1','pri_covid_precautions_4','pri_health_4','pri_attendance_usual',
    'pri_support_provider_3','pri_food_guidance','pri_fees','pri_health_0','pri_language_9','pri_qualification_1','pri_qualification_0','pri_funding_donations','pri_network_type_1','pri_food_donation',
    'pri_qualification_4','pri_support_provider_1','pri_records_2','pri_qualification_2','pri_clinic_travel_97','pri_zoning','pri_covid_precautions_1',
    'pri_fees_exceptions','pri_covid_precautions_97','pri_covid_awareness_97','pri_funding_food','pri_money__1','pri_network_type_97','pri_registered_npo',
    'pri_library','pri_network','pri_school','pri_network_type_3','pri_funding_salary_0','pri_money_97','pri_clinic_travel_3','pri_funding_salary_97',
    'pri_support_provider_2','pri_covid_fund_applied','pri_meal_4','pri_precovid_attendance','pri_funding_4','pri_support_provider_5',
    'pri_health_2','pri_meal_1','pri_language_1','pri_records_3','pri_funding_salary_1','pri_garden','pri_records_5','pri_qualification_7',
    'pri_records_1','pri_language_7','pri_funding_5','pri_funding_1','pri_food_type_2','pri_mobile',
    'pri_covid_precautions_2','pri_registered_cipc','pri_health_3','pri_funding_2','pri_food_type_1','pri_food_type_3','pri_funding_salary_3','pri_funding_3','pri_language_3',
    'pri_health_97','pri_food_type_0','pri_language_11','pri_subsidy','pri_network_type_2','pri_transport','pri_language_10','pri_clinic_travel_5']

    
    num_cols_pri = ['pri_fees_amount_pv','pri_difficult_hold','pri_staff_employed','pri_clinic_time','pri_registered_partial','pri_difficult_learn',
                    'pri_amount_funding_fees','pri_support_dsd','pri_registered_dsd','pri_education','pri_expense_materials','pri_difficult_see','pri_food_parents_lunch',
                    'pri_fees_amount_4_6','pri_expense_food','pri_registered_programme','pri_expense_maintenance','pri_difficult_hear','pri_food_parents_breakfast','pri_attendance',
                    'pri_fees_paid_proportion','pri_time_open_hours','pri_fees_amount_2_3','pri_same_language','pri_food_parents_morning','pri_difficult_walk',
                    'pri_amount_funding_dsd','pri_parents_contact','pri_fees_amount','pri_expense_staff',
                    'pri_fees_amount_0_1','pri_support_ngo','pra_job','pri_capacity','pri_year','pri_children_4_6_years','pri_dsd_year','pri_days',
                    'pri_time_open_minutes','pri_difficult_communicate','pri_time_close_hours','pri_support_dbe','pri_expense_other','pri_food_parents_afternoon',
                    'pri_support_municipality','pri_covid_staff_salaries','pri_expense_rent','pri_expense_admin','pri_covid_staff_retrench','pri_time_close_minutes','pri_parents_activities']
    
    other_cols_pri = ["pri_support_providerother", "pri_fees_exceptions_other", "pri_expenseother", 'pri_fundingother', 'pri_qualificationother', 'pri_moneyother', 'pri_landother', 'pri_facilitiesother', 
                      "pri_covid_awareness_other", "pri_funding_salaryother", "pri_locationother", 'pri_covid_precautions_other', "pri_support_providerother","pri_food_donorother"]
    
    cat_cols_pri = ['pri_bank','pri_separate','pri_founderother','pri_kitchen','pri_support_frequency','pri_food_donor','pri_founder','pri_parents_frequency','pri_location','pri_internet_user', 'pri_facilities','pri_name_network_alliance','pri_name_network_ngo',
                      'pri_name_network_forum','pri_reason_register_year','pri_languages',
                      'pri_land']


    ### Create date and time columns
    # Date when facility opened
    df["pri_year"] = 2023 - df["pri_year"]

    # Date when interview was conducted (I think)
    df["pri_date"] =  df['pri_date'].str[:4].astype(float).astype("Int32")
    df["pri_date"] = 2023 - df["pri_date"].astype(float)


    # Opening and closing times
    df["pri_calc_time_close"] = pd.to_datetime(df["pri_calc_time_close"], format='%H:%M').dt.hour
    df["pri_calc_time_open"] = pd.to_datetime(df["pri_calc_time_open"], format='%H:%M').dt.hour
    df["pri_time_open"] = df["pri_calc_time_close"] - df["pri_calc_time_open"]

    ### Encode the "other"-features
    # I propse to simply encode features with 1 if people claimed the "other"-category. Altternatively, this would be a categorical variable with many NaNs.
    for i in other_cols_pri:
        df.loc[~df[i].isnull(), i] = 1
        df[i] = df[i].astype(float)

    ### Encode binary cols
    for col in bin_cols_pri:
        #print(col)
        le_ = LabelEncoder()
        df[col].loc[~df[col].isna()] = le_.fit_transform(df[col].loc[~df[col].isna()].astype(str))
        df[col] = df[col].astype(float)

    ### Encoded Elsewhere
    # Those variables that are open ended and were then also queried using individual categories were dropped. 
    # We use their category name to find all related variables. If any of the related variables is not missing, we encode the other related variables that are missing with 0.
    for i in encoded_elsewhere_pri:
        prefix_cols = df.filter(like=i).columns.tolist()
        check_not_nan = lambda row: not row[prefix_cols].isnull().all()
        mask = df.apply(check_not_nan, axis=1)
        df.loc[mask, prefix_cols] = df.loc[mask, prefix_cols].fillna(0)

    for var in cat_cols_pri:
        counts = df[var].value_counts()
        df[var] = df[var].replace(counts[counts < 50].index, "Other")

    # creation of dummy encoding
    pri_recoded = pd.get_dummies(df,
                                  columns=cat_cols_pri,
                                  dtype=int)
    pri_recoded.columns = [re.sub(r'\W+', '_', col) for col in pri_recoded.columns]
    df = pri_recoded

    ################## Process "pra_"-features ############################################################
    # Drop because info is already encoded
    encoded_elsewhere_pra = ["pra_plans", "pra_cohort", "pra_plan_4yrs", "pra_plan_5yrs", "pra_qualification", "pra_previous", "pra_ncf_trainer", "pra_training", "pra_groupings"]
    df = df.drop(encoded_elsewhere_pra,axis=1)

    ### Special encodings
    df['pra_engaged'].loc[df['pra_engaged']=='Seldom'] = 0
    df['pra_engaged'].loc[df['pra_engaged']=='Sometime'] = 1
    df['pra_engaged'].loc[df['pra_engaged']=='Often'] = 2
    df['pra_engaged'] = df['pra_engaged'].astype(float)

    for col in ["pra_free_play", "pra_free_play_outdoor"]:
        #print(col)
        df[col] = df[col].apply(trans_func_prafree)

    for col in ["pra_motivate_support", "pra_motivate_recognition", "pra_motivate_mentoring"]:
        #print(col)
        df[col] = df[col].apply(trans_func_pramotivate)


    for col in ["pra_job"]:
        #print(col)
        df[col] = df[col].apply(trans_func_prajob)

    for col in ["pra_salary"]:
        #print(col)
        df[col] = df[col].apply(trans_func_prasalary)

    # Date when interview was conducted (I think)
    df["pra_date"] =  df['pra_date'].str[:4].astype(float).astype("Int32")
    df["pra_date"] = 2023 - df["pra_date"]
    df["pra_date"] = df["pra_date"].astype(float)

    ### Define sets of variables
    bin_cols_pra = ['pra_training_1','pra_cohort_4','pra_training_6','pra_ncf_trainer_1','pra_special_referrals','pra_plan_5yrs_97','pra_qualification_2',
    'pra_clearance_police','pra_groupings_5','pra_breadwinner','pra_plan_4yrs_97','pra_groupings_2','pra_groupings_3','pra_plan_5yrs_2','pra_clearance_ncp',
    'pra_cohort_6','pra_plan_4yrs_2','pra_training_4','pra_ncf_trainer_2','pra_cohort_3','pra_plan_approved','pra_qualification_1',
    'pra_ncf_training','pra_qualification_3','pra_plan_4yrs_3','pra_cohort_5','pra_qualification_97','pra_plan_5yrs_4','pra_plan_5yrs_3',
    'pra_plans_0','pra_ind','pra_qualification_4','pra_ncf_trainer_5','pra_training_3','pra_special_training',
    'pra_online_training','pra_qualification_6','pra_paid','pra_qualification_5','pra_ncf_trainer_4','pra_cohort_1','pra_cohort_2','pra_plan_5yrs_5','pra_plan_5yrs_1','pra_groupings_1',
    'pra_plan_4yrs_1','pra_training_0','pra_plan_4yrs_4','pra_plans_1','pra_ncf_trainer_3',
    'pra_plan_ncf','pra_training_2','pra_plans_2','pra_cohort_0','pra_groupings_4','pra_learnership','pra_qualification_7','pra_ncf_trainer_97','pra_gender','pra_qualification_0',
    'pra_plans_3','pra_training_5']

    num_cols_pra = ['pra_education','pra_class_space','pra_hhsize','pra_class_size','pra_salary','pra_free_play_outdoor','pra_class_present',
                    'pra_motivate_mentoring','pra_measure_rectangle_width','pra_class_attendance_precovid',
                    'pra_job','pra_motivate_recognition','pra_experience','pra_motivate_support','pra_class_attendance',
                    'pra_measure_rectangle_length','pra_free_play','pra_engaged']
    
    other_cols_pra = ["pra_plan_4yrsother", "pra_plan_5yrsother", "pra_educationother", "pra_qualificationother", "pra_ncf_trainerother", "pra_class_size_large", "pra_class_space_small", "pra_class_space_large"]

    cat_cols_pra = ['pra_shape','pra_agency_understand','pra_agency_explore','pra_agency_learn','pra_agency_questions','pra_agency_play','pra_agency_choice','pra_agency_order', 'pra_class_language','pra_online_training_details','pra_language']


    ### Summarize values
    #Count how often the value child appears => new summary column
    subset_columns = ['pra_agency_choice', 'pra_agency_explore', 'pra_agency_questions', "pra_agency_understand", 'pra_agency_play', 'pra_agency_learn', "pra_agency_order"]

    def count_value(row, value):
        return row.eq(value).sum()

    df['count_child'] = df[subset_columns].apply(count_value, args=('Child',), axis=1)

    df['count_practioner'] = df[subset_columns].apply(count_value, args=('Practitioner',), axis=1)

    df['count_both'] = df[subset_columns].apply(count_value, args=('Both',), axis=1)

    #df = df.drop(subset_columns,axis=1)
    #test = test.drop(subset_columns,axis=1)

    ### Encode the "other"-features: I propse to simply encode features with 1 if people claimed the "other"-category. Altternatively, this would be a categorical variable with many NaNs.
    for i in other_cols_pra:
        df.loc[~df[i].isnull(), i] = 1
        df[i] = df[i].astype(float)

     ### Encode binary cols
    for col in bin_cols_pra:
        #print(col)
        le_ = LabelEncoder()
        df[col].loc[~df[col].isna()] = le_.fit_transform(df[col].loc[~df[col].isna()].astype(str))
        df[col] = df[col].astype(float)

    ### Encoded Elsewhere
    # Those variables that are open ended and were then also queried using individual categories were dropped. 
    # We use their category name to find all related variables. If any of the related variables is not missing, we encode the other related variables that are missing with 0.
    for i in encoded_elsewhere_pra:
        prefix_cols = df.filter(like=i).columns.tolist()
        check_not_nan = lambda row: not row[prefix_cols].isnull().all()
        mask = df.apply(check_not_nan, axis=1)
        df.loc[mask, prefix_cols] = df.loc[mask, prefix_cols].fillna(0)

    for var in cat_cols_pra:
        counts = df[var].value_counts()
        df[var] = df[var].replace(counts[counts < 50].index, "Other")

    # creation of dummy encoding
    pri_recoded = pd.get_dummies(df,
                                 columns=cat_cols_pra,
                                 dtype=int)
    pri_recoded.columns = [re.sub(r'\W+', '_', col) for col in pri_recoded.columns]
    df = pri_recoded


    ################## Process "pqa_"-features ############################################################
    pqa_binary = ["pqa_class_age_1","pqa_class_age_2","pqa_class_age_3","pqa_class_age_4","pqa_class_age_5",
                  "pqa_class_age_6"]
    pqa_ohe = ["pqa_class"]
    pqa_high_card = ["pqa_date"]


    for col in ['pqa_relationships_peers', 'pqa_relationships_staff', 'pqa_relationships_acknowledge',
                'pqa_relationships_discipline',
                'pqa_environment_areas', 'pqa_environment_variety', 'pqa_environment_appropriate',
                'pqa_environment_accessible', 'pqa_environment_open', 'pqa_environment_outdoor',
                'pqa_assessment_observation', 'pqa_assessment_systematic',
                'pqa_curriculum_ncf', 'pqa_curriculum_plan', 'pqa_curriculum_balance', 'pqa_curriculum_numeracy',
                'pqa_curriculum_literacy',
                'pqa_teaching_choice', 'pqa_teaching_engagement', 'pqa_teaching_participation',
                'pqa_teaching_questions',
                'pqa_teaching_support',
                ]:
        df[col] = df[col].apply(trans_func_pqa)

    # Drop features where information is already entailed in others or simply uninformative
    drop_cols = ["pqa_class_age"]
    df = df.drop(drop_cols, axis=1)


    ################## Process "obs_"-features ############################################################
    obs_binary = ["obs_access", "obs_access_disability_0","obs_access_disability_1","obs_access_disability_2",
                  "obs_access_disability_3","obs_access_disability_4","obs_access_disability_5",
                  "obs_access_disability_6","obs_accessible", "obs_area_0", "obs_area_1", "obs_area_2", "obs_area_3",
                  "obs_area_4", "obs_area_5","obs_area_6", "obs_area_7","obs_area_8", "obs_books", "obs_books_age",
                  "obs_electricity_working", "obs_equipment__1", "obs_equipment_0","obs_equipment_1","obs_equipment_2",
                  "obs_equipment_3","obs_equipment_4","obs_equipment_5","obs_fence","obs_fencing_play_area",
                  "obs_firstaid","obs_gate", "obs_handwashing_0", "obs_handwashing_1","obs_handwashing_2",
                  "obs_handwashing_3","obs_handwashing_97","obs_handwashing_friendly", "obs_hazard_0", "obs_hazard_1",
                  "obs_hazard_2","obs_hazard_3","obs_hazard_4","obs_hazard_5","obs_hazard_6","obs_hazard_7",
                  "obs_hazard_8", "obs_hazard_97", "obs_materials_0","obs_materials_1","obs_materials_10",
                  "obs_materials_11","obs_materials_12","obs_materials_13","obs_materials_14","obs_materials_15",
                  "obs_materials_16","obs_materials_17","obs_materials_18","obs_materials_19","obs_materials_2",
                  "obs_materials_20","obs_materials_3","obs_materials_4","obs_materials_5","obs_materials_6",
                  "obs_materials_7","obs_materials_8","obs_materials_9","obs_materials_97","obs_menu_compliance",
                  "obs_menu_display","obs_menu_same","obs_outdoor","obs_potable", "obs_safety_0","obs_safety_1",
                  "obs_safety_10","obs_safety_2","obs_safety_3","obs_safety_4","obs_safety_5","obs_safety_6",
                  "obs_safety_7","obs_safety_8","obs_safety_9","obs_shared","obs_space", "obs_toilet_0","obs_toilet_1",
                  "obs_toilet_2","obs_toilet_3","obs_toilet_4","obs_toilet_5","obs_toilet_6","obs_toilet_7",
                  "obs_toilet_8", "obs_toilet_clean","obs_toilet_paper","obs_toilets_children","obs_toilets_gender",
                  "obs_water_running"
                  ]
    obs_ohe = ["obs_building", "obs_material_display", "obs_water"]
    obs_high_card = ["obs_date"]

    # Ordinal encoding of equipment condition feature
    df['obs_condition_equipment'].loc[df['obs_condition_equipment'] == 'Bad (Mostly broken and unused)'] = 0
    df['obs_condition_equipment'].loc[df['obs_condition_equipment'] == 'Okay (some in working condition)'] = 1
    df['obs_condition_equipment'].loc[df['obs_condition_equipment'] == 'Fine (mostly in working condition)'] = 2
    df['obs_condition_equipment'].loc[df['obs_condition_equipment'] == 'Very good'] = 3
    df['obs_condition_equipment'] = df['obs_condition_equipment'].astype(float)

    # Engineer stacking games available?
    df["obs_materials_stackinggames"] = (df["obs_materialsother"] == "STACKING GAMES") * 1

    # Engineering of cooking features - merge census and non census
    ohe_cooking = OneHotEncoder()
    ohe_cooking.fit(df[["obs_cooking"]])

    df_ohe_cooking = pd.DataFrame(ohe_cooking.fit_transform(df[["obs_cooking"]]).toarray(),
                                  columns=ohe_cooking.get_feature_names_out(["obs_cooking"]), index=df.index)

    df_ohe_cooking["obs_cooking_Gas"].loc[df["obs_cooking_census"] == "Gas"] = 1.
    df_ohe_cooking["obs_cooking_Electricity from mains"].loc[df["obs_cooking_census"] == "Electricity"] = 1.
    df_ohe_cooking["obs_cooking_Coal or wood"] = (df["obs_cooking_census"] == "Coal or wood") * 1
    df_ohe_cooking["obs_cooking_Paraffin"].loc[df["obs_cooking_census"] == "Paraffin"] = 1.
    df_ohe_cooking.loc[df.obs_cooking_census == "Other", "obs_cooking_Other"] = 1.
    df_ohe_cooking.loc[df.obs_cooking_census == "Other", "obs_cooking_nan"] = 0.
    df_ohe_cooking.loc[
        np.logical_and(df.obs_cooking_census == "None", df.obs_cooking != "None"), "obs_cooking_None"] = 1.
    df_ohe_cooking.loc[
        np.logical_and(df.obs_cooking_census == "None", df.obs_cooking != "None"), "obs_cooking_nan"] = 0.
    df_ohe_cooking["obs_cooking_nan"] = (df_ohe_cooking.drop("obs_cooking_nan", axis=1).sum(axis=1) == 0) * 1.
    df[df_ohe_cooking.columns] = df_ohe_cooking


    # Engineering of heating features - merge census and non census
    ohe_heating = OneHotEncoder()
    ohe_heating.fit(df[["obs_heating"]])

    df_ohe_heating = pd.DataFrame(ohe_heating.fit_transform(df[["obs_heating"]]).toarray(),
                                  columns=ohe_heating.get_feature_names_out(["obs_heating"]), index=df.index)
    df_ohe_heating = df_ohe_heating.rename({"obs_heating_Electricity from mains": "obs_heating_Electricity"}, axis=1)
    df_ohe_heating.loc[df_ohe_heating["obs_heating_Electricity from generator"] == 1, "obs_heating_Electricity"] = 1
    df_ohe_heating = df_ohe_heating.drop("obs_heating_Electricity from generator", axis=1)

    df_ohe_heating["obs_heating_Gas"].loc[df["obs_heating_census"] == "Gas"] = 1.
    df_ohe_heating["obs_heating_Electricity"].loc[df["obs_heating_census"] == "Electricity"] = 1.
    df_ohe_heating["obs_heating_Coal or wood"] = (df["obs_heating_census"] == "Coal or wood") * 1
    df_ohe_heating["obs_heating_Paraffin"].loc[df["obs_heating_census"] == "Paraffin"] = 1.
    df_ohe_heating["obs_heating_Solar"].loc[df["obs_heating_census"] == "Solar"] = 1.
    df_ohe_heating.loc[df.obs_heating_census == "Other", "obs_heating_Other"] = 1.
    df_ohe_heating.loc[df.obs_heating_census == "Other", "obs_heating_nan"] = 0.
    df_ohe_heating.loc[
        np.logical_and(df.obs_heating_census == "None", df.obs_heating != "None"), "obs_heating_None"] = 1.
    df_ohe_heating.loc[
        np.logical_and(df.obs_heating_census == "None", df.obs_heating != "None"), "obs_heating_nan"] = 0.
    df_ohe_heating["obs_heating_nan"] = (df_ohe_heating.drop("obs_heating_nan", axis=1).sum(axis=1) == 0) * 1.
    df[df_ohe_heating.columns] = df_ohe_heating

    # Engineering of lighting features - merge census and non census
    ohe_lighting = OneHotEncoder()
    ohe_lighting.fit(df[["obs_lighting"]])

    df_ohe_lighting = pd.DataFrame(ohe_lighting.fit_transform(df[["obs_lighting"]]).toarray(),
                                   columns=ohe_lighting.get_feature_names_out(["obs_lighting"]), index=df.index)
    df_ohe_lighting = df_ohe_lighting.rename({"obs_lighting_Electricity from mains": "obs_lighting_Electricity"},
                                             axis=1)
    df_ohe_lighting.loc[df_ohe_lighting["obs_lighting_Electricity from generator"] == 1, "obs_lighting_Electricity"] = 1
    df_ohe_lighting = df_ohe_lighting.drop("obs_lighting_Electricity from generator", axis=1)

    df_ohe_lighting["obs_lighting_Gas"].loc[df["obs_lighting_census"] == "Gas"] = 1.
    df_ohe_lighting["obs_lighting_Electricity"].loc[df["obs_lighting_census"] == "Electricity"] = 1.
    df_ohe_lighting["obs_lighting_Coal or wood"] = (df["obs_lighting_census"] == "Coal or wood") * 1
    df_ohe_lighting["obs_lighting_Paraffin"].loc[df["obs_lighting_census"] == "Paraffin"] = 1.
    df_ohe_lighting["obs_lighting_Solar"].loc[df["obs_lighting_census"] == "Solar"] = 1.
    df_ohe_lighting["obs_lighting_Candles"].loc[df["obs_lighting_census"] == "Candles"] = 1.
    df_ohe_lighting.loc[df.obs_lighting_census == "Other", "obs_lighting_Other"] = 1.
    df_ohe_lighting.loc[df.obs_lighting_census == "Other", "obs_lighting_nan"] = 0.
    df_ohe_lighting.loc[
        np.logical_and(df.obs_lighting_census == "None", df.obs_lighting != "None"), "obs_lighting_None"] = 1.
    df_ohe_lighting.loc[
        np.logical_and(df.obs_lighting_census == "None", df.obs_lighting != "None"), "obs_lighting_nan"] = 0.
    df_ohe_lighting["obs_lighting_nan"] = (df_ohe_lighting.drop("obs_lighting_nan", axis=1).sum(axis=1) == 0) * 1.
    df[df_ohe_lighting.columns] = df_ohe_lighting

    # Engineer water features
    df["obs_water_bought"] = (df["obs_waterother"].apply(
        lambda x: ("BUY" in x or "BOUGHT" in x) if x == x else False)) * 1
    df["obs_water_tank"] = (df["obs_waterother"].apply(
        lambda x: ("TANK" in x or "TRUCK" in x) if x == x else False)) * 1

    # Drop features where information is already entailed in others or simply uninformative
    drop_cols = ["obs_access_disability", "obs_materials", "obs_materialsother", "obs_area", "obs_equipment","obs_handwashing",
                 "obs_handwashingother", "obs_hazard", "obs_safety", "obs_toilet", "obs_toilet_97",
                 "obs_cooking", "obs_cooking_1","obs_cooking_2","obs_cooking_3","obs_cooking_4",
                 "obs_cooking_5","obs_cooking_6","obs_cooking_census",
                 "obs_heating","obs_heating_1", "obs_heating_2", "obs_heating_3","obs_heating_4","obs_heating_5",
                 "obs_heating_6", "obs_heating_7", "obs_heating_census",
                 "obs_lighting", "obs_lighting_1", "obs_lighting_2", "obs_lighting_3", "obs_lighting_4",
                 "obs_lighting_5", "obs_lighting_6", "obs_lighting_8", "obs_lighting_census",
                 "obs_water_running_none","obs_waterother"
                 ]
    df = df.drop(drop_cols, axis=1)



    ################## Process "count_"-features ############################################################
    # All are numeric!


    ################## Process "teacher_"-features ############################################################
    teacher_binary = ["teacher_social_met","teacher_emotional_met","teacher_selfcare_met"]
    teacher_ohe = []
    teacher_high_card = []

    for col in ["teacher_social_assistance", "teacher_social_nonaggressive", "teacher_social_cooperate",
                "teacher_social_ideas", "teacher_social_initiative", "teacher_social_peers"]:
        df[col] = df[col].apply(trans_func_teacher_social)


    for col in ['teacher_emotional_understand', 'teacher_emotional_appropriate', 'teacher_emotional_independent',
                'teacher_emotional_adjust', 'teacher_emotional_confidence', 'teacher_emotional_selfstarter']:
        df[col] = df[col].apply(trans_func_teacher_emotional)


    ################## Process "child_"-features ############################################################
    # Drop child_id (unique sample identifier)
    df = df.drop(["child_id"],axis=1)


    # Categorize features
    child_binary = ["child_gender"]
    child_ohe = ["child_grant", "child_years_in_programme", "child_stunted"]
    child_high_card = ["child_enrolment_date", "child_date"]

    # Encode child_age_group as binary
    df["child_age_group"].loc[df["child_age_group"]=="50-59 months"] = 0
    df["child_age_group"].loc[df["child_age_group"]=="Younger than 50 months"] = 0
    df["child_age_group"].loc[df["child_age_group"]=="60-69 months"] = 1
    df["child_age_group"].loc[df["child_age_group"]=="70 Months or older"] = 1
    df["child_age_group"] = df["child_age_group"].astype(float)

    # Encode child_observe features as Ordinal
    for col in ['child_observe_attentive', 'child_observe_concentrated', 'child_observe_diligent',
                'child_observe_interested']:
        df[col] = df[col].apply(trans_func_child_observe)

    df[['child_observe_attentive', 'child_observe_concentrated', 'child_observe_diligent',
        'child_observe_interested']] = df[
        ['child_observe_attentive', 'child_observe_concentrated', 'child_observe_diligent',
         'child_observe_interested']].astype(float)


    # Engineer features
    df["child_multilingual"] = df["child_languages"].apply(lambda x: x.count("+") if x == x else np.nan)

    # Drop features where information is already entailed in others or simply uninformative
    drop_cols = ["child_dob", "child_attends", "child_languages"]
    df = df.drop(drop_cols, axis=1)


    ################## Process other features ############################################################
    # Encode binary features
    other_binary = ["professionals_practitioners", "hle_ind", "sef_ind", "pre_covid", "urban", "gps_ind", "language_match",
                "quintile_used", "census", "elp_ind"]

    # OHE Encode low cardinality categorical features
    other_ohe = ["healthother", "plan", "grade_r", "data_year", "quintile", "ses_proxy", "ses_cat",
                "sanitation_learners", "sanitation_educators", "phase_natemis", "facility_type", "prov_best", "language_assessment", "id_team", "mn_best", "id_prov", "dc_best", "id_ward",
                "id_enumerator", "id_facility", "language_assessment_w2", "ward_best", "language_child"]


    # Ordinal encode certificate features
    for col in ['certificate_registration_partial', 'certificate_registration_program', 'certificate_registration_npo',
                'certificate_register']:
        df[col] = df[col].apply(trans_func_certificate)

    # Drop features where information is already entailed in others or simply uninformative
    drop_cols = ["practitioner", "health", "hle_ecd_other", "id_dc_best", "id_mn_best", "positionother",
                 "positionotherreason"]
    df = df.drop(drop_cols, axis=1)

    # Drop features for now which might be useful if processed carefully
    drop_cols = ["gps", "other_practitioner"]
    df = df.drop(drop_cols, axis=1)


    ################## Do encodings ############################################################
    # Encode Binary features
    for col in pqa_binary + obs_binary + teacher_binary + child_binary + other_binary:
        # print(col)
        le_ = LabelEncoder()
        df[col].loc[~df[col].isna()] = le_.fit_transform(df[col].loc[~df[col].isna()].astype(str))
        df[col] = df[col].astype(float)

    for var in other_ohe:
        counts = df[var].value_counts()
        df[var] = df[var].replace(counts[counts < 50].index, "Other")

    # creation of dummy encoding
    pri_recoded = pd.get_dummies(df,
                                 columns=other_ohe,
                                 dtype=int)
    pri_recoded.columns = [re.sub(r'\W+', '_', col) for col in pri_recoded.columns]
    df = pri_recoded

    ################################## Throw out and rename ##################################
    threshold = len(df) * 0.8  # keep columns with at least 80% non-NA values
    df = df.dropna(axis=1, thresh=threshold)

    df["child_years_in_programme"] = df["child_years_in_programme"].replace({"Do Not Know": 1, "1st year in the programme": 1, "2nd year in programme": 2, "3rd year in programme": 3})
    df["child_stunted"] = df["child_stunted"].replace(
        {"Normal": 0, "Moderately stunted": 1, "Severely stunted": 1})
    df.rename(columns={"child_gender": 'gender'}, inplace=True)

    threshold = 0.2
    df = df[df.isnull().mean(axis=1) <= threshold]

    for col in df.columns:
        if df[col].isnull().any():
            mode_val = df[col].mode(dropna=True)
            if not mode_val.empty:
                df[col].fillna(mode_val[0], inplace=True)

    conditions = [
        (df['target'] >= 0) & (df['target'] <= 30),
        (df['target'] >= 31) & (df['target'] <= 60),
        (df['target'] > 60)
    ]

    choices = [0, 1, 2]

    df['target_recode'] = np.select(conditions, choices, default=np.nan)
    df.drop("target", axis=1, inplace=True)
    df.rename(columns={"target_recode": 'target'}, inplace=True)
    df = df.dropna(subset=['target'])

    return df


