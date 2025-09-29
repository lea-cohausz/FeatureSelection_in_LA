import pandas as pd
import numpy as np

# Preparation largely taken from: https://www.kaggle.com/code/irenepal/open-university-learning-analytics-data-merging

# prep demo
demo = pd.read_csv('open+university+learning+analytics+dataset/studentInfo.csv')

demo["code_module_presentation"]=demo[['code_module','code_presentation']].agg('-'.join,axis=1)

demo = demo.astype({'id_student':'string'})
demo["code_module_student_presentation"]=demo[['code_module_presentation','id_student']].agg('-'.join,axis=1)

demo['cohort'] = demo['code_presentation'].str[-1:]

uniqueValues = demo['imd_band'].unique()
demo['imd_band'] = demo['imd_band'].replace(['10-20'], '10-20%')
uniqueValues = demo['imd_band'].unique()

# prep courses
courses = pd.read_csv('open+university+learning+analytics+dataset/courses.csv')
courses["code_module_presentation"]=courses[['code_module','code_presentation']].agg('-'.join,axis=1)
courses = courses.drop(columns=['code_module','code_presentation'])

# merge the two
demo = demo.merge(courses,on="code_module_presentation", how="inner")

# prep registration
reg = pd.read_csv('open+university+learning+analytics+dataset/studentRegistration.csv')
reg = reg.astype({'id_student':'string'})
reg["code_module_student_presentation"]=reg[['code_module','code_presentation','id_student']].agg('-'.join,axis=1)
reg = reg.drop(columns=['code_module','code_presentation','id_student'])

# merge
demo = demo.merge(reg,on="code_module_student_presentation", how="inner")

# prep assessments
assmt = pd.read_csv('open+university+learning+analytics+dataset/assessments.csv')
std_assmt = pd.read_csv('open+university+learning+analytics+dataset/studentAssessment.csv')
merged_assmt = assmt.merge(std_assmt,on="id_assessment", how="inner")

merged_assmt["code_module_presentation"]=merged_assmt[['code_module','code_presentation']].agg('-'.join,axis=1)
merged_assmt = merged_assmt.astype({'id_student':'string'})
merged_assmt["code_module_student_presentation"]=merged_assmt[['code_module','code_presentation','id_student']].agg('-'.join,axis=1)
merged_assmt["score"] = merged_assmt["score"].replace({"?": np.nan})
merged_assmt["score"] = merged_assmt["score"].astype(float)
merged_assmt["weight"] = merged_assmt["weight"].astype(float)

merged_assmt["component_score"] = merged_assmt["score"] * merged_assmt["weight"]
merged_assmt["component_score"] = merged_assmt["component_score"].div(100)

final_score = merged_assmt.groupby('code_module_student_presentation').sum("component_score")
final_score = final_score.drop(columns=['id_assessment','date_submitted','is_banked','score', 'weight'], axis=1)
final_score = final_score.rename(columns = {'component_score': 'final_score'}, inplace = False)

# merge
demo = demo.merge(final_score,on="code_module_student_presentation", how="outer")

demo.loc[demo['code_module'].isin(['CCC', 'DDD']), 'final_score'] = demo['final_score']/2

# prep vle
vle = pd.read_csv('open+university+learning+analytics+dataset/vle.csv')
vle = vle.astype({'id_site':'string'})
vle["code_module_site_presentation"]=vle[['code_module','code_presentation','id_site']].agg('-'.join,axis=1)
vle = vle.drop(columns=['code_module','code_presentation','id_site'])

std_vle = pd.read_csv('open+university+learning+analytics+dataset/studentVle.csv')
std_vle = std_vle.astype({'id_site':'string'})
std_vle["code_module_site_presentation"]=std_vle[['code_module','code_presentation','id_site']].agg('-'.join,axis=1)
std_vle = std_vle.astype({'id_student':'string'})
std_vle["code_module_student_presentation"]=std_vle[['code_module','code_presentation','id_student']].agg('-'.join,axis=1)

vle = vle.merge(std_vle,on="code_module_site_presentation", how="outer")

sum_click = vle.groupby('code_module_student_presentation').sum("sum_click")
sum_click = sum_click.drop("date", axis=1)

# merge
demo = demo.merge(sum_click,on="code_module_student_presentation", how="outer")

count_click = vle.groupby('code_module_student_presentation').count()
count_click = count_click.drop(columns=['week_from','week_to', 'activity_type', 'code_module_site_presentation', 'code_module','code_presentation','id_student','id_site','date'])
count_click = count_click.rename(columns = {'sum_click': 'count_click'}, inplace = False)

# merge
demo = demo.merge(count_click,on="code_module_student_presentation", how="outer")

demo.to_csv("merged_data.csv", index=False)