# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 18:13:42 2021

@author: deshp
"""

import pandas as pd

df = pd.read_csv('glassdoor_jobs.csv')

# Salary Parsing

df = df[df['Salary Estimate'] != '-1']
df = df[df['Founded'].notnull()]

salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
minus_kd = salary.apply(lambda x: x.replace('K','').replace('$',''))

df['min_salary'] = minus_kd.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = minus_kd.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (df['min_salary'] + df['max_salary']) / 2

# State field
df['job_state'] = df.Location.apply(lambda x: x.split(',')[1])
df.job_state.value_counts()

# Age of company
df['company_age'] = df.Founded.apply(lambda x: 2021 - x)

# Parsing job description

#python
df['python_jd'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
df['python_jd'].value_counts()

# r studio
df['r'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() else 0)
df['r'].value_counts()

#aws
df['aws'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
df['aws'].value_counts()

#spark
df['spark'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
df['spark'].value_counts()

#excel
df['excel'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)
df['excel'].value_counts()

df.to_csv('salary_data_cleaned.csv', index = False)