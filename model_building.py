# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 13:39:40 2021

@author: deshp
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('eda_data.csv')

# Choose columns
df.columns

df_model = df[['avg_salary', 'Rating', 'Size', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 'comp_num', 'hourly', 'employer_provided', 'job_state', 'same_state', 'age', 'python_yn', 
       'spark', 'aws', 'excel', 'job_simp', 'seniority', 'desc_len']]


# Get dummy data'
df_dummy = pd.get_dummies(df_model)


# Split test and train data
from sklearn.model_selection import train_test_split

X = df_dummy.drop('avg_salary', axis = 1)
y = df_dummy.avg_salary.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Multiple linear regression
# 1. StatsModel ols regression
import statsmodels.api as sm

X_sm = X = sm.add_constant(X)

model = sm.OLS(y, X_sm)
model.fit().summary()


# 2. sklearn linear regression
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lr = LinearRegression()
lr.fit(X_train, y_train)

np.mean(cross_val_score(lr, X_train, y_train, scoring = 'neg_mean_absolute_error', cv=3))

# lasso regression
lr_l = Lasso(alpha = .2)
lr_l.fit(X_train, y_train)
np.mean(cross_val_score(lr_l, X_train, y_train, scoring = 'neg_mean_absolute_error', cv=3))

alpha = []
error = []

for i in range(1, 100):
    alpha.append(i/100)
    lrl = Lasso(alpha = (i/100))
    error.append(np.mean(cross_val_score(lrl, X_train, y_train, scoring = 'neg_mean_absolute_error', cv=3)))
    
plt.plot(alpha, error)

err = tuple(zip(alpha, error))
df_err = pd.DataFrame(err, columns = ['alpha', 'error'])
df_err[df_err.error == max(df_err.error)]

# random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
np.mean(cross_val_score(rf, X_train, y_train, scoring = 'neg_mean_absolute_error', cv=3))

# tune models GridsearchCV
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10, 300, 10), 'criterion':('mse', 'mae'), 'max_features':('auto', 'sqrt', 'log2')}
gs = GridSearchCV(rf, parameters, scoring = 'neg_mean_absolute_error', cv=3)
gs.fit(X_train, y_train)
gs.best_score_
gs.best_estimator_

# Test ensembles
tpred_lr = lr.predict(X_test)
tpred_lrl = lr_l.predict(X_test)
tpred_gs = gs.best_estimator_.predict(X_test)
 
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, tpred_lr)
mean_absolute_error(y_test, tpred_lrl)
mean_absolute_error(y_test, tpred_gs)


# productionize a model in flask

import pickle
pickl = {'model': gs.best_estimator_}
pickle.dump( pickl, open( 'model_file' + ".p", "wb" ) )

file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

model.predict(X_test.iloc[1,:].values.reshape(1,-1))

list(X_test.iloc[1,:])