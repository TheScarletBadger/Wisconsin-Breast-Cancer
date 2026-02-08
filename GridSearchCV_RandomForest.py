# -*- coding: utf-8 -*-
"""
header
"""
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from numpy import round
from matplotlib import pyplot as plt
from math import sqrt

# fetch dataset from uci machine learning repository
rawdata = fetch_ucirepo(id=17) 
df = rawdata.data.original

#drop id numbers
df.drop(["ID"],axis=1,inplace=True)

#extract raw predictors
X = df.drop(["Diagnosis"],axis=1)

#extract targets and encode to malignant = 1, benign = 0
Y = df["Diagnosis"]=='M'
Y = Y.astype('int')

#partition into train (90%) and test (10%) splits
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=42)

#initialize pipeline (note scaler not actually needed for random forest)
pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', RandomForestClassifier(max_depth=100))])

#number of random forest estimators to evaluate
estimator_array = [10, 20, 30, 40, 50, 100, 200, 500, 1000]

#number of cross-validation folds
folds = 5

#create gridsearch parameter dictionary
params = {'classifier__n_estimators': estimator_array}

#use gridsearch to perform cross-validation accross parameters
gsc = GridSearchCV(pipeline, params, cv=folds, verbose=3, return_train_score=True)

#fit gridsearch to training data and report score for best parameters
gsc.fit(X_train,Y_train)
print('\n')
print(f'Best Parameters = {gsc.best_params_}')
print(f'Best Score = {gsc.best_score_}\n')

Y_test_preds =  gsc.predict(X_test)
test_accuracy = round(accuracy_score(Y_test,Y_test_preds),decimals=3)
test_precision = round(precision_score(Y_test,Y_test_preds),decimals=3)
print(f'Best Model: Test Set Accuracy = {test_accuracy}')
print(f'Best Model: Test Set Precision = {test_precision}')

#plot mean cross-validation score +- SEM over number of estimators
fig = plt.figure()
plt.errorbar(estimator_array,
         gsc.cv_results_['mean_test_score'],
         gsc.cv_results_['std_test_score']/sqrt(folds),
         )
plt.xlabel('n estimators')
plt.ylim([0.9,1])
plt.ylabel('test score (mean of folds +- SEM)')
