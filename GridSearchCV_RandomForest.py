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

#number of principal components
n_pc = 2

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

#partition into train (80%) and test (20%) splits
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)

#initialize pipeline
pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', RandomForestClassifier(max_depth=100))])

params = param_grid={'classifier__n_estimators': [10, 100, 1000]}

gc = GridSearchCV(pipeline, params, cv=10, verbose=3,return_train_score=True)

gc.fit(X_train,Y_train)

print(gc.best_params_)
print(gc.best_score_)

