# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 20:52:23 2026

@author: Barry
"""
from ucimlrepo import fetch_ucirepo
from numpy import round
import sklearn

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
X_raw_train, X_raw_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, stratify=Y)

#initialize scaler object and fit to raw training data
X_scaler = sklearn.preprocessing.StandardScaler()
X_scaler.fit(X_raw_train)

#transform raw training data
X_scaled_train = X_scaler.transform(X_raw_train)

#fit logistic regression model to scaled data without first performing PCA
log_regressor = sklearn.linear_model.LogisticRegression()
log_regressor.fit(X_scaled_train, Y_train)

#calculate and print accuracy and precision scores for model without PCA
train_preds = log_regressor.predict(X_scaled_train)
train_accuracy = round(sklearn.metrics.accuracy_score(Y_train,train_preds),decimals=3)
train_precision = round(sklearn.metrics.precision_score(Y_train,train_preds),decimals=3)
test_preds = log_regressor.predict(X_scaler.transform(X_raw_test))
test_accuracy = round(sklearn.metrics.accuracy_score(Y_test,test_preds),decimals=3)
test_precision = round(sklearn.metrics.precision_score(Y_test,test_preds),decimals=3)
print('Without Dimension Reduction')
print(f'Training Set: Accuracy = {train_accuracy}, Precision = {train_precision}')
print(f'Test Set: Accuracy = {test_accuracy}, Precision = {test_precision}\n')

#initialize pca object with n_pc principal components and fit to training data
pca = sklearn.decomposition.PCA(n_components=n_pc)
pca.fit(X_scaled_train)

#print variance explained by principal components
evr = pca.explained_variance_ratio_.sum()*100
print(f'{evr.round(decimals=1)}% of variance in predictor variables is captured in {n_pc} principal components\n')


log_regressor_pca = sklearn.linear_model.LogisticRegression()
log_regressor_pca.fit(pca.transform(X_scaled_train), Y_train)
train_preds_pca = log_regressor_pca.predict(pca.transform(X_scaled_train))
train_accuracy_pca = round(sklearn.metrics.accuracy_score(Y_train,train_preds_pca),decimals=3)
train_precision_pca = round(sklearn.metrics.precision_score(Y_train,train_preds_pca),decimals=3)
test_preds_pca = log_regressor_pca.predict(pca.transform(X_scaler.transform(X_raw_test)))
test_accuracy_pca = round(sklearn.metrics.accuracy_score(Y_test,test_preds_pca),decimals=3)
test_precision_pca = round(sklearn.metrics.precision_score(Y_test,test_preds_pca),decimals=3)
print('With Dimension Reduction')
print(f'Training Set: Accuracy = {train_accuracy_pca}, Precision = {train_precision_pca}')
print(f'Test Set: Accuracy = {test_accuracy_pca}, Precision = {test_precision_pca}')


