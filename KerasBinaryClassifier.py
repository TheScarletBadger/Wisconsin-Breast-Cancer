# -*- coding: utf-8 -*-
"""
Simple neural network classification
"""
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from keras import Input
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# fetch dataset from uci machine learning repository
rawdata = fetch_ucirepo(id=17) 
df = rawdata.data.original

#drop id numbers
df.drop(["ID"],axis=1,inplace=True)

#extract raw predictors
x = df.drop(["Diagnosis"],axis=1).values


#transform raw training data

y = (df["Diagnosis"] == 'M').astype(int)

#partition into train (80%) and test (20%) splits
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



model = Sequential()
model.add(Input(shape=(x_train.shape[1],)))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'] )
model.fit(x_train, y_train, epochs=50, verbose=2)

pred_test= model.predict(x_test)
scores = model.evaluate(x_test, y_test, verbose=2)
print('Accuracy on test data: {} \n Error on test data: {}'.format(scores[1], 1 - scores[1]))