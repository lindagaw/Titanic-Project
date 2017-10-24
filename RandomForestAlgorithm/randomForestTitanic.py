# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 11:05:19 2017

@author: Robert
"""

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import preprocessing

import numpy
import sys
import pandas

#CSV Parsing - Python's got a csv parser already, so we'll just use that.

trainDatasetCSV = sys.argv[1]
testDatasetCSV = sys.argv[2] 

trainFeatureNames = ["PassengerId","Survived","Pclass"	,"Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]
testFeatureNames = ["PassengerId","Pclass"	,"Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]
dfTrain = pandas.read_csv(trainDatasetCSV, header=0, names = trainFeatureNames)
dfTest = pandas.read_csv(testDatasetCSV, header=0, names = testFeatureNames)

#In order for the fit function to work properly, I must convert the strings to numeric types
# with Label Encoder. 
le = preprocessing.LabelEncoder()
dfTrain['Name'] = le.fit_transform(dfTrain['Name'])
dfTrain['Ticket'] = le.fit_transform(dfTrain['Ticket'])
dfTrain['Sex'] = le.fit_transform(dfTrain['Sex'])

dfTest['Name'] = le.fit_transform(dfTest['Name'])
dfTest['Ticket'] = le.fit_transform(dfTest['Ticket'])
dfTest['Sex'] = le.fit_transform(dfTest['Sex'])


#As age, cabin columns are incomplete, don't use them for training or testing. We'll sieve them out.
unwantedFeatures = ["Age", "Cabin", "Embarked", "Fare", "Survived"]
features = dfTrain.axes[1]
wantedFeatures = [feature for feature in features if not feature in unwantedFeatures]

#Grow the forest. Beware of giant spiders!
forest = RFC(n_jobs=2,n_estimators=50)

y, _ = pandas.factorize(dfTrain['Survived'])

forest.fit(dfTrain[wantedFeatures], y)

#Have to take out 'Survived' feature for testing.
unwantedFeatForTest = ["Age", "Cabin", "Embarked", "Fare", "Survived"]
features = dfTest.axes[1]
wantedFeatForTrain = [feature for feature in features if not feature in unwantedFeatForTest]

preds = forest.predict(dfTest[wantedFeatForTrain])





