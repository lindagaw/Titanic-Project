
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold



FeatureNames = ['PassengerId','Survived','Pclass','FName','LName','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
testFeatureNames = ["PassengerId","Pclass"	,"Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]
df= pd.read_csv('../Preprocessed/preprocessed.csv', header=0, names = FeatureNames)
#Testdf=pd.read_csv('test.csv', names = testFeatureNames)
i=0
kfOut = KFold(n_splits=5)
for train, test in kfOut.split(df['Pclass']):
    #print (i,'\n')
    foldTrain=df.iloc[train]
#    print(xfoldTrain.shape)
    foldTest=df.iloc[test]
    foldTrain.to_csv("../SubFiles/TrainPart"+str(i)+".csv", sep=',')
    foldTest.to_csv("../SubFiles/TestPart"+str(i)+".csv", sep=',')
    i=i+1

