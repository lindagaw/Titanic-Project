"""
==================================================
Plot different SVM classifiers in the iris dataset
==================================================

Comparison of different linear SVM classifiers on a 2D projection of the iris
dataset. We only consider the first 2 features of this dataset:

- Sepal length
- Sepal width

This example shows how to plot the decision surface for four SVM classifiers
with different kernels.

The linear models ``LinearSVC()`` and ``SVC(kernel='linear')`` yield slightly
different decision boundaries. This can be a consequence of the following
differences:

- ``LinearSVC`` minimizes the squared hinge loss while ``SVC`` minimizes the
  regular hinge loss.

- ``LinearSVC`` uses the One-vs-All (also known as One-vs-Rest) multiclass
  reduction while ``SVC`` uses the One-vs-One multiclass reduction.

Both linear models have linear decision boundaries (intersecting hyperplanes)
while the non-linear kernel models (polynomial or Gaussian RBF) have more
flexible non-linear decision boundaries with shapes that depend on the kind of
kernel and its parameters.

.. NOTE:: while plotting the decision function of classifiers for toy 2D
   datasets can help get an intuitive understanding of their respective
   expressive power, be aware that those intuitions don't always generalize to
   more realistic high-dimensional problems.

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score



FeatureNames = ['PassengerId','Survived','Pclass','FName','LName','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
testFeatureNames = ["PassengerId","Pclass"	,"Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]
df= pd.read_csv('../Preprocessed/preprocessed.csv', header=0, names = FeatureNames)
le = preprocessing.LabelEncoder()
min_max_scaler = preprocessing.MinMaxScaler()

print(df)
#df['PassengerId'] = le.fit_transform(df['PassengerId'])
#df['FName'] = le.fit_transform(df['FName'])
#df['Pclass'] = le.fit_transform(df['Pclass'])
#df['Fare'] = le.fit_transform(df['Fare'])
#df['Ticket'] = le.fit_transform(df['Ticket'])
df['Sex'] = le.fit_transform(df['Sex'])
df['Parch']=le.fit_transform(df['Parch'])
df['Embarked']=le.fit_transform(df['Embarked'])
#df['Age'] =min_max_scaler.fit_transform(df['Age'])
#df['Fare']=min_max_scaler.fit_transform(df['Fare'])

df['Fare']=df['Fare'].str[2:-1]
df['Fare']= pd.to_numeric(df['Fare'], errors='coerce',downcast="float")


df['Pclass']=df['Pclass'].str[2:-1]

df['SibSp']=df['SibSp'].str[2:-1]

print(df['Pclass'])
i=0



print(df['Fare'])






SelectedFeatures=['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
X = df[SelectedFeatures].values
testX=df[SelectedFeatures].values
#X = df[['Fare','Age']].values
print(X)
y = df['Survived'].values
print(X)

#we create an instance of SVM and fit out data. We do not scale our
#data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
#          svm.SVC(kernel='rbf', gamma=0.7, C=C)
#          ,svm.SVC(kernel='poly', degree=3, C=C)
#         ,svm.SVC(kernel='sigmoid', C=C)
          )
models = (clf.fit(X, y) for clf in models)
print(models)

# title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')


for clf in models:
   Z = clf.predict(X)
   print (accuracy_score(y, Z, normalize=True))

