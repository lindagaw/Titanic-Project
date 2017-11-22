import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import csv

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

names = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

df = pd.read_csv('train.txt', header=None, names=names)
df.head()

predictions = []

X = np.array(df.ix[:, ('Sex','Age','Pclass','SibSp','Parch')])     # end index is exclusive
y = np.array(df['Survived'])   # another way of indexing a pandas df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
cv_scores = []

for k in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

MSE = [1 - x for x in cv_scores]

optimal_k = MSE.index(min(MSE))+1
print("The optimal number of neighbors is %d" % optimal_k)

myList = list(range(1,50))
plt.plot(myList, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()

def predict(X_train, y_train, x_test, k):
    distances = []
    targets = []

    for i in range(len(X_train)):
        distance = np.sqrt(np.sum(np.square(x_test - X_train[i, :])))
        distances.append([distance, i])

    distances = sorted(distances)

    for i in range(k):
        index = distances[i][1]
        targets.append(y_train[index])

    return Counter(targets).most_common(1)[0][0]

def kNearestNeighbor(X_train, y_train, X_test, predictions, k):
    for i in range(len(X_test)):
        predictions.append(predict(X_train, y_train, X_test[i, :], k))
        
predictions = []

names = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Survived']
test = pd.read_csv('test1.txt', header=None, names=names)
test.head()

X = np.array(test.ix[:, ('Sex','Age','Pclass','SibSp','Parch')])     # end index is exclusive

kNearestNeighbor(X_train, y_train, X, predictions, 7)
predictions = np.asarray(predictions)
#print(predictions)
test["Survived"] = predictions
#print(test)
test.to_csv("test2.csv", sep=',', index = False)
