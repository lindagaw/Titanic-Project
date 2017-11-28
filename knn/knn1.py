import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#names = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

df = pd.read_csv('train.csv') 
df = df.drop(['Name','Ticket', 'Cabin'], axis=1)
df = df.dropna()
df = pd.get_dummies(df, ['Embarked'])

df.head()

X = np.array(df.ix[:, ('Pclass','Sex','Age','SibSp','Parch','Fare','Embarked_C','Embarked_Q','Embarked_S')])     # end index is exclusive
y = np.array(df['Survived'])   # another way of indexing a pandas df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
cv_scores = []
real_scores = []

for k in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=k)
    # cross validation change to 5
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')

    cv_scores.append(scores.mean())

for k in range(1,6):
    knn = KNeighborsClassifier(n_neighbors=k)
    # cross validation change to 5
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    print("Fold: ") + str(k) + str(scores)

MSE = [1 - x for x in cv_scores]

optimal_k = MSE.index(min(MSE))+1
print("The optimal number of neighbors is %d" % optimal_k)

myList = list(range(1,50))
plt.plot(myList, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()

def train(X_train, y_train):
    # do nothing 
    return

def predict(X_train, y_train, x_test, k):
    # create list for distances and targets
    distances = []
    targets = []

    for i in range(len(X_train)):
        # first we compute the euclidean distance
        distance = np.sqrt(np.sum(np.square(x_test - X_train[i, :])))
        # add it to list of distances
        distances.append([distance, i])

    # sort the list
    distances = sorted(distances)

    # make a list of the k neighbors' targets
    for i in range(k):
        index = distances[i][1]
        targets.append(y_train[index])

    # return most common target
    return Counter(targets).most_common(1)[0][0]

def kNearestNeighbor(X_train, y_train, X_test, predictions, k):
    # train on the input data
    train(X_train, y_train)

    # loop over all observations
    for i in range(len(X_test)):
        predictions.append(predict(X_train, y_train, X_test[i, :], k))

scaler = StandardScaler()
features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked_C','Embarked_Q','Embarked_S']
X = df[features]
y = df['Survived']

predictions = []

kNearestNeighbor(X_train, y_train, X_test, predictions, optimal_k)

# transform the list into an array
predictions = np.asarray(predictions)
# evaluating accuracy
accuracy = accuracy_score(y_test, predictions)


#print('\nThe accuracy of our classifier is' % accuracy*100)
print("The testing accuracy is: ") + str(accuracy)
