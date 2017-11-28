# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 11:05:19 2017

@author: Robert
"""
from sklearn.utils import shuffle
from sklearn import model_selection
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy
import sys
import pandas
import itertools

def runRFC_DoubleCrossValid(data_trainSplits,data_testSplits, target_trainSplits, target_testSplits, numOuterFolds, numInnerFolds):

    outerScores = []
    outerConfusionMatrices = []
    forests = []
    trainSplitIndices = numpy.arange(len(data_trainSplits)) 
    
    masterDataMatrix = []

    for splitIndex in trainSplitIndices:
        dataTrain, dataTest = data_trainSplits[splitIndex], data_testSplits[splitIndex]
        targetTrain, targetTest = target_trainSplits[splitIndex][0], target_testSplits[splitIndex][0]

        #In the inner fold experiment, we'll test different values of num estimators in smaller k-fold validations.
        # We'll eventually get a classifier of maximum average accuracy out of it, which we'll display.
        innerScores_maxAvg = []
        forest = RFC(n_jobs=2,n_estimators= 25)
    
        scores = []
        scores = model_selection.cross_val_score(forest, dataTrain, y=targetTrain, cv = numInnerFolds)
        if len(innerScores_maxAvg) == 0 or (numpy.mean(scores) > numpy.mean(innerScores_maxAvg)):
            innerScores_maxAvg = scores
        
        innerScoresForDataMatrix = numpy.transpose(innerScores_maxAvg)
        masterDataMatrix.append(innerScoresForDataMatrix)

        #After number of estimators of maximum accuracy has been found, graph them out.
#        plt.figure(3)
#        plt.title('Table Results for RFC Accuracy, {}th fold'.format(splitIndex))
#        plt.gca().xaxis.set_visible(False)
#        plt.gca().yaxis.set_visible(False)
#        columns = ["%Acc"]
#        plt.table(cellText = numpy.transpose([innerScores_maxAvg]), colLabels = columns)
#        plt.savefig("Table_{}Fold".format(splitIndex+1))
        
        print("{0}|{1}".format(splitIndex+1,innerScores_maxAvg))
       

        
        #Fit the model to the data, because it apparently ain't doing it itself.
        
        forest.fit(dataTrain, targetTrain)
        
        #Now, we predict on the test set, getting the accuracy score, confusion matrix.
        test_predictions = forest.predict(dataTest)
        test_correctPredictions = (test_predictions == targetTest)
        test_numCorrectPreds = test_correctPredictions.sum()
        test_numPredictions = len(test_correctPredictions)
        
        test_accuracy = test_numCorrectPreds / test_numPredictions * 100
        outerScores.append(test_accuracy)
        outerConfusionMatrix = confusion_matrix(targetTest, test_predictions)
        outerConfusionMatrices.append(outerConfusionMatrix)
        
        #Append the random forests to master list
        forests.append(forest)
        
    #Save Accuracies for each Fold
    
    plt.figure(1)
    plt.title("Box Plot of Accuracies, 25 Estimators")
    plt.xlabel("Random Forest Classifier, ith Fold")
    plt.ylabel("Prediction Accuracy")
    plt.boxplot(masterDataMatrix)
    plt.savefig("RFC_FoldAccuracies")

    
    #Plot the final amalgamate box plot.
    plt.figure(2)
    plt.title("Final Box Plot of Test Accuracies, RFC")
    plt.xlabel("Random Forest Classifier")
    plt.ylabel("Prediction Accuracy")
    plt.boxplot(outerScores)
    plt.savefig("RFC_FinalTestAccuracies")
    
    print(outerScores)

    i = 0
    for cm in outerConfusionMatrices:
        plt.figure()
        plot_confusion_matrix(cm, forest.classes_, title="Confusion Matrix, Fold {}".format(i+1 ))
        plt.savefig("RFCConfusionMatrix_Fold{}".format(i+1))
        i = i+1
        
    i = 0
    for forest in forests:
        importances = forest.feature_importances_
        indices = numpy.argsort(importances)
        # Plot the feature importances of the forest
        plt.figure()
        plt.title('Feature Importances of Forest, Fold {}'.format(i+1))
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), wantedFeatures[indices])
        plt.xlabel('Relative Importance')
        plt.savefig("RFCFeatureImportances_Fold{}".format(i+1))

        i = i+1
#This function is cribbed from the SK Learn Documentation, here:
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
        
        
        

#CSV Parsing - Python's got a csv parser already, so we'll just use that.

trainDatasetCSV = sys.argv[1]
#testDatasetCSV = sys.argv[2]
trainCSV0 = "../SubFiles/TrainPart0.csv"
trainCSV1 = "../SubFiles/TrainPart1.csv"
trainCSV2 = "../SubFiles/TrainPart2.csv"
trainCSV3 = "../SubFiles/TrainPart3.csv"
trainCSV4 = "../SubFiles/TrainPart4.csv"

testCSV0 = "../SubFiles/TestPart0.csv"
testCSV1 = "../SubFiles/TestPart1.csv"
testCSV2 = "../SubFiles/TestPart2.csv"
testCSV3 = "../SubFiles/TestPart3.csv"
testCSV4 = "../SubFiles/TestPart4.csv"

featureNames = ["PassengerId","Survived","Pclass"	,"FName", "LName", "Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]
#testFeatureNames = ["PassengerId","Pclass"	,"Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]
dfTrain0 = pandas.read_csv(trainCSV0, header=0, names = featureNames)
dfTrain1 = pandas.read_csv(trainCSV1, header=0, names = featureNames)
dfTrain2 = pandas.read_csv(trainCSV2, header=0, names = featureNames)
dfTrain3 = pandas.read_csv(trainCSV3, header=0, names = featureNames)
dfTrain4 = pandas.read_csv(trainCSV4, header=0, names = featureNames)

dfTest0 = pandas.read_csv(testCSV0, header=0, names = featureNames)
dfTest1 = pandas.read_csv(testCSV1, header=0, names = featureNames)
dfTest2 = pandas.read_csv(testCSV2, header=0, names = featureNames)
dfTest3 = pandas.read_csv(testCSV3, header=0, names = featureNames)
dfTest4 = pandas.read_csv(testCSV4, header=0, names = featureNames)

dfTrainSplits = [dfTrain0, dfTrain1, dfTrain2, dfTrain3, dfTrain4]
dfTestSplits = [dfTest0, dfTest1, dfTest2, dfTest3, dfTest4]
#First, we drop a few features: Cabin for sparsity, Ticket for difficulty of parsing, and Name for 

for i in range(len(dfTrainSplits)):
    df = dfTrainSplits[i]
    df = df.drop(['FName', 'LName', 'Ticket','Cabin', 'PassengerId'], axis=1)
    df = df.dropna()
    df = df[df.Embarked !=" ''"]
    df = shuffle(df)
    dfTrainSplits[i] = df
    
for i in range(len(dfTestSplits)):
    df = dfTestSplits[i]
    df = df.drop(['FName', 'LName', 'Ticket','Cabin', 'PassengerId'], axis=1)
    df = df.dropna()
    df = df[df.Embarked !=" ''"]
    df = shuffle(df)
    dfTestSplits[i] = df
#Encode
for i in range(len(dfTrainSplits)):
    dfTrainSplits[i] = pandas.get_dummies(dfTrainSplits[i], columns=['Sex','Embarked'])
for i in range(len(dfTestSplits)):
    dfTestSplits[i] = pandas.get_dummies(dfTestSplits[i], columns=['Sex','Embarked'])

#Convert Parch column to integers.
char = "'"
for i in range(len(dfTrainSplits)):
    dfTrainSplits[i]['Parch'] = dfTrainSplits[i]['Parch'].str.replace(char, '')
    dfTrainSplits[i]['Parch'] = pandas.to_numeric(dfTrainSplits[i]['Parch'])
for i in range(len(dfTestSplits)):
    dfTestSplits[i]['Parch'] = dfTestSplits[i]['Parch'].str.replace(char, '')
    dfTestSplits[i]['Parch'] = pandas.to_numeric(dfTestSplits[i]['Parch'])

#We don't want the Survived feature in our training set, since that leads to overfitting up the wazoo.
unwantedFeatures = ['Survived']
features = dfTrainSplits[0].axes[1]
wantedFeatures = numpy.array([feature for feature in features if not feature in unwantedFeatures])

y_trainSplits = []
y_testSplits = []

data_trainSplits = []
data_testSplits = []
for i in range(len(dfTrainSplits)):
    y_trainSplits.append(pandas.factorize(dfTrainSplits[i]['Survived']))
    data_trainSplits.append(dfTrainSplits[i].drop(['Survived'], axis=1))
for i in range(len(dfTestSplits)):
    y_testSplits.append(pandas.factorize(dfTestSplits[i]['Survived']))
    data_testSplits.append(dfTestSplits[i].drop(['Survived'], axis=1))

runRFC_DoubleCrossValid(data_trainSplits, data_testSplits, y_trainSplits, y_testSplits, 5, 5)
