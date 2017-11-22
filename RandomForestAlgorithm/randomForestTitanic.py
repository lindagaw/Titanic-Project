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

def runRFC_DoubleCrossValid(data, target, numOuterFolds, numInnerFolds):
    #For <numOuterFolds> outer folds:
    outerKF = model_selection.KFold(n_splits = numOuterFolds)
    print("RFC Number of Outer Folds: {}".format(outerKF.get_n_splits(data)))

    outerScores = []
    outerConfusionMatrices = []

    #trainIndices, testIndices = outerKF.split(data)
    for outerTrainIndex, outerTestIndex in outerKF.split(data):
        dataTrain, dataTest = data.iloc[outerTrainIndex], data.iloc[outerTestIndex]
        targetTrain, targetTest = target[0][outerTrainIndex], target[0][outerTestIndex]

        #In the inner fold experiment, we'll test different values of num estimators in smaller k-fold validations.
        # We'll eventually get a classifier of maximum average accuracy out of it, which we'll display.
        innerScores_maxAvg = []
        optimalModel = None
        optimalNumEstimators = 0
        for i in range(5,20):
            forest = RFC(n_jobs=2,n_estimators= i)
                        
            scores = []
            scores = model_selection.cross_val_score(forest, dataTrain, y=targetTrain, cv = numInnerFolds)
            if len(innerScores_maxAvg) == 0 or (numpy.mean(scores) > numpy.mean(innerScores_maxAvg)):
                innerScores_maxAvg = scores
                optimalModel = forest
                optimalNumEstimators = optimalModel.get_params()['n_estimators']
        
        
        plt.figure()
        plt.title("Box Plot of Accuracies, Best N Estimators")
        plt.xlabel("Random Forest Classifier, {} estimators".format(optimalNumEstimators))
        plt.ylabel("Prediction Accuracy")
        plt.boxplot(innerScores_maxAvg)

        #After number of estimators of maximum accuracy has been found, graph them out.
        plt.figure()
        plt.title('Result of Best Inner Fold Experiment, RFC')
        plt.gca().xaxis.set_visible(False)
        plt.gca().yaxis.set_visible(False)
        columns = ["%Acc"]
        plt.table(cellText = numpy.transpose([innerScores_maxAvg]), colLabels = columns)

       
        #Check if model's been selected properly.
        if(optimalModel is None):
            raise ValueError("Optimal Model not set!\n")
        
        #Fit the model to the data, because it apparently ain't doing it itself.
        
        optimalModel.fit(dataTrain, targetTrain)
        
        #Now, we predict on the test set, getting the accuracy score, confusion matrix.
        test_predictions = optimalModel.predict(dataTest)
        test_correctPredictions = (test_predictions == targetTest[0])
        test_numCorrectPreds = test_correctPredictions.sum()
        test_numPredictions = len(test_correctPredictions)
        
        test_accuracy = test_numCorrectPreds / test_numPredictions * 100
        outerScores.append(test_accuracy)
        outerConfusionMatrix = confusion_matrix(targetTest, test_predictions)
        outerConfusionMatrices.append(outerConfusionMatrix)
    #Plot the final amalgamate box plot.
    plt.figure()
    plt.title("Final Box Plot of Test Accuracies, RFC")
    plt.xlabel("Random Forest Classifier")
    plt.ylabel("Prediction Accuracy")
    plt.boxplot(outerScores)
    plt.savefig('BP_RFC_Esth{}_Final.png'.format(len(optimalModel.estimators_)), bbox_inches='tight')

    for cm in outerConfusionMatrices:
        plt.figure()
        plot_confusion_matrix(cm, optimalModel.classes_, title="Confusion Matrix of RFC {}".format(len(optimalModel.estimators_)) )
    
    return optimalModel    
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

featureNames = ["PassengerId","Survived","Pclass"	,"Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]
#testFeatureNames = ["PassengerId","Pclass"	,"Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]
df = pandas.read_csv(trainDatasetCSV, header=0, names = featureNames)

#First, we drop a few features: Cabin for sparsity, Ticket for difficulty of parsing, and Name for 
df = df.drop(['Name','Ticket','Cabin'], axis=1)
df_complete = df.dropna()
df_complete = shuffle(df_complete)

#Encode
df_complete = pandas.get_dummies(df_complete, columns=['Sex','Embarked'])

dfTrain = df_complete.head(int(numpy.ceil(3*len(df_complete)/4)))
dfTest = df_complete.tail(int(numpy.floor(len(df_complete)/4)))


#We don't want the Survived feature in our training set, since that leads to overfitting up the wazoo.
unwantedFeatures = ['Survived']
features = df_complete.axes[1]
wantedFeatures = [feature for feature in features if not feature in unwantedFeatures]

y = pandas.factorize(df_complete['Survived'])
data = df_complete.drop(['Survived'], axis=1)

optimalRFC = runRFC_DoubleCrossValid(data, y, 5, 5)


#forests = []
##First, test out changing the number of estimators.
#for i in range(50):
#    forest = RFC(n_jobs=2,n_estimators= (i+1)) #Test an exponential pattern 
#    
#    
#    y, _ = pandas.factorize(dfTrain['Survived'])
#    
#    forest.fit(dfTrain[wantedFeatures], y)
#    
#    forests.append(forest)
#
#
## ----- Model Testing -------
#forestPreds = []
#forestErrors = []
#
#unwantedFeatForTest = ['Survived']
#features = dfTest.axes[1]
#wantedFeatForTest = [feature for feature in features if not feature in unwantedFeatForTest]
#for forest in forests:
#    
#    preds = forest.predict(dfTest[wantedFeatForTest])
#    
#    predictionCorrect = (preds == dfTest['Survived'])
#    forestPreds.append(preds)
#    forestErrors.append(predictionCorrect)
#
##---Statistics and Plotting ----
#forestPredictionAccuracies = []
#
#for i in range(len(forests)):
#    numCorrectPreds = (forestErrors[i] == True).sum()
#    numTotalPreds = len(forestErrors[i])
#    forestPredictionAccuracy = numCorrectPreds/numTotalPreds
#    forestPredictionAccuracies.append(forestPredictionAccuracy)
#
#forestPredictionAccuracyAvg = numpy.average(forestPredictionAccuracies)






