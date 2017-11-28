

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import seaborn as sns



FeatureNames = ['PassengerId','Survived','Pclass','FName','LName','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
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



#print(df)






SelectedFeatures=['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
X = df[SelectedFeatures].values
testX=df[SelectedFeatures].values
#X = df[['Fare','Age']].values
#print(X)
y = df['Survived'].values
#print(X)

#we create an instance of SVM and fit out data. We do not scale our
#data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C)
          ,svm.SVC(kernel='rbf', gamma=0.7, C=C)
          ,svm.SVC(kernel='poly', degree=2, C=C)
#          ,svm.SVC(kernel='poly', degree=3, C=C)
         ,svm.SVC(kernel='sigmoid', C=C)
          )


Algs=('SVM(linear)',
      'LinearSVC',
      'SVM(RBF)',
      'SVM(poly, degree=2)',
#      'SVM(poly, degree=3)',
       'SVM(sigmoid)'
        )


for clf in models:
    clf.fit(X, y)
    Z = clf.predict(X)
    print (accuracy_score(y, Z, normalize=True))
   
   
  ###########Selected features
features = SelectedFeatures
print(features)
####################

NumFold1=5
NumFold2=5
#implemented using splitted files to ensure fair comparison
#kf1 = KFold(n_splits=NumFold1)
kf2= KFold(n_splits=NumFold2)
TestAccuracydf=pd.DataFrame()
ValAccuracydf=pd.DataFrame()
BestTraininglAccuracydf=pd.DataFrame()
foldnumber=0



TestCrossdf=pd.DataFrame()
for i in range(NumFold1):
    #print (i,'\n')
    foldTrain=pd.read_csv("../SubFiles/TrainPart"+str(i)+".csv", header=0, names = FeatureNames)
#    print(xfoldTrain.shape)
    foldTest=df= pd.read_csv("../SubFiles/TestPart"+str(i)+".csv", header=0, names = FeatureNames)
    
    foldTrain['Sex'] = le.fit_transform(foldTrain['Sex'])
    foldTrain['Parch']=le.fit_transform(foldTrain['Parch'])
    foldTrain['Embarked']=le.fit_transform(foldTrain['Embarked'])
    
    foldTest['Sex'] = le.fit_transform(foldTest['Sex'])
    foldTest['Parch']=le.fit_transform(foldTest['Parch'])
    foldTest['Embarked']=le.fit_transform(foldTest['Embarked'])    

#    print(foldTrain)
#    print(foldTest)
    j=0
#    TrainCrossdf=pd.DataFrame()

    

    AverageAccuracy={key: 0 for key in Algs}
    BestAlgo=None
    BestAccuracy=0
    BestModel=None
#
    
    for valTrain, ValTest in kf2.split(foldTrain['Survived'] ):
        print (i," ", j, " ","\n")
        
        ######################################
        foldTrainVal=foldTrain.iloc[ valTrain]
        foldTestVal=foldTrain.iloc[ValTest]
        
        for model, Alg in zip(models,Algs):
            
            print(Alg)
            
            Dictionary={}   
            Dictionary['Algorithem']=Alg
            Dictionary['_TestFold' ]=i
                  
            model.fit(foldTrainVal[features], foldTrainVal['Survived'])
            preds = model.predict(foldTestVal[features])
            #print(foldTrainVal['Survived'])
            #print(preds)
            #tempCrossdf=pd.crosstab(index=foldTestVal['Class'], columns=preds, rownames=['actual'], colnames=['preds'])
    #        TrainCrossdf=TrainCrossdf+tempCrossdf
            Dictionary['_ValFold']=j
            
            tn, fp, fn, tp=confusion_matrix(foldTestVal['Survived'], preds).ravel()
            
            #print ("tn, fp, fn, tp  :\n",tn," ", fp," ", fn," ", tp)
#           tp, fn, fp, tn=confusion_matrix(foldTestVal['Class'], preds).ravel()
            Dictionary['Accuracy']=accuracy_score(foldTestVal['Survived'], preds, normalize=True) 
            #Dictionary['ValPrecisionFold%d' % j ]        =precision_score(foldTestVal['Class'], preds, average='macro') 
            #Dictionary['ValRecallFold%d' % j ]        =recall_score(foldTestVal['Class'], preds, average='macro')
            if((tp+fp)==0):
                Dictionary['Precision' ]        =1
            else:
                Dictionary['Precision' ]        =tp/(tp+fp)
            Dictionary['Recall']        = tp/(tp+fn)
            Dictionary['Specificty']         = tn / (tn+fp)
            ValAccuracydf=ValAccuracydf.append(Dictionary, ignore_index=True)
            AverageAccuracy[Alg]=AverageAccuracy[Alg]+Dictionary['Accuracy']
            
                
            
        j=j+1
   
       
    for model, Alg in zip(models,Algs):
        AverageAccuracy[Alg]=AverageAccuracy[Alg]/NumFold2
        if(AverageAccuracy[Alg]>BestAccuracy):
            BestAccuracy=AverageAccuracy[Alg]
            BestAlgo=Alg
            BestModel=model
    print(AverageAccuracy)    
        
    TestDictionary={}  
    TestDictionary['Optimal hyperparameter/Model']=BestAlgo
    TestDictionary['Validation accuracy']=BestAccuracy
    TestDictionary['FolD#']=i
    BestModel.fit(foldTrainVal[features], foldTrainVal['Survived'])
    preds = BestModel.predict(foldTest[features])
    TestDictionary['Test Accuracy' ]=accuracy_score(foldTest['Survived'], preds, normalize=True) 
    BestTraininglAccuracydf=BestTraininglAccuracydf.append(TestDictionary, ignore_index=True)
       
    for model, Alg in zip(models,Algs):
        Dictionary={}   

        Dictionary['_TestFold' ]=i
        Dictionary['Algorithem']=Alg
        
        model.fit(foldTrainVal[features], foldTrainVal['Survived'])
        preds = model.predict(foldTest[features])
        tempCrossdf=pd.crosstab(index=foldTest['Survived'], columns=preds, rownames=['actual'], colnames=['preds'])
#        TrainCrossdf=TrainCrossdf+tempCrossdf
   
#        tp, fn, fp, tn=confusion_matrix(foldTest['Class'], preds).ravel()
        
        tn, fp, fn, tp=confusion_matrix(foldTest['Survived'], preds).ravel()
        #print ("tn, fp, fn, tp  :\n",tn," ", fp," ", fn," ", tp)
        Dictionary['Accuracy' ]=accuracy_score(foldTest['Survived'], preds, normalize=True) 
        if((tp+fp)==0):
            Dictionary['Precision' ]        =1
        else:
            Dictionary['Precision' ]        =tp/(tp+fp)
        Dictionary['Recall']        =tp/(tp+fn)
        Dictionary['Specificty' ]         = tn / (tn+fp)
        TestAccuracydf=TestAccuracydf.append(Dictionary, ignore_index=True)
        
ValAccuracydf.to_csv("ValidationResult.csv", sep=',')
TestAccuracydf.to_csv("TestResult.csv", sep=',')
BestTraininglAccuracydf.to_csv("BestTrainingAcuuracy.csv", sep=',') 
plt.figure()
print(ValAccuracydf)
print(TestAccuracydf)
ax = sns.boxplot(x='Algorithem', y='Accuracy', data=ValAccuracydf, linewidth=2.5,color='white')
plt.xticks(rotation=90)
plt.tight_layout()
plt.setp(ax.lines, color=".1")
ax.set_title("Validation Accuracy")
plt.savefig('ValBoxPlotAcc',bbox_inches='tight')
plt.show()
plt.figure()

ax = sns.boxplot(x='Algorithem', y='Precision', data=ValAccuracydf, linewidth=2.5,color='white')
plt.xticks(rotation=90)
plt.setp(ax.lines, color=".1")
plt.tight_layout()
ax.set_title("Validation Precision")
plt.savefig('ValBoxPlotPrec',bbox_inches='tight')
plt.show()
plt.figure()

ax = sns.boxplot(x='Algorithem', y='Recall', data=ValAccuracydf, linewidth=2.5,color='white')
plt.xticks(rotation=90)
plt.setp(ax.lines, color=".1")
plt.tight_layout()
ax.set_title("Validation Recall")
plt.savefig('ValBoxPlotRecall',bbox_inches='tight')
plt.show()
plt.figure()

ax = sns.boxplot(x='Algorithem', y='Specificty', data=ValAccuracydf,linewidth=2.5, color='white')
plt.xticks(rotation=90)
plt.setp(ax.lines, color=".1")
ax.set_title("Validation Specificity")
plt.savefig('ValBoxPlotSpeci',bbox_inches='tight')
plt.show()
plt.figure()
##############################################
ax = sns.boxplot(x='Algorithem', y='Accuracy', data=TestAccuracydf,linewidth=2.5, color='white')
plt.xticks(rotation=90)
plt.setp(ax.lines, color=".1")
plt.tight_layout()
ax.set_title("Test Accuracuy")
plt.savefig('TestBoxPlotAcc',bbox_inches='tight')
plt.show()
plt.figure()

ax = sns.boxplot(x='Algorithem', y='Precision', data=TestAccuracydf, linewidth=2.5,color='white')
plt.xticks(rotation=90)
plt.setp(ax.lines, color=".1")
plt.tight_layout()
ax.set_title("Test Precision")
plt.savefig('TestBoxPlotPrec',bbox_inches='tight')
plt.show()
plt.figure()

ax = sns.boxplot(x='Algorithem', y='Recall', data=TestAccuracydf,linewidth=2.5, color='white')
plt.xticks(rotation=90)
plt.setp(ax.lines, color=".1")
plt.tight_layout()
ax.set_title("Test Recall")
plt.savefig('TestBoxPlotRecall',bbox_inches='tight')
plt.show()
plt.figure()


ax = sns.boxplot(x='Algorithem', y='Specificty', data=TestAccuracydf, linewidth=2.5,color='white')
plt.xticks(rotation=90)
plt.setp(ax.lines, color=".1")
plt.tight_layout()
ax.set_title("Test Specificity")
plt.savefig('TestBoxPlotSpec',bbox_inches='tight')
plt.show()
plt.figure()
#print(features)

