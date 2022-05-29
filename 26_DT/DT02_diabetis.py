#Topic: DT - Diabetis Data Set
#-----------------------------
#pip install graphviz  #install whichever library is not present
#pip install pydotplus

# Load libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics, tree

#%%%% : Load Data
url='https://raw.githubusercontent.com/DUanalytics/datasets/master/csv/pima-indians-diabetes.csv'
#This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.
#https://www.kaggle.com/uciml/pima-indians-diabetes-database
#he datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.
# load dataset
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(url, header=None, names=col_names)
pima.head()
pima.label.value_counts() #how many are diabetic - 268
pima.shape

#%%% : Feature Selection
#need to divide given columns into two types of variables dependent(or target variable) and independent variable(or feature variables).

#split dataset in features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
X = pima[feature_cols] # Features - bmi, age etc
y = pima.label # Target variable : has diabetes =1
#predict y on X
#%%% Splitting Data
#To understand model performance, dividing the dataset into a training set and a test set is a good strategy.
#Let's split the dataset by using function train_test_split(). You need to pass 3 parameters features, target, and test_set size.
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 
# 70% training and 30% test : each for train and test (X & y)
X_train.head()

#%%%: Building Decision Tree Model :create a Decision Tree Model using Scikit-learn.
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
y_train
#Predict the response for test dataset
y_pred = clf.predict(X_test)
y_pred
#%%% : Evaluating Model
# estimate, how accurately the classifier or model can predict the type of cultivars.# Accuracy can be computed by comparing actual test set values and predicted values.
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#classification rate of 67.53%, considered as good accuracy. You can improve this accuracy by tuning the parameters in the Decision Tree Algorithm.
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))



print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average="weighted"))
print("Recall:",metrics.recall_score(y_test, y_pred, average="weighted"))





y_pred_proba = clf.predict_proba(X_test)[::,1]
y_pred_proba
y_pred_proba.shape


fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)

fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

#together

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show();







from sklearn.linear_model import LogisticRegression

LR =LogisticRegression()

LR.fit(X_train, y_train)

#predict
y_pred = LR.predict(X_test)
y_pred

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))



print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))



y_pred_proba = LR.predict_proba(X_test)[::,1]
y_pred_proba
y_pred_proba.shape


fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)

fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

#together

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show();








from sklearn.ensemble import RandomForestClassifier

RF=RandomForestClassifier()

RF.fit(X_train, y_train)

#predict
ypred3 = RF.predict(X_test)
ypred3

print("Accuracy:",metrics.accuracy_score(y_test, ypred3))
print("Precision:",metrics.precision_score(y_test, ypred3))
print("Recall:",metrics.recall_score(y_test, ypred3))




#metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

cr=classification_report(y_test, y_pred)
print(cr)
'''
'              precision    recall  f1-score   support\n\n           0       0.79      0.90      0.84       146\n           1       0.78      0.58      0.66        85\n\n    accuracy                           0.78       231\n   macro avg       0.78      0.74      0.75       231\nweighted avg       0.78      0.78      0.78       231\n'
'''


#%%% 
y_test.shape, y_pred.shape
y_test.head()
y_pred[0:6]




clf3 = DecisionTreeClassifier(max_depth=3)
# Train Decision Tree Classifer
clf3 = clf3.fit(X_train,y_train)
#Visualise

from graphviz import Source
from sklearn import tree
tree.plot_tree(decision_tree=clf3, fontsize=8)


#display(SVG(graph3b.pipe(format='svg')))
X_train[0:1]  
#Class:1 : glucose > 127, glucose < 158, bmi, age,
#Predict the response for test dataset
y_pred3 = clf3.predict(X_test)
len(X_test)
y_pred3
len(y_pred3)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred3))
#classification rate increased to 77.05%, which is better accuracy than the previous model.

#----
clf4 = DecisionTreeClassifier(criterion="gini", max_depth=3)
# Train Decision Tree Classifer
clf4 = clf4.fit(X_train,y_train)
y_pred4 = clf4.predict(X_test)
tree.plot_tree(decision_tree= clf4, fontsize=8)

fig = plt.figure(figsize=(10,8))
_ = tree.plot_tree(clf4, filled=True, fontsize=7)  #see plot

print("Accuracy:",metrics.accuracy_score(y_test, y_pred4))


#%%%%
#Optimizing Decision Tree Performance
#criterion : optional (default=”gini”) or Choose attribute selection measure: This parameter allows us to use the different-different attribute selection measure. Supported criteria are “gini” for the Gini index and “entropy” for the information gain.

#splitter : string, optional (default=”best”) or Split Strategy: This parameter allows us to choose the split strategy. Supported strategies are “best” to choose the best split and “random” to choose the best random split.

#max_depth : int or None, optional (default=None) or Maximum Depth of a Tree: The maximum depth of the tree. If None, then nodes are expanded until all the leaves contain less than min_samples_split samples. The higher value of maximum depth causes overfitting, and a lower value causes underfitting (Source).

#%%%% - short summary
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))