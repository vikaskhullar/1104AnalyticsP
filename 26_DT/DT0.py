#Topic:Decision Tree - Class and Regr
#-----------------------------
#Read the case from this link
#https://stackabuse.com/decision-trees-in-python-with-scikit-learn/

#%% Classification Tree - Predict if the Currency is FAKE or not depending upon features

#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import os
#os.listdir('D:\Training ML Code Notes\Henry Harvin Trainer\PyAnalyticsTrainer\PyAnalyticsTrainerVikas\XIM Later\DT') #change the folder to see what are the file in folder
#dataset
#data = pd.read_csv('E:/analytics/projects/pyanalytics/data/bill_authentication.csv')
#data = pd.read_csv('https://raw.githubusercontent.com/DUanalytics/pyAnalytics/master/data/bill_authentication.csv')
data = pd.read_csv('https://raw.githubusercontent.com/vikaskhullar/Dataset/master/binary.csv')
data.head()
data.shape
data.columns
data
#data preparation : X & Y
X= data.drop('admit', axis=1) #axis=1 -> column
y= data['admit']
X
y
y.value_counts()



#split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.05)
X_train.shape
X_test.shape


#model
from sklearn.tree import DecisionTreeClassifier

clsModel = DecisionTreeClassifier()  #model with parameter

clsModel.fit(X_test, y_test)

tree.plot_tree(decision_tree= clsModel)

#predict
ypred1 = clsModel.predict(X_test)
ypred1


tree.plot_tree(decision_tree= clsModel)



#metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

cr=classification_report(y_test, ypred1)

cr

'''
  precision    recall  f1-score   support\n\n           
  0       0.99      1.00      0.99        77\n           
  1       1.00      0.98      0.99        61\n\n    
  accuracy                           0.99       138\n   
  macro avg       0.99      0.99      0.99       138\n
  weighted avg       0.99      0.99      0.99       138\n'
  
'''


confusion_matrix(y_true=y_test, y_pred=ypred1)
accuracy_score(y_true=y_test, y_pred=ypred1)




print("Accuracy:",metrics.accuracy_score(y_test, ypred1))
print("Precision:",metrics.precision_score(y_test, ypred1))
print("Recall:",metrics.recall_score(y_test, ypred1))




y_pred_proba = clsModel.predict_proba(X_test)[::,1]
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


'''
'              precision    recall  f1-score   support\n\n
           0       0.99      0.99      0.99       159\n
           1       0.98      0.99      0.99       116\n\
accuracy                           0.99
'''


from sklearn.linear_model import LogisticRegression

LR =LogisticRegression()

LR.fit(X_train, y_train)

#predict
ypred2 = LR.predict(X_test)
ypred2



#metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

cr=classification_report(y_test, ypred2)

cr

confusion_matrix(y_true=y_test, y_pred=ypred2)
accuracy_score(y_true=y_test, y_pred=ypred2)




print("Accuracy:",metrics.accuracy_score(y_test, ypred2))
print("Precision:",metrics.precision_score(y_test, ypred2))
print("Recall:",metrics.recall_score(y_test, ypred2))




y_pred_proba = clsModel.predict_proba(X_test)[::,1]
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







'''
'              precision    recall  f1-score   support\n\n
           0       0.99      0.99      0.99       159\n
           1       0.99      0.99      0.99       116\n\n
accuracy                           0.99      
'''


'''
Gini index or Gini impurity measures the degree or probability of a particular 
variable being wrongly classified when it is randomly chosen. But what is 
actually meant by ‘impurity’? If all the elements belong to a single class, 
then it can be called pure. The degree of Gini index varies between 0 and 1, 
where 0 denotes that all elements belong to a certain class or if there exists 
only one class, and 1 denotes that the elements are randomly distributed across
various classes. A Gini Index of 0.5 denotes equally distributed elements 
into some classes.
'''


from sklearn import tree
tree.plot_tree(decision_tree=clsModel, fontsize=5)

'''
tree.plot_tree(decision_tree=clsModel, feature_names=['Var', 'Skew', ' Kur',  'Ent'], class_names=['1','0'], fontsize=8)
#not a good way to draw graphs.. other methods to be experimented
tree.plot_tree(decision_tree=clsModel, max_depth=2, feature_names=['Var', 'Skew', ' Kur',  'Ent'], class_names=['Orgiginal','Fake'], fontsize=8)
tree.plot_tree(decision_tree=clsModel, max_depth=3, feature_names=['Var', 'Skew', ' Kur',  'Ent'], class_names=['Orgiginal','Fake'], fontsize=8)
'''