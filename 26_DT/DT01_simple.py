#Topic: Decision Tree
#-----------------------------
#libraries

#Decision Tree - one of the most popular ML algo
#obsvervation to conclusion(category); Observations are represented as branches, conclusions as leaves
#classification tree - target variable discrete value
#reression tree - target is continuous value

#Method1
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np

#DT
X = [[10], [20], [15], [30], [18], [17]] #is it raining #0-no, 1-yes
Y = [0,1,0,1,1,0]  #class labels -  0- play no, 1- play yes
X
Y

clf = tree.DecisionTreeClassifier()
clf.fit(X,Y)

#visualise
tree.plot_tree(decision_tree= clf)
tree.plot_tree(decision_tree= clf, feature_names =['raining'], class_names=['Dont Play','Play'])  #1-Play, 0=dontplay

#predict for unknown instance
clf.predict([[0.4]])



from sklearn import tree
tree.plot_tree(decision_tree=clf)

#now create case of Cretig rating and DV
#https://docs.google.com/spreadsheets/d/1TIKi-K6qGU5RlLgqsQOp9bVkgGELtKDF6YtnnHFJMmo/edit#gid=1380162551

