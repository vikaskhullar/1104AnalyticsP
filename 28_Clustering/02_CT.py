#Topic: Clustering
#-----------------------------
#making groups and find group properties
#libraries

#kmeans - minimise distance of points in the cluster with their centeroid

import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt

#data 
#data = pd.read_csv('data/clustering.csv')
url='https://raw.githubusercontent.com/DUanalytics/pyAnalytics/master/data/clustering.csv'
data = pd.read_csv(url)
data.shape
data.head()
data.describe()
data.columns

#visualise
plt.scatter(data.ApplicantIncome, data.LoanAmount)
plt.xlabel('Income')
plt.ylabel('LoanAmt')
plt.show();

#standardize data : Scaling

#missing values
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html
data= data[['ApplicantIncome', 'LoanAmount']]
data.dtypes
data.isnull().any()
sum(data.isnull().any(axis=1))
data.index[data.isnull().any(axis=1)]
data.isnull().sum().sum()  #75 missing values 
data.isnull().sum(axis=0)  #columns missing
data.isnull().sum(axis=1)


data1 = data.dropna()

data1.isnull().any()
data1.isnull().sum().sum()
data1.dtypes

data2 =  data1.select_dtypes(exclude=['object'])
data2.head()
data2.dtypes

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
data2_scaled = scalar.fit_transform(data2)
data2_scaled

#pd.DataFrame(data2_scaled).describe()



#%%choosing no of clusters
kmeans_kwargs = {'init':'random', 'n_init':10, 'max_iter': 300}
sse=[]
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(data)
    sse.append(kmeans.inertia_)
sse

plt.style.use('fivethirtyeight')
plt.plot(range(1,11), sse)
plt.xticks(range(1,11))
plt.xlabel('No of clusters')
plt.ylabel('SSE')
plt.show();

kl = KneeLocator(x=range(1,11), y=sse, curve='convex', direction='decreasing')
kl.elbow


#kmeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)  #hyper parameters

kmeans.fit(data)
kmeans.inertia_  #sum of sq distances of samples to their centeroid
kmeans.cluster_centers_
kmeans.labels_
kmeans.n_iter_  #iterations to stabilise the clusters


centroids = kmeans.cluster_centers_


plt.scatter(data['ApplicantIncome'], data['LoanAmount'], c= kmeans.labels_.astype(float), s=50)

cl = ['r','g','b']
plt.scatter(centroids[:, 0], centroids[:, 1], c=cl, s=80, marker="*")
plt.show()



data1['labels'] = kmeans.labels_

data1

data1.to_csv("Loan_Clustered.csv")

data2['labels'].value_counts


data2['pred'] = kmeans.predict(data2_scaled)

data2.columns

data2.to_csv("Loan.csv")


datatest = pd.DataFrame({'ApplicantIncome':  [2770], 'CoapplicantIncome':[2000], 
                         'LoanAmount':[121],
       'Loan_Amount_Term':[290], 'Credit_History':[1]})

datatest
data2.columns
datatest_scaled = scalar.transform(datatest)

datatest
datatest_scaled

kmeans.predict(datatest_scaled)

data2_scaled[1:5]

data.columns
data2.columns
NCOLS = data2.columns 
#['ApplicantIncome','CoapplicantIncome','LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
NCOLS
clusterNos = kmeans.labels_
clusterNos
type(clusterNos)



plt.scatter(data2.ApplicantIncome, data2.LoanAmount, c=clusterNos)
plt.scatter(data2.ApplicantIncome, data2.Credit_History, c=clusterNos) #better distinction
plt.scatter(data2.ApplicantIncome, data2.Loan_Amount_Term, c=clusterNos) #better distinction




'''

K Means clustering requires prior knowledge of K i.e. no. of clusters you want 
to divide your data into. But, you can stop at whatever number of clusters you 
find appropriate in hierarchical clustering by interpreting the dendrogram.

Difference between K Means and Hierarchical clustering
Hierarchical clustering can’t handle big data well but K Means clustering can. 
This is because the time complexity of K Means is linear i.e. O(n) while that 
of hierarchical clustering is quadratic i.e. O(n2).
In K Means clustering, since we start with random choice of clusters, the 
results produced by running the algorithm multiple times might differ. 
While results are reproducible in Hierarchical clustering.
K Means is found to work well when the shape of the clusters is hyper spherical
(like circle in 2D, sphere in 3D).

https://www.analyticsvidhya.com/blog/2016/11/an-introduction-to-clustering-and-different-methods-of-clustering/
'''

#Now use this information ;
#which customers you would like to target.

#hierarchical clustering
import scipy.cluster.hierarchy as shc
dend = shc.dendrogram(shc.linkage(data2_scaled, method='ward'))

plt.figure(figsize = (10,7))
plt.title("Dendrogram")
dend = shc.dendrogram(shc.linkage(data2_scaled, method='ward'))

d= list(dend['color_list'])

import numpy as np
np.unique(d)

plt.axhline(y=5, color='r', linestyle='--')
plt.show();

#another method for Hcluster from sklearn
from sklearn.cluster import AgglomerativeClustering

aggCluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')

aggCluster

aggCluster.fit_predict(data2_scaled)

aggCluster
aggCluster.labels_

aggCluster.fit_predict(data2_scaled)
#compare