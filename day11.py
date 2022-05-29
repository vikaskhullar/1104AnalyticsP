# -*- coding: utf-8 -*-
"""
Created on Mon May  2 21:01:15 2022

@author: vikas
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = {'x1': [25,34,22,27,33,33,31, 22,35,34,67,54,57,43,50,57,59,52,65, 47,49,48,35,33,44,45,38,43,51,46],'x2': [79,51,53,78,59,74,73,57,69,75,51,32, 40,47,53,36,35,58, 59,50,25,20,14,12,20,5,29,27,8,7]       }

df = pd.DataFrame(data,columns=['x1','x2'])
df

df.to_csv("data.csv")

plt.scatter(df['x1'], df['x2'])


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2) 
kmeans.fit(df)

centroids = kmeans.cluster_centers_
centroids

label = kmeans.labels_
label

plt.scatter(df['x1'], df['x2'], c=label)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=80, marker="*")




kmeans = KMeans(n_clusters=10) 
kmeans.fit(df)

centroids = kmeans.cluster_centers_
centroids

label = kmeans.labels_
label

plt.scatter(df['x1'], df['x2'], c=label)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=80, marker="*")



kmeans.inertia_



data = {'x1': [25,34,22,27,33,33,31, 22,35,34,67,54,57,43,50,57,59,52,65, 47,49,48,35,33,44,45,38,43,51,46],'x2': [79,51,53,78,59,74,73,57,69,75,51,32, 40,47,53,36,35,58, 59,50,25,20,14,12,20,5,29,27,8,7]       }
df = pd.DataFrame(data,columns=['x1','x2'])
type(df)
sse=[10025]


for k in range(2,11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)
    sse.append(kmeans.inertia_)

sse


plt.plot(range(1,11), sse)
df.shape

from kneed import KneeLocator
kl = KneeLocator(x=range(1,11), y=sse, curve='convex', direction='decreasing')
kl.elbow



kmeans = KMeans(n_clusters=3)

kmeans.fit(df)

centroids = kmeans.cluster_centers_
centroids

label = kmeans.labels_
label

plt.scatter(df['x1'], df['x2'], c=label)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=80, marker="*")


# Loan Data

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('Loan.csv')
df = df.drop(['Unnamed: 0'], axis=1)
df

from sklearn.cluster import KMeans

sse=[491148680]
for k in range(2,11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)
    sse.append(kmeans.inertia_)
    
sse

plt.plot(range(1,11),sse)


from kneed import KneeLocator
kl = KneeLocator(x=range(1,11), y=sse, curve='convex', direction='decreasing')
kl.elbow


kmeans = KMeans (n_clusters=4)
kmeans.fit(df)

df.columns


centroids = kmeans.cluster_centers_
centroids

label = kmeans.labels_
label


plt.scatter(df['ApplicantIncome'], df['LoanAmount'], c=label)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=80, marker="*")




#Income


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("income.csv")
df.columns

from sklearn.cluster import KMeans
sse=['412840']
for k in range(2,11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)
    sse.append(kmeans.inertia_)
    
sse


plt.plot(range(1,11),sse)


from kneed import KneeLocator
kl = KneeLocator(x=range(1,11), y=sse, curve='convex', direction='decreasing')
kl.elbow


kmeans = KMeans(n_clusters=3)

kmeans.fit(df)

kmeans.labels_

df['labels']=kmeans.labels_

df.to_csv('incomeLab.csv')


#Food Odering

df =pd.read_csv("FoodOrder.csv")

df.columns

df.dtypes

pd.get_dummies(df['NPS_Category'])




from sklearn import preprocessing

col = df.select_dtypes(include='object').columns

for c in col:
    le = preprocessing.LabelEncoder()
    le.fit(df[c])
    df[c] = le.transform(df[c])

df.dtypes

df.to_csv('foodnum.csv')

df = df.drop(['Cust_Id'], axis=1)


from sklearn.cluster import KMeans
sse=[124288]
for k in range(2,11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)
    sse.append(kmeans.inertia_)
    
sse


plt.plot(range(1,11),sse)


from kneed import KneeLocator
kl = KneeLocator(x=range(1,11), y=sse, curve='convex', direction='decreasing')
kl.elbow


kmeans = KMeans(n_clusters=3)

kmeans.fit(df)

kmeans.labels_

df['labels']=kmeans.labels_

df.to_csv('foodordernum.csv')


#Mtcars

from pydataset import data
df = data('mtcars')
df.columns

from sklearn.cluster import KMeans
sse=[182564]
for k in range(2,11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)
    sse.append(kmeans.inertia_)
    
sse


plt.plot(range(1,11),sse)


from kneed import KneeLocator
kl = KneeLocator(x=range(1,11), y=sse, curve='convex', direction='decreasing')
kl.elbow


kmeans = KMeans(n_clusters=3)

kmeans.fit(df)

kmeans.labels_

df['labels']=kmeans.labels_

df.to_csv('mtcarslables.csv')




# SegimentationData

df = pd.read_csv('Segmentation_Data v01.csv')

df=df.drop(['Cust_id'], axis=1)
df.columns
from sklearn.cluster import KMeans
sse=[2504243]
for k in range(2,11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)
    sse.append(kmeans.inertia_)
    
sse


plt.plot(range(1,11),sse)


from kneed import KneeLocator
kl = KneeLocator(x=range(1,11), y=sse, curve='convex', direction='decreasing')
kl.elbow


kmeans = KMeans(n_clusters=3)

kmeans.fit(df)

kmeans.labels_

df['labels']=kmeans.labels_

df.to_csv('SD.csv')



































































