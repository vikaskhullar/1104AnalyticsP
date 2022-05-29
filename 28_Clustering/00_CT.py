#Topic:Clustering - marks and mtcars
#-----------------------------
#libraries

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
import pandas as pd

data = {'x1': [25,34,22,27,33,33,31, 22,35,34,67,54,57,43,50,57,59,52,65, 47,49,48,35,33,44,45,38,43,51,46],'x2': [79,51,53,78,59,74,73,57,69,75,51,32, 40,47,53,36,35,58, 59,50,25,20,14,12,20,5,29,27,8,7]       }
  
df = pd.DataFrame(data,columns=['x1','x2'])
print (df)

plt.scatter(df['x1'], df['x2'])



kmeans = KMeans(n_clusters=20)
kmeans.fit(df)

centroids = kmeans.cluster_centers_

print(centroids)

kmeans.labels_.astype(float)


plt.scatter(df['x1'], df['x2'], c= kmeans.labels_.astype(float), s=50)

plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=80, marker="*")
plt.show()






sse=[]

kmeans_kwargs = {'init':'random', 'n_init':10, 'max_iter': 300}


for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(df)
    sse.append(kmeans.inertia_)
sse

plt.style.use('fivethirtyeight')
plt.plot(range(1,11), sse)
plt.xticks(range(1,11))
plt.xlabel('No of clusters')
plt.ylabel('SSE')
plt.show();



from kneed import KneeLocator
kl = KneeLocator(x=range(1,11), y=sse, curve='convex', direction='decreasing')
kl.elbow



kmeans = KMeans(n_clusters=3)
kmeans.fit(df)

centroids = kmeans.cluster_centers_

print(centroids)

kmeans.labels_.astype(float)


plt.scatter(df['x1'], df['x2'], c= kmeans.labels_.astype(float), s=50)

cl = ['r','g','b']
plt.scatter(centroids[:, 0], centroids[:, 1], c=cl, s=80, marker="*")
plt.show()


df['lab'] = kmeans.labels_

df


!pip install kneed
from kneed import KneeLocator


kl = KneeLocator(x=range(1,11), y=sse, curve='convex', direction='decreasing')
kl.elbow




kmeans.predict([[22,66]])

kmeans.predict([[66,99]])

kmeans.predict([[40,20]])

kmeans.predict([[60,40]])

#%% 4 clusters
kmeans = KMeans(n_clusters=4).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)

plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()


#mtcars

from pydataset import data
mtcars = data('mtcars')
data = mtcars.copy()

data.head(2)


sse=[]
kmeans_kwargs = {'init':'random', 'n_init':10, 'max_iter': 300}

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



from kneed import KneeLocator
kl = KneeLocator(x=range(1,11), y=sse, curve='convex', direction='decreasing')
kl.elbow





kmeans = KMeans( init = 'random', n_clusters=2,  max_iter=300)
kmeans
kmeans.fit(data)

kmeans.cluster_centers_  #average or rep values
kmeans.labels_

data['labels'] = kmeans.labels_

data


'''
array([1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0,
       0, 0, 0, 1, 1, 1, 0, 1, 0, 1])

array([1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0,
       0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
'''


from sklearn.preprocessing import StandardScaler
#need for scaling : height & weight are in different scales
scaler = StandardScaler()

scaled_features = scaler.fit_transform(data)

scaled_features[:5]  #values between -3 to +3

kmeans = KMeans( init = 'random', n_clusters=3,  max_iter=300)
kmeans
kmeans.fit(scaled_features)

kmeans.inertia_

kmeans.cluster_centers_  #average or rep values
kmeans.n_iter_  #in 6 times, clusters stabilised

kmeans.labels_
kmeans.cluster_centers_.shape
kmeans.cluster_centers_[0:1]
#https://realpython.com/k-means-clustering-python/

data1=data

data["labels1"] =kmeans.labels_
data
