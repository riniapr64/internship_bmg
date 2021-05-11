import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

data = pd.read_excel('test data.xlsx')

data.head()

data_x = data.iloc[:, 0:2]
data_x.head()

plt.scatter(data.v1, data.v2, s =10, c = "c", marker = "o", alpha = 1)
plt.title("Performance")
plt.show()

data_array = np.array(data_x)
print(data_array)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_array)
data_scaled

kmeans = KMeans(n_clusters = 3, random_state=123)

kmeans.fit(data_scaled)
kmeans

print(kmeans.cluster_centers_)

data["cluster"] = kmeans.labels_

data.cluster

output = plt.scatter(data_scaled[:,0], data_scaled[:,1], s = 100, c = data.cluster, marker = "o", alpha = 1, )
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c="red", s=200, alpha=1 , marker="o");
plt.title("Hasil Clustering K-Means")
plt.colorbar (output)
plt.show()

from pandas import DataFrame

df = DataFrame (data, columns= ['v1', 'v2','cluster'])

df.to_csv(r'Documents\bismillah1.csv')