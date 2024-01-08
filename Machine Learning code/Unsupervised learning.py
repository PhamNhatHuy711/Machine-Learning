import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

data = pd.read_excel("c:\\Users\\LEGION\\Downloads\\ds_country_preprocessing.xlsx")
df = pd.DataFrame(data)
print(df)

data_features = data.drop(['country'], axis=1)
a=[]
K=range(1,10)
for i in K:
    kmean=KMeans(n_clusters=i, n_init=10)
    kmean.fit(data_features)
    a.append(kmean.inertia_)
    
plt.plot(K,a,marker='o')
plt.title('Elbow Method',fontsize=20)
plt.xlabel('Số cụm',fontsize=16)
plt.ylabel('Tổng khoảng cách bình phương',fontsize=16)
plt.show()


for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, n_init=11, random_state=42)
    cluster_labels = kmeans.fit_predict(data_features)
    silhouette_avg = silhouette_score(data_features, cluster_labels)
    print(f"For n_clusters = {n_clusters}, the average silhouette_score is {silhouette_avg}")


# Chọn số cụm (ví dụ: 3 cụm)
num_clusters = 3

# Tạo mô hình K-Means
kmeans = KMeans(n_clusters=num_clusters)

# Fit mô hình vào dữ liệu
kmeans.fit(data_features)

# Lấy thông tin về nhãn của từng điểm dữ liệu
cluster_labels = kmeans.labels_

# Thêm nhãn cụm vào tập dữ liệu gốc
data_features['cluster'] = cluster_labels

print(data_features)

import matplotlib.pyplot as plt

# Lấy thông tin về tọa độ của các cụm
cluster_centers = kmeans.cluster_centers_
# Convert to DataFrame
df = pd.DataFrame(data)
df = data.drop(['country'], axis=1)


# Scale the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df_two_columns = df[['income', 'gdpp']]

# Scale the two selected columns
scaler = StandardScaler()
df_two_columns_scaled = scaler.fit_transform(df_two_columns)

# Apply K-Means clustering on the scaled two columns
kmeans = KMeans(n_clusters=3, random_state=42)
df_two_columns['cluster'] = kmeans.fit_predict(df_two_columns_scaled)

# Get the centroids for the two columns
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
colors = ['blue', 'red', 'green']
# Plotting
plt.figure(figsize=(10,6))
for i in range(3):
    # Plot data points
    plt.scatter(df_two_columns[df_two_columns['cluster'] == i]['income'], 
                df_two_columns[df_two_columns['cluster'] == i]['gdpp'], 
                s=50, c=colors[i], label=f'Cluster {i+1}')
    
    # Plot centroids
    plt.scatter(centroids[i, 0], centroids[i, 1], s=200, c='yellow', marker='s')

plt.title('Thu nhập so với GDPP với phân cụm K-Means,fontsize=20',fontsize=20)
plt.xlabel('Thu nhập',fontsize=16)
plt.ylabel('GDPP',fontsize=16)
plt.legend()
plt.grid(True)
plt.show()


from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
neighbors = NearestNeighbors(n_neighbors=2)
neighbors_fit = neighbors.fit(df_two_columns_scaled)
distances, indices = neighbors_fit.kneighbors(df_two_columns_scaled)

# Sort the distances
sorted_distances = np.sort(distances, axis=0)
sorted_distances = sorted_distances[:, 1]  # Take the second closest point distance (not the point itself)

# Plotting the K-distance Graph
plt.figure(figsize=(10,6))
plt.plot(sorted_distances)
plt.title('Đồ thị khoảng cách K',fontsize=20)
plt.xlabel('Điểm đ,fontsize=20ược sắp xếp theo khoảng cách',fontsize=16)
plt.ylabel('Eps (khoảng cách đến điểm thứ n gần nhất)',fontsize=16)
plt.grid(True)
plt.show()

# This graph will help us to visually identify the 'eps' value at the point of maximum curvature, also known as the "elbow" method
# min_samples is usually chosen based on domain knowledge and the dataset at hand; a common starting point is 2 or more

from sklearn.cluster import DBSCAN

# Apply DBSCAN algorithm to the scaled data
# The parameters for eps and min_samples can be adjusted as needed
# Here, they are chosen arbitrarily
dbscan = DBSCAN(eps=1.5, min_samples=5)
clusters_dbscan = dbscan.fit_predict(df_two_columns_scaled)

# Adding DBSCAN cluster labels to the dataframe copy
df_two_columns['dbscan_cluster'] = clusters_dbscan

# Get unique clusters labels
unique_clusters = set(clusters_dbscan)

# Plotting
plt.figure(figsize=(10,6))
for cluster in unique_clusters:
    # Selecting data points that belong to the current cluster
    cluster_data = df_two_columns[df_two_columns['dbscan_cluster'] == cluster]
    # Assigning a label for noise (-1) and for clusters
    label = 'Noise' if cluster == -1 else f'Cluster {cluster}'
    # Plot data points
    plt.scatter(cluster_data['income'], cluster_data['gdpp'], s=50, label=label)

plt.title('Thu nhập so với GDPP với Phân cụm DBSCAN',fontsize=20)
plt.xlabel('Thu nhập',fontsize=16)
plt.ylabel('GDPP',fontsize=16)
plt.legend()
plt.grid(True)
plt.show()


from sklearn.cluster import AgglomerativeClustering

# Apply Hierarchical Agglomerative Clustering (HAC) algorithm
# Using the default linkage criterion 'ward' which minimizes the variance of clusters being merged.
# We will choose the number of clusters to be 3, as per the original request for clustering into 3 clusters

hac = AgglomerativeClustering(n_clusters=2)
clusters_hac = hac.fit_predict(df_two_columns_scaled)

# Adding HAC cluster labels to the dataframe copy
df_two_columns['hac_cluster'] = clusters_hac
import scipy.cluster.hierarchy as sch

plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = False
dendrogram = sch.dendrogram(sch.linkage(data_features, method='ward'))
plt.title("Dendrogram sử dụng phương pháp Ward", fontsize=20)
plt.xlabel('Cụm', fontsize=16)
plt.ylabel('khoảng cách Euclide', fontsize=16)
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = False
plt.show()
# Plotting
plt.figure(figsize=(10,6))
# We use the cluster labels as color map for plotting
scatter = plt.scatter(df_two_columns['income'], df_two_columns['gdpp'], c=clusters_hac, cmap='viridis', s=50)
plt.title('Thu nhập so với GDPP với Phân cụm HAC',fontsize=20)
plt.xlabel('Thu nhập',fontsize=16)
plt.ylabel('GDPP',fontsize=16)
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.grid(True)
plt.show()

# Display the number of clusters
num_clusters_hac = len(set(clusters_hac))
num_clusters_hac

from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

# Assuming df_two_columns_scaled is the scaled data and the clustering labels are as assigned in the provided code

# K-Means Clustering
kmeans_labels = df_two_columns['cluster']

# DBSCAN Clustering
dbscan_labels = df_two_columns['dbscan_cluster']

# Hierarchical Agglomerative Clustering (HAC)
hac_labels = df_two_columns['hac_cluster']

# Calculating metrics for K-Means
kmeans_calinski = calinski_harabasz_score(df_two_columns_scaled, kmeans_labels)
kmeans_davies = davies_bouldin_score(df_two_columns_scaled, kmeans_labels)

# Calculating metrics for DBSCAN
dbscan_calinski = calinski_harabasz_score(df_two_columns_scaled, dbscan_labels)
dbscan_davies = davies_bouldin_score(df_two_columns_scaled, dbscan_labels)

# Calculating metrics for HAC
hac_calinski = calinski_harabasz_score(df_two_columns_scaled, hac_labels)
hac_davies = davies_bouldin_score(df_two_columns_scaled, hac_labels)

print("Chỉ số của kmeans_calinski",kmeans_calinski)
print("Chỉ số của kmeans_davies",kmeans_davies) 
print("Chỉ số của dbscan_calinski",dbscan_calinski) 
print("Chỉ số của dbscan_davies",dbscan_davies) 
print("Chỉ số của hac_calinski",hac_calinski) 
print("Chỉ số của hac_davies",hac_davies)

import matplotlib.pyplot as plt

# Dữ liệu của bạn
algos = ['K-means', 'DBSCAN', 'HAC']
davies_bouldin = [kmeans_davies, dbscan_davies, hac_davies]

# Vẽ biểu đồ cột
plt.bar(algos, davies_bouldin, color='skyblue')
plt.xlabel('Các Thuật Toán',fontsize=16)
plt.ylabel('Chỉ số Davies-Bouldin',fontsize=16)
plt.title('Biểu Đồ So Sánh Davies-Bouldin Giữa Các Thuật Toán',fontsize=20)
plt.show()
