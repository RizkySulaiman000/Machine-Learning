import pandas as pd
from sklearn.cluster import KMeans

# (step 1) Menyediakan dataset yang sesuai dengan kasus Clustering.
dataset = pd.read_csv('wine.csv')

# (step 2) Menentukan jumlah Cluster.
k = 3 

# (step 3) Menentukan algoritma clustering
kmeans = KMeans(n_clusters=k)

# (step 4) Proses Clustering
kmeans.fit(dataset)

# Mendapatkan label dari tiap cluster/kelompok
labels = kmeans.labels_

# Menampilkan hasil clustering
for i, label in enumerate(labels):
    print("Instance", i, "belongs to cluster", label)

