'''import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/LENOVO/Downloads/Mall_Customers.csv")
labelled = LabelEncoder()
df['Genre'] = labelled.fit_transform(df['Genre'])
scaled = StandardScaler()
df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']] = scaled.fit_transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
df = df.drop('CustomerID', axis=1)

n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=34, n_init=6)
kmeans.fit(df)
plt.figure(figsize=(10, 5))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=kmeans.labels_, cmap='viridis', alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s=200, c='red', marker='X')

for i, center in enumerate(kmeans.cluster_centers_):
    plt.text(center[1], center[2], f'Cluster {i + 1}\n({center[1]:.2f}, {center[2]:.2f})',
             fontsize=12, ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=0.5))

plt.title(f'K-means Clustering with k = {n_clusters}')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
print("Coordinates of Centroids:")
for i, center in enumerate(kmeans.cluster_centers_):
    print(f"Cluster {i + 1}: {center}")
   
genre = input("Enter Gender (Male/Female): ").strip()
age = float(input("Enter Age: "))
annual_income = float(input("Enter Annual Income (k$): "))
spending_score = float(input("Enter Spending Score (1-100): "))

encoded_genre = labelled.transform([genre])[0]
user_input = np.array([[encoded_genre, age, annual_income, spending_score]])
user_input_scaled = user_input.copy()  
user_input_scaled[:, 1:] = scaled.transform(user_input[:, 1:])  
predicted_cluster = kmeans.predict(user_input_scaled)
print(f"The entered data belongs to Cluster: {predicted_cluster[0] + 1}")'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target  # Actual labels, used only for color-coding in plot

# Applying KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, label='Clustered Data')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, 
            c='red', marker='X', label='Centroids')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clustering on Iris Dataset (2D PCA Projection)")
plt.legend()
plt.grid(True)
plt.show()

# Optionally, print cluster center coordinates in original feature space
print("Cluster Centers (in original feature space):\n", kmeans.cluster_centers_)
