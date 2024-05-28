import numpy as np
import pandas as pd
import streamlit as st

def initialize_centroids(X, k):
    """ Randomly initialize k centroids from the data points. """
    return X[np.random.choice(X.shape[0], k, replace=False)]

def assign_clusters(X, centroids):
    """ Assign each data point to the nearest centroid. """
    distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def update_centroids(X, labels, k):
    """ Update the centroids by calculating the mean of the assigned points. """
    return np.array([X[labels == i].mean(axis=0) for i in range(k)])

def kmeans(X, k, max_iters=100, tol=1e-4):
    """ The K-means clustering algorithm. """
    centroids = initialize_centroids(X, k)
    for i in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        centroids = new_centroids
    return centroids, labels

# Streamlit UI
st.title("K-means Clustering")

st.write("### Upload your data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.write(data.head())
    
    st.write("### Select Features for Clustering")
    features = st.multiselect("Choose features", data.columns.tolist())
    
    if features:
        X = data[features].values
        
        st.write("### Set Number of Clusters (k)")
        k = st.slider("k", 1, 10, 3)
        
        st.write("### Run K-means Clustering")
        if st.button("Run"):
            centroids, labels = kmeans(X, k)
            
            st.write("### Clustered Data")
            data['Cluster'] = labels
            st.write(data)
            
            st.write("### Centroids")
            st.write(centroids)
            
            st.write("### Cluster Plot")
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
            ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
            ax.set_xlabel(features[0])
            ax.set_ylabel(features[1])
            st.pyplot(fig)
