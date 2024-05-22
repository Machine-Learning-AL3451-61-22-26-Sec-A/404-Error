# Install necessary packages
!pip install streamlit scikit-learn

# Import required libraries
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Load the Iris dataset
dataset = load_iris()

# Streamlit app title
st.title("Iris Flower Classification")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(dataset["data"], dataset["target"], random_state=0)

# Model training
kn = KNeighborsClassifier(n_neighbors=1)
kn.fit(X_train, y_train)

# Streamlit app
st.sidebar.title("Iris Flower Classifier")

# User input for new data point
sepal_length = st.sidebar.slider("Sepal Length", float(dataset.data[:, 0].min()), float(dataset.data[:, 0].max()), float(dataset.data[:, 0].mean()))
sepal_width = st.sidebar.slider("Sepal Width", float(dataset.data[:, 1].min()), float(dataset.data[:, 1].max()), float(dataset.data[:, 1].mean()))
petal_length = st.sidebar.slider("Petal Length", float(dataset.data[:, 2].min()), float(dataset.data[:, 2].max()), float(dataset.data[:, 2].mean()))
petal_width = st.sidebar.slider("Petal Width", float(dataset.data[:, 3].min()), float(dataset.data[:, 3].max()), float(dataset.data[:, 3].mean()))

# Prediction for the new data point
x_new = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = kn.predict(x_new)
predicted_class = dataset.target_names[prediction][0]

# Display prediction result
st.write(f"Predicted Iris Class: {predicted_class}")
