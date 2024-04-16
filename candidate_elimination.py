import streamlit as st
import numpy as np
import pandas as pd

def learn(concepts, target):
    specific_h = concepts[0].copy()
    general_h = [['?' for _ in range(len(specific_h))] for _ in range(len(specific_h))]
    for i, h in enumerate(concepts):
        if target[i] == "yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
        if target[i] == "no":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'
    indices = [i for i, val in enumerate(general_h) if val == ['?' for _ in range(len(specific_h))]]
    for i in indices:
        general_h.remove(['?' for _ in range(len(specific_h))])
    return specific_h, general_h

def main():
    st.title("Concept Learning Algorithm")

    # Read the dataset
    data = pd.read_csv('/content/drive/MyDrive/ml/datasets/enjoysport.csv')
    st.write("Dataset:")
    st.write(data)

    concepts = data.values[:, :-1]
    target = data.values[:, -1]

    # Apply the concept learning algorithm
    s_final, g_final = learn(concepts, target)

    st.subheader("Final Specific Hypothesis:")
    st.write(s_final)
    
    st.subheader("Final General Hypotheses:")
    st.write(g_final)

if __name__ == "__main__":
    main()

