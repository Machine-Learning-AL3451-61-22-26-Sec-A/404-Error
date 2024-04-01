import pandas as pd
from pandas import DataFrame
data=pd.read_csv('/content/drive/MyDrive/ml/datasets/enjoysport.csv')
concepts=data.values[:,:-1]
target=data.values[:,-1]
def learn(concepts, target):
    specific_h = concepts[0].copy()
    general_h = [['?' for i in range(len(specific_h))] for i in range(len(specific_h))]
    for i, h in enumerate(concepts):
        if target[i] == "yes":
            #print(target[i])
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
    indices = [i for i,val in enumerate(general_h) if val==['?' for i in range(len(specific_h))]]
    for i in indices:
        general_h.remove(['?' for i in range(len(specific_h))])
    return specific_h, general_h
s_final, g_final = learn(concepts, target)
print("Final S:", s_final, sep="\n")
print("Final G:", g_final, sep="\n")
