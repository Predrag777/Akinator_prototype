import numpy as np
import pandas as pd

data=pd.read_csv('Dataset.csv')
data = data.drop(data.columns[0], axis=1) # Color
data = data.drop(data.columns[4], axis=1) # Color
data = data.drop(data.columns[4], axis=1) # Color
data = data.drop(data.columns[3], axis=1) # Color
data = data.drop(data.columns[3], axis=1) # Color
data = data.drop(data.columns[4], axis=1) # Color
data = data.drop(data.columns[4], axis=1) # Color


questions=data.columns.tolist()
print(questions)

def questions_builder(questions):
    c=0
    arr=[]
    for i in questions:
        if c==0:
            arr.append("Is your animal tall?")
        elif c==1:
            arr.append("Is your animal heavy?")
        elif c==2:
            arr.append("Is your animal long-lived?")
        elif c==3:
            arr.append("Is your animal fast?")
        elif c<76:
            arr.append(f"Is {i} enemy of your animal?")
        elif c<84:
            arr.append(f"Is {i} color of your animal?")
        else:
            arr.append(f'Is {i} habitat of your animal?')
        c+=1
    return arr

questions=questions_builder(questions)
for i in questions:
    print(i)