import pandas as pd
import numpy as np

data = pd.read_csv("Animal Dataset.csv")
questions = data.columns.tolist()[1:]  # Base for questions
y = np.array(data.iloc[:, 0])  # Output
data = data.iloc[:, 1:]  # Only features

features = np.array(data)  # Make numpy array




height_data=features[:,0]
weight_data=features[:,1]
diet=  features[:,4]
predators=features[:, 6]

for i in range(len(predators)):
    print(y[i],'  ',predators[i])
def parse_data(data, log=False):
    min_num=99999
    max_num=-99999
    arr=[]
    for i in data:
        i=i.replace(',','.')
        if '-' in i:
            temp=i.split('-')
            if(float(temp[1])>70) and log:
                grade=70.0
            else:
                grade=(float(temp[1])+float(temp[0]))/2
            arr.append(grade)
        elif 'Up to' in i:
            temp=i.split('to')
            if float(temp[1])>70 and log:
                arr.append(70)
            else:
                arr.append(float(temp[1]))

        else:
            arr.append(0)

    return arr


def normalize_height(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    return (data - mean) / std_dev

def diet_data(data):# Need more optimization
    arr=[]
    for i in data:
        if "Carnivore" in i:
            arr.append(1)
        elif "Herbivore" in i:
            arr.append(2)
        elif "Insectivore" in i:
            arr.append(3)
        elif "Omnivore" in i:
            arr.append(4)
        elif "Carnivore" not in i and "Piscivore" in i:
            arr.append(5)
        elif "Filter Feeder":
            arr.append(6)
    return arr


arr=parse_data(height_data, True)
arr2=parse_data(weight_data)

data_height = np.array(arr)
data_weight = np.array(arr2)
data_diet=diet_data(diet)

def custom_tanh_scaling(x):
    return np.tanh((x - 60) / 10)


def z_score_scale_to_range(arr):
    arr = np.array(arr, dtype=np.float64)
    mean, std = np.mean(arr), np.std(arr)

    if std == 0:
        return np.zeros_like(arr)

    z_scaled = (arr - mean) / std
    return 2 * (z_scaled - np.min(z_scaled)) / (np.max(z_scaled) - np.min(z_scaled)) - 1


scaled_data_height = custom_tanh_scaling(data_height)
scaled_data_weight = custom_tanh_scaling(data_weight)
'''
if len(data_diet)==len(scaled_data_weight) and len(data_diet)==len(weight_data):
    for i in range(len(scaled_data_height)):
        print(y[i],'   ',scaled_data_weight[i],'  ', weight_data[i],'  ',data_diet[i])'''



