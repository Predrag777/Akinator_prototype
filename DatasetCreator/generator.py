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
colors=features[:, 2]
top_speed=features[:, 12]

def parse_speed(data):
    print(len(data))
    arr=[]
    for i in data:
        if '-' in i:
            if "(in water)" in i:
                i = i.replace("(in water)", "")
            temp = i.split('-')

            grade = (float(temp[1]) + float(temp[0])) / 2
            if grade>40.0:
                grade=40
            arr.append(grade)
        elif "Not Applicable" in i or "Varies" in i:
            arr.append(0.0)
        else:

            if (float(i) > 40):
                arr.append(40.0)
            else:
                arr.append(float(i))
    return arr

def parse_data(data, log=False):
    min_num=99999
    max_num=-99999
    arr=[]
    for i in data:
        i=i.replace(',','.')
        if '-' in i:

            if "(in water)" in i:
                i=i.replace("(in water)", "")
            temp = i.split('-')
            grade = (float(temp[1]) + float(temp[0])) / 2
            if(grade>70) and log:
                grade=70.0
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


arr =  parse_data(height_data, True)
arr2 = parse_data(weight_data, True)
arr3 = parse_speed(top_speed)
data_height = np.array(arr)
data_weight = np.array(arr2)
data_diet=diet_data(diet)
data_speed=np.array(arr3)


def custom_tanh_scaling(x):
    return np.tanh((x - 60) / 10)


def z_score_scale_to_range(arr):
    arr = np.array(arr, dtype=np.float64)
    mean, std = np.mean(arr), np.std(arr)

    if std == 0:
        return np.zeros_like(arr)

    z_scaled = (arr - mean) / std
    return 2 * (z_scaled - np.min(z_scaled)) / (np.max(z_scaled) - np.min(z_scaled)) - 1


def min_max_scale(arr):
    arr = np.array(arr, dtype=np.float64)  # Osiguravamo da je niz float
    min_val, max_val = np.min(arr), np.max(arr)

    if min_val == max_val:  # Sprečava deljenje nulom
        return np.zeros_like(arr)

    scaled = 2 * (arr - min_val) / (max_val - min_val) - 1
    return scaled

scaled_data_height = custom_tanh_scaling(data_height)
scaled_data_weight = custom_tanh_scaling(data_weight)
scaled_data_speed = min_max_scale(data_speed)



data.iloc[:, 0] = scaled_data_height
data.iloc[:, 1] = scaled_data_weight
data.iloc[:, 12] = scaled_data_speed

data.to_csv("scaled_dataset.csv", index=False)


# If you want to use new created dataset, you need to drop column color, habitants and predators