import pandas as pd
import numpy as np

# PARAMETERS
# To remove extreme values, we introduced the parameters which would be used as constraints for what would be considered as large, heavy or fast.
# For human:
MAX_HEIGHT=100 # Every animal which height is above 100cm would be considered as big
MAX_WEIGHT=70 # Every animal which weight is above 70kg would be considered as heavy
MAX_LIFESPAN=30 # If lifespan is above 30 years it s long-lived
MAX_SPEED=40 # If top speed is above 40km/h it is fast animal


data = pd.read_csv("Animal Dataset.csv")
questions = data.columns.tolist()[1:]  # Base for questions
y = np.array(data.iloc[:, 0])  # Output


features = np.array(data)  # Make numpy array



height_data=features[:,1]
weight_data=features[:,2]
diet=  features[:,5]
predators=features[:, 7]
colors=features[:, 3]
top_speed=features[:, 13]

def parse_speed(data):
    arr=[]
    for i in data:
        if '-' in i:
            if "(in water)" in i:
                i = i.replace("(in water)", "")
            temp = i.split('-')

            grade = (float(temp[1]) + float(temp[0])) / 2
            if grade>MAX_SPEED:
                grade=MAX_SPEED
            arr.append(grade)
        elif "Not Applicable" in i or "Varies" in i:
            arr.append(0.0)
        else:

            if (float(i) > MAX_SPEED):
                arr.append(MAX_SPEED)
            else:
                arr.append(float(i))
    return arr

def parse_data(data, type_feature):
    arr=[]
    for i in data:
        i=i.replace(',','.')
        if '-' in i:
            if "(in water)" in i:
                i=i.replace("(in water)", "")
            temp = i.split('-')
            grade = (float(temp[1]) + float(temp[0])) / 2
            if(grade>70):
                if "weight" in type_feature:
                    grade=MAX_WEIGHT
                elif "height" in type_feature:
                    grade=MAX_HEIGHT
                else:
                    raise ValueError("You did not use apropriate type_feature parameter. Try with: weight, height, lifespan...")
            arr.append(grade)
        elif 'Up to' in i:
            temp=i.split('to')
            if float(temp[1])>70:
                if "weight" in type_feature:
                    arr.append(MAX_WEIGHT)
                elif "height" in type_feature:
                    arr.append(MAX_HEIGHT)
                else:
                    raise ValueError("You did not use apropriate type_feature parameter. Try with: weight, height, lifespan...")
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


arr =  parse_data(height_data, "height")
arr2 = parse_data(weight_data, "weight")
arr3 = parse_speed(top_speed)
data_height = np.array(arr)
data_weight = np.array(arr2)
data_diet=diet_data(diet)
data_speed=np.array(arr3)

# SCALING METHODS
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

scaled_data_height = min_max_scale(data_height)
scaled_data_weight = min_max_scale(data_weight)
scaled_data_speed = min_max_scale(data_speed)



data.iloc[:, 1] = scaled_data_height
data.iloc[:, 2] = scaled_data_weight
data.iloc[:, 13] = scaled_data_speed


data.to_csv("scaled_dataset.csv", index=False)


# If you want to use new created dataset, you need to drop column color, habitants and predators