import pandas as pd
import numpy as np
import random

# PARAMETERS
# To remove extreme values, we introduced the parameters which would be used as constraints for what would be considered as large, heavy or fast.
# For human:
MAX_HEIGHT=100 # Every animal which height is above 100cm would be considered as big
MAX_WEIGHT=70 # Every animal which weight is above 70kg would be considered as heavy
MAX_LIFESPAN=30 # If lifespan is above 40 years it is long-lived
MAX_SPEED=40 # If top speed is above 40km/h it is fast animal
CONSERVATION_STATUS = { # Rating the conservation status
                        "Least Concern": 10,
                        "Near Threatened": 20,
                        "Vulnerable": 50,
                        "Endangered": 50,
                        "Critically Endangered": 50,
                        "Extinct": 10,
                        "Extinct (around 58 million years ago)": 10,
                        "Extinct (around 4,000 years ago)": 10,
                        "Not Evaluated": 10,
                        "Data Deficient": 10,
                        "Not Applicable": 10,
                        "Varies": 20
}




# Load data
data = pd.read_csv("new_animal_dataset.csv")
questions = data.columns.tolist()[1:]  # Base for questions
y = np.array(data.iloc[:, 0])  # Output

features = np.array(data)  # Make numpy array of features for easier data manipulation


# Get data
height_data=features[:,1]
weight_data=features[:,2]
diet=  features[:,5]
predators=features[:, 7]
colors=features[:, 3]
lifespan=features[:, 4]
top_speed=features[:, 13]
conservation=features[:, 10]

# This function need to become part of the parse_data function
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
    years_months=1.0 # For animals which lifespan is several months
    years_weeks=1.0 # For animals which lifespan is several weeks
    years_days=1.0 # For animals which lifespan is several days

    # Constraints for the maximum values
    constraint=0
    if "weight" in type_feature:
        constraint=MAX_WEIGHT
    elif "height" in type_feature:
        constraint = MAX_HEIGHT
    elif "lifespan" in type_feature:
        constraint = MAX_LIFESPAN

    for i in data:
        i=i.replace(',','.')
        if '-' in i:
            if "(in water)" in i:
                i=i.replace("(in water)", "")
            if "weeks" in i:
                years_weeks=7/365   # One week in one year
                i=i.replace("weeks", "")
            if "months" in i:
                years_months=1/12 # One month in one year
                i=i.replace("months", "")
            if "days" in i:
                years_days=1/365 # One day in one year
                i=i.replace("days", "")
            if "years" in i:
                i=i.replace("years", "")
            temp = i.split('-')
            rating = (float(temp[1]) + float(temp[0])) / 2

            if "lifespan" in type_feature:
                rating=rating*years_months*years_weeks*years_days

            if(rating>constraint):
                if "weight" in type_feature:
                    rating=MAX_WEIGHT
                elif "height" in type_feature:
                    rating=MAX_HEIGHT
                elif "lifespan" in type_feature:
                    rating=MAX_LIFESPAN
                else:
                    raise ValueError("You did not use apropriate type_feature parameter. Try with: weight, height, lifespan...")
            arr.append(rating)
        elif 'Up to' in i:
            if "weeks" in i:
                years_weeks=7/365   # One week in one year
                i=i.replace("weeks", "")
            if "months" in i:
                years_months=1/12 # One month in one year
                i=i.replace("months", "")
            if "days" in i:
                years_days=1/365 # One day in one year
                i=i.replace("days", "")
            if "years" in i:
                i=i.replace("years", "")
            temp=i.split('to')
            curr_val = float(temp[1])
            if "lifespan" in type_feature:
                curr_val=years_months*years_weeks*years_days
            if curr_val>constraint:
                if "weight" in type_feature:
                    arr.append(MAX_WEIGHT)
                elif "height" in type_feature:
                    arr.append(MAX_HEIGHT)
                elif "lifespan" in type_feature:
                    arr.append(MAX_LIFESPAN)
                else:
                    raise ValueError("You did not use apropriate type_feature parameter. Try with: weight, height, lifespan...")
            else:
                arr.append(curr_val)

        else:
            arr.append(0)

    return arr


def parse_conservation_status(data):
    return [random.randint(CONSERVATION_STATUS.get(status, 30) - 5, CONSERVATION_STATUS.get(status, 30)) for status in data]


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

# Parse data
arr =  parse_data(height_data, "height")
arr2 = parse_data(weight_data, "weight")
arr3 = parse_speed(top_speed)
arr4 = parse_data(lifespan, "lifespan")
arr5 = parse_conservation_status(conservation)

# Convert to numpy
data_height = np.array(arr)
data_weight = np.array(arr2)
data_diet=diet_data(diet)
data_speed=np.array(arr3)
data_lifespan = np.array(arr4)
data_conservation = np.array(arr5)

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



# Scaled data to introduce Fuzzy-logic where a user would be able to answer not just with answers Yes (prob: 100%) or No(prob: -100%)
# but with Maybe(prbo: +50% or -50%) and dont know (NEUTRAL: 0)
scaled_data_height = min_max_scale(data_height)
scaled_data_weight = min_max_scale(data_weight)
scaled_data_speed = min_max_scale(data_speed)
scaled_data_lifespan = min_max_scale(data_lifespan)
scaled_data_conservation = min_max_scale(data_conservation)


# Replace old data with scaled data in certain columns:
data.iloc[:, 1] = scaled_data_height # Height
data.iloc[:, 2] = scaled_data_weight # Weight
data.iloc[:, 13] = scaled_data_speed # Top-speed
data.iloc[:, 4] = scaled_data_lifespan # Lifespan
data.iloc[:, 10] = scaled_data_conservation # Conservation status


# Drop columns
data = data.drop(data.columns[3], axis=1) # Color
data = data.drop(data.columns[4], axis=1) # Habitats
data = data.drop(data.columns[4], axis=1) # Predators
data = data.drop(data.columns[4], axis=1) # Predators
data = data.drop(data.columns[4], axis=1) # Top speed


data.to_csv("scaled_dataset.csv", index=False) # Create new dataset which would be used