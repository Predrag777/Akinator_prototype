import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# PARAMETERS
basic_colors = {"black", "brown", "orange", "white", "gray", "yellow",
                "tan", "gold", "green", "silver", "pink", "red", "blue"}


# Assigned probability for the colors
def assign_color_probabilities(color_string):
    if not isinstance(color_string, str) or color_string.strip() == "":
        return {}

    colors = [c.strip().lower() for c in color_string.replace(" with ", ", ").split(",")]

    # Pick the basic colors
    filtered_colors = [c for c in colors if c in basic_colors]

    if not filtered_colors:
        return {} 

    # Normalizing
    num_colors = len(filtered_colors)
    color_distribution = {color: 1 / num_colors for color in filtered_colors}

    return color_distribution


# Data load
data = pd.read_csv("Animal Dataset.csv")

y = np.array(data.iloc[:, 0])  # Animals names
features = np.array(data.iloc[:, 1:])  # Features

# Colones for building new ones
predators = data.iloc[:, 7]  # Predators
habitats = data.iloc[:, 6]  # Habitat
colors = data.iloc[:, 3]  # Color

# Make a list for better handling
predator_lists = [p.split(", ") if isinstance(p, str) else [] for p in predators]
habitats_list = [p.split(", ") if isinstance(p, str) else [] for p in habitats]

# Find probability of appearing color for certain animal based on the description for color
color_vectors = [assign_color_probabilities(p) for p in colors]

# Dataframe for colors
color_columns = sorted(basic_colors)  
color_data = pd.DataFrame([{col: cv.get(col, 0) for col in color_columns} for cv in color_vectors])


mlb_predators = MultiLabelBinarizer()
mlb_habitats = MultiLabelBinarizer()

predator_encoded = mlb_predators.fit_transform(predator_lists)
habitats_encoded = mlb_habitats.fit_transform(habitats_list)

# Create Dataframe
predator_df = pd.DataFrame(predator_encoded, columns=mlb_predators.classes_)
habitats_df = pd.DataFrame(habitats_encoded, columns=mlb_habitats.classes_)

# Merging
new_data = pd.concat([data, predator_df, color_data, habitats_df], axis=1)


new_data.to_csv("new_animal_dataset.csv", index=False)