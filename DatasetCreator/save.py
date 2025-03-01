import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# Data load
data = pd.read_csv("Animal Dataset.csv")

y = np.array(data.iloc[:, 0])  # Animals names
features = np.array(data.iloc[:, 1:])  # Features

# Colones for building new ones
predators = data.iloc[:, 7]  # Predators
habitats = data.iloc[:, 6]  # Habitants
colors = data.iloc[:, 3]  # Colors

# Convert to strings
predator_lists = [p.split(", ") if isinstance(p, str) else [] for p in predators]
colors_list = [p.split(", ") if isinstance(p, str) else [] for p in colors]
habitats_list = [p.split(", ") if isinstance(p, str) else [] for p in habitats]

mlb_predators = MultiLabelBinarizer()
mlb_colors = MultiLabelBinarizer()
mlb_habitats = MultiLabelBinarizer()

predator_encoded = mlb_predators.fit_transform(predator_lists)
colors_encoded = mlb_colors.fit_transform(colors_list)
habitats_encoded = mlb_habitats.fit_transform(habitats_list)

# Create Dataframe
predator_df = pd.DataFrame(predator_encoded, columns=mlb_predators.classes_)
colors_df = pd.DataFrame(colors_encoded, columns=mlb_colors.classes_)
habitats_df = pd.DataFrame(habitats_encoded, columns=mlb_habitats.classes_)

# Merging
new_data = pd.concat([data, predator_df, colors_df, habitats_df], axis=1)
'''print(predator_df.columns.tolist())
print(habitats_df.columns.tolist())
print(colors_df.columns.tolist())'''

features = new_data.iloc[:, 1:].values

questions = new_data.columns.tolist()


new_data.to_csv("new_animal_dataset.csv", index=False)










