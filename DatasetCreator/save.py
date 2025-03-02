import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# PARAMETERS
basic_colors = {"black", "brown", "orange", "white", "gray", "yellow",
                "tan", "gold", "green", "silver", "pink", "red", "blue"}
basic_habitats = {
    "Savanna", "Grassland", "Forest", "Rainforest", "Desert", "Mountain", "Tundra",
    "Ocean", "Freshwater", "Wetlands", "Coastal", "Underground", "Scrubland", "Island"
}

# Function to assign probabilities for colors
def color_prob(color_string):
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

# Function to map habitats to standardized categories
def habitats_assigned(habitat_string):
    if not isinstance(habitat_string, str) or habitat_string.strip() == "":
        return set()

    habitats = {h.strip().lower() for h in habitat_string.replace(" and ", ", ").split(",")}

    mapped_habitats = set()
    for h in habitats:
        if "savanna" in h or "plains" in h:
            mapped_habitats.add("Savanna")
        elif "grassland" in h or "prairie" in h:
            mapped_habitats.add("Grassland")
        elif "forest" in h or "woodland" in h or "broadleaf" in h:
            mapped_habitats.add("Forest")
        elif "rainforest" in h or "tropical" in h:
            mapped_habitats.add("Rainforest")
        elif "desert" in h or "arid" in h or "scrubland" in h:
            mapped_habitats.add("Desert")
        elif "mountain" in h or "alpine" in h:
            mapped_habitats.add("Mountain")
        elif "tundra" in h:
            mapped_habitats.add("Tundra")
        elif "ocean" in h or "marine" in h or "sea" in h:
            mapped_habitats.add("Ocean")
        elif "freshwater" in h or "river" in h or "lake" in h:
            mapped_habitats.add("Freshwater")
        elif "wetland" in h or "marsh" in h or "swamp" in h:
            mapped_habitats.add("Wetlands")
        elif "coast" in h or "coral reef" in h:
            mapped_habitats.add("Coastal")
        elif "underground" in h or "cave" in h or "burrow" in h:
            mapped_habitats.add("Underground")
        elif "island" in h or "madagascar" in h or "galápagos" in h:
            mapped_habitats.add("Island")

    return mapped_habitats

# Data load
data = pd.read_csv("Animal Dataset.csv")

y = np.array(data.iloc[:, 0])  # Animal names
features = np.array(data.iloc[:, 1:])  # Features

# Columns for building new ones
predators = data.iloc[:, 7]  # Predators
habitats = data.iloc[:, 6]  # Habitat
colors = data.iloc[:, 3]  # Color

# Make a list for better handling
predator_lists = [p.split(", ") if isinstance(p, str) else [] for p in predators]
habitats_list = [habitats_assigned(h) for h in habitats]

# Find probability of appearing color for certain animals
color_vectors = [color_prob(p) for p in colors]

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