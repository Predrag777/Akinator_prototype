

import joblib
class_model = joblib.load("../Model/akinatorModel.pkl")
animal_models = joblib.load("../Model/animal_models.pkl")


new_animal = [[1,0,0,1,0,0,0,1,1,1,0,0,4,1,0,1]]

# Class
predicted_class = class_model.predict(new_animal)[0]




if isinstance(animal_models[predicted_class], str):
    predicted_animal = animal_models[predicted_class]
else:
    predicted_animal = animal_models[predicted_class].predict(new_animal)[0]

print(f"Predviđena klasa: {predicted_class}")
print(f"Predviđena životinja: {predicted_animal}")