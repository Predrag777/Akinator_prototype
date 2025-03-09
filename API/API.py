import numpy as np
import pandas as pd
import math
from collections import Counter


############## CLASSES
class Animal:
    def __init__(self, name, arr):
        self.name = name
        self.arr = arr  # Numerical representation of an item
        self.confidence = 1e-5  # Small initial confidence to avoid division by zero


class Question:
    def __init__(self, id, text):
        self.id = id
        self.text = text.strip()

class Node:
    def __init__(self, question=None, character=None):
        self.question = question
        self.character = character
        self.left = None
        self.right = None


######################################################
# Functions for data loading
def load_animals(path):
    arr = []
    data = pd.read_csv(path)
    animal_names = np.array(data.iloc[:, 0])
    data = data.drop(data.columns[16], axis=1)  # Remove color column
    animal_features = np.array(data.iloc[:, 1:])
    for i in range(len(animal_names)):
        arr.append(Animal(animal_names[i], animal_features[i]))
    return arr


def load_questions(path):
    questions = []
    with open(path) as file:
        for id, line in enumerate(file, start=1):
            questions.append(Question(id, line))
    return questions


animals = load_animals("Data/DatasetLLM.csv")
questions = load_questions("Data/questions_dataset3.txt")
QUESTION_LEN = len(questions)

def answer_question(question):
    while True:
        try:
            val = float(input(question.text + " (1: Yes, -1: No, 0.2: Maybe, -0.2: Maybe no, 0: Dont know) "))
            if val in [1, -1, 0.2, -0.2, 0]:
                return val
        except ValueError:
            pass

def evaluate(answer, question):
    print(answer,"    ",question)
    for item in animals:
        if question.id - 1 >= len(item.arr):  # Out of bounds check
            continue
        similarity = (1 - abs(answer - item.arr[question.id - 1])) / QUESTION_LEN
        item.confidence += similarity


def entropy(answers):
    counter = Counter(answers)
    prob = [c / len(answers) for c in counter.values()]
    return -sum(p * math.log2(p) for p in prob if p > 0)  # Shannon entropy


def information_gain(animals, question):
    answers = [animal.arr[question.id - 1] for animal in animals]
    initial_entropy = entropy(answers)

    grouped_data = {}
    for animal in animals:
        ans = animal.arr[question.id - 1]
        if ans not in grouped_data:
            grouped_data[ans] = []
        grouped_data[ans].append(animal)

    weighted_entropy = sum(
        (len(group) / len(animals)) * entropy([a.arr[question.id - 1] for a in group])
        for group in grouped_data.values()
    )

    return initial_entropy - weighted_entropy


def build_decision_tree(animals, questions):
    if len(set(a.name for a in animals)) == 1:
        return Node(character=animals[0].name)

    best_question = max(questions, key=lambda q: information_gain(animals, q), default=None)


    left = [a for a in animals if a.arr[best_question.id - 1] <= 0]
    right = [a for a in animals if a.arr[best_question.id - 1] > 0]

    if not left or not right:
        most_common = Counter([a.name for a in animals]).most_common(1)
        return Node(character=most_common[0][0]) if most_common else None

    new_node = Node(question=best_question)
    other_questions = [q for q in questions if q != best_question]

    new_node.left = build_decision_tree(left, other_questions)
    new_node.right = build_decision_tree(right, other_questions)

    return new_node



def answer_question(node):
    if node.character:
        return node.character
    answer = float(input(node.question.text + " (1: Yes, -1: No, 0: Maybe, -0.2: Maybe yes, 0.2: Maybe no) "))
    if answer >= 0:
        return answer_question(node.right)
    else:
        return answer_question(node.left)

def gameplay():
    root = build_decision_tree(animals, questions)
    result = answer_question(root)
    print(f"My answer is {result}")

gameplay()


from typing import Union


from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

questions_left=questions.copy()



#gameplay()

@app.route("/question", methods=["GET"])
def send_question():
    ss=find_the_best_question(questions_left)

    return jsonify({"Question": f"{ss.text}", "Question_id":f"{ss.id}"})



@app.route("/submit", methods=["POST"])
def receive_data():
    data = request.json
    answer = float(data.get("answer"))
    question = data.get("question")
    # Build object question
    curr_question = next((q for q in questions_left if q.text == question), None)

    evaluate(float(answer),curr_question)
    questions_left.remove(curr_question)
    best_item = max(items, key=lambda x: x.confidence)

    return jsonify({"message": "Primljeno", "Item": best_item.name, "confidence": best_item.confidence}), 200



if __name__ == '__main__':
    app.run(debug=True, port=5000)

