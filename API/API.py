import numpy as np
import pandas as pd
import ast
import math
from collections import Counter

QUESTION_LEN = 26  # Number of questions


############## CLASSES
class Item:
    def __init__(self, name, item_type, arr):
        self.name = name
        self.arr = arr  # Numerical representation of an item
        self.item_type = item_type  # Item category
        self.confidence = 1e-5  # Small initial confidence to avoid division by zero


class Question:
    def __init__(self, id, text):
        self.id = id
        self.text = text.strip()


######################################################
# Functions for data loading

def load_items(path):
    items = []
    with open(path) as file:
        for line in file:
            l = line.strip().split(":")
            items.append(Item(l[0], l[1], ast.literal_eval(l[2])))
    return items


def load_questions(path):
    questions = []
    with open(path) as file:
        for id, line in enumerate(file, start=1):
            questions.append(Question(id, line))
    return questions


# Load data
items = load_items("Data/items_dataset.txt")
questions = load_questions("Data/questions_dataset.txt")


def retrive_question(question):
    while True:
        try:
            val = float(input(question.text + " (1: Yes, -1: No, 0.2: Maybe, -0.2: Maybe no, 0: Dont know) "))
            if val in [1, -1, 0.2, -0.2, 0]:
                return val
        except ValueError:
            pass


def evaluate(answer, question):
    for item in items:
        if question.id - 1 >= len(item.arr):  # Out of bounds check
            continue
        similarity = (1 - abs(answer - item.arr[question.id - 1])) / QUESTION_LEN
        item.confidence += similarity


def entropy(answers):
    counter = Counter(answers)
    prob = [c / len(answers) for c in counter.values()]
    return -sum(p * math.log2(p) for p in prob if p > 0)  # Shannon entropy


def find_the_best_question(questions_left):
    best_question = None
    best_entropy = -1

    for question in questions_left:
        answers = [item.arr[question.id - 1] for item in items]
        e = entropy(answers)
        if e > best_entropy:
            best_entropy = e
            best_question = question

    return best_question


def gameplay():
    questions_left = questions.copy()

    while questions_left:
        curr_question = find_the_best_question(questions_left)
        if not curr_question:
            print("There is no more questions")
            break

        # Retrieve next question
        answer = retrive_question(curr_question)

        # Update confidence of each item
        evaluate(answer, curr_question)

        # Remove asked question
        questions_left.remove(curr_question)


        best_item = max(items, key=lambda x: x.confidence)
        print("The biggest probability is", best_item.name , "(with confidence: ",best_item.confidence)

        # If confidence is high enough, end the game
        if best_item.confidence > 0.9:
            print("Finished! API's answer is:",{best_item.name})
            break


#gameplay()


from typing import Union


from flask import Flask, request, jsonify
from flask_cors import CORS  

app = Flask(__name__)
CORS(app) 


@app.route("/submit", methods=["POST"])
def receive_data():
    data = request.json
    answer = data.get("answer")
    print("Primljen odgovor:", answer)

    return jsonify({"message": "Primljeno", "your_answer": answer}), 200


if __name__ == "__main__":
    app.run(debug=True)


@app.route('/game', methods=['POST'])
def form_example():
    return {"Result" : "SS"}

if __name__ == '__main__':
    app.run(debug=True, port=5000)