import numpy as np
import pandas as pd
import math
from collections import Counter

# PARAMETERS
# TRESHOLDS
THRESHOLD_YES = 0.7
THRESHOLD_NO = -0.7

class Animal:
    def __init__(self, name, arr):
        self.name = name
        self.arr = arr  # Numerical representation of an item
        self.confidence = 1e-5  # Small initial confidence to avoid division by zero

class Question:
    def __init__(self, id, text):
        self.id = id
        self.text = text.strip()

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


def evaluate(answer, question):
    print(answer, "    ", question)
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

class Node:
    def __init__(self, question=None, character=None):
        self.left = None
        self.middle_left = None
        self.middle_right = None
        self.right = None
        self.character = character
        self.question = question

def build_tree(animals, questions):
    if not questions:
        most_common = Counter([a.name for a in animals]).most_common(1)
        return Node(character=most_common[0][0]) if most_common else None

    max_question_val = 0
    best_question = None

    for q in questions:
        gain = information_gain(animals, q)
        if gain > max_question_val:
            max_question_val = gain
            best_question = q

    if not best_question: # If there is no best questions, we need to return the most suited animal. It would be the end of game
        most_common = Counter([a.name for a in animals]).most_common(1)
        return Node(character=most_common[0][0]) if most_common else None

    remaining_questions = [q for q in questions if q != best_question]  # Remove used questions

    left_group = []  # For Yes (above THRESHOLD_YES)
    middle_left_group = []  # For maybe yes (Just above 0)
    middle_right_group = []  # For maybe no (Just lower than 0)
    right_group = []  # For No (below THRESHOLD_NO)

    for animal in animals:
        val = animal.arr[best_question.id - 1]
        if val > THRESHOLD_YES:
            left_group.append(animal)
        elif 0 < val <= THRESHOLD_YES:
            middle_left_group.append(animal)
        elif THRESHOLD_NO <= val < 0:
            middle_right_group.append(animal)
        else:  # val < THRESHOLD_NO
            right_group.append(animal)

    if not (left_group or middle_left_group or middle_right_group or right_group):
        most_common = Counter([a.name for a in animals]).most_common(1)
        return Node(character=most_common[0][0]) if most_common else None

    # Check how size of samples would changed during time
    '''print(len(left_group),'    ',len(middle_left_group),'    ',len(animals))
    print(len(right_group), '    ', len(middle_right_group), '    ', len(animals))'''

    # Node is a question, and each answer will lead us to one of four animal's groups

    node = Node(question=best_question)

    node.left = build_tree(left_group, remaining_questions) if left_group else None                         # Animals which have big confidence for answer to be YES
    node.middle_left = build_tree(middle_left_group, remaining_questions) if middle_left_group else None    # Animals which have big confidence for answer to be MAYBE YES
    node.middle_right = build_tree(middle_right_group, remaining_questions) if middle_right_group else None # Animals which have big confidence for answer to be MAYBE NO
    node.right = build_tree(right_group, remaining_questions) if right_group else None                      # Animals which have big confidence for answer to be NO

    return node

def answer_question(node):  # Answer the questions. By answering , algorithm would led Akinator through the tree to correct answer
    if node.character:
        return node.character

    answer = float(input(node.question.text + " (1: Yes, -1: No, 0.5: Maybe Yes, -0.5: Maybe No, 0: Dont know) "))
    if answer == 1: #GO LEFT
        return answer_question(node.left) if node.left else answer_question(node.middle_left or node.right)
    elif answer == 0.5:#GO MIDDLE LEFT
        return answer_question(node.middle_left) if node.middle_left else answer_question(node.left or node.right)
    elif answer == -0.5:#GO MIDDLE RIGHT
        return answer_question(node.middle_right) if node.middle_right else answer_question(node.right or node.left)
    elif answer == -1:#GO RIGHT
        return answer_question(node.right) if node.right else answer_question(node.middle_right or node.left)
    else:#
        return answer_question(node.middle_left) if node.middle_left else answer_question(node.left or node.right)

def gameplay():
    root = build_tree(animals, questions)
    result = answer_question(root)
    print(f"My answer is {result}")

gameplay() # Run gameplay



# Do not use API part, because algorithm is in testing phase!
'''from typing import Union


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
    app.run(debug=True, port=5000)'''

