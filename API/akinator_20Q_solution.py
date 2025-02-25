import numpy as np
import pandas as pd
import ast
import math
from collections import Counter


QUESTION_LEN=26 #We defined 26 questions in the /Data/questions_dataset.txt


##############CLASSES
class Item:
    def __init__(self, name, item_type, arr):
        self.name=name
        self.arr=arr       # Numerical representation of an item
        self.item_type=item_type # Which type item belongs
        self.confidence=0  #Default confidence is 0

class Question:
    def __init__(self, id, text):
        self.id=id
        self.text=text
######################################################

#Functions for data loading
def load_items(path):
    items=[]
    with open(path) as file:
        for line in file:
            l=line.split(":")
            items.append(Item(l[0], l[1], ast.literal_eval(l[2])))
    return items
def load_questions(path):
    questions=[]
    id=1
    with open(path) as file:
        for line in file:
            questions.append(Question(id, line))
            id+=1

    return questions
### Load data
items=load_items("Data/items_dataset.txt")
questions=load_questions("Data/questions_dataset.txt")




#Functions
def retrive_question(question):
    while True:
        try:
            val = float(input(question.text + " ")) #Provide an answer
            if val in [1, -1, 0.2, -0.2, 0]:  #True, false, maybe, maybe not, don't know
                return val
        except ValueError:
            pass


def evaluate(answer, question):  # When player answer the question, program will update confidence of each item
    for item in items:
        if question.id - 1 >= len(item.arr):    # Out of bounds
            return 0
        return (1 - abs(answer - item.arr[question.id - 1])) / QUESTION_LEN


def entropy(answers):
    counter=Counter(answers)
    prob=[c/len(answers) for c in counter]

    return -sum(p*math.log2(p) for p in prob if p>0)   # Shannoa entropy https://en.wikipedia.org/wiki/Entropy_(information_theory)


# Best question would be retrived based on the entropy level
def retrive_next_question():
    best_question = None
    best_entropy = -1

    for question in questions:
        answers = [item.arr[question.id - 1] for item in items]
        e = entropy(answers)

        if e > best_entropy:
            best_entropy = e
            best_question = question

    return best_question



def gameplay():
    
    while True:



curr_question=questions[0]
answer =1


entropy([1,-1,0,0.5,-0.5])