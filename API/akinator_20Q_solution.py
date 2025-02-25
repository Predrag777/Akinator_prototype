import numpy as np
import pandas as pd


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
            items.append(Item(l[0], l[1], l[2]))
    return items
def load_questions(path):
    questions=[]
    id=1
    with open(path) as file:
        for line in file:
            questions.append(Question(id, line))


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

def evalouate(answer, question):# Make an evaloation for the certain answer on the certain question
    return (1 - abs(answer - items.arr[question.id - 1])) / QUESTION_LEN


curr_question=questions[0]
answer =1

print(curr_question.id, curr_question.text, answer)
print(evalouate(answer, curr_question))


