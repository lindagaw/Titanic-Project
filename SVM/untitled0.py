#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 10:24:53 2017

@author: marzieh
"""

import csv

all_passengers = []
passengers_without_age = []
passengers_with_age = []
with open('test.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile)
    '''
    0-PassengerId, 1-Survived, 2-Pclass, 3-Name, 4-Sex, 5-Age, 6-SibSp, 7-Parch, 8-Ticket, 9-Fare, 10-Cabin, 11-Embarked
    '''
    for item in reader:
        item[1] = float( item[1] )
        if item[5] == '':
            passengers_without_age.append( item )
        else:
            item[5] = float( item[5] )
            passengers_with_age.append( item )

        all_passengers.append(item)

'''
Goal: calculate P( Y = y | X = x)
1. Calcuate count( X = survival, Y = age_group) => X: did this person actually survive
2. For all Y = y, calculate P(y)
3. Calculate count( X = 1, Y = y ) + count( X = 0, Y = y)

4. Summing over y', calculate count(X, Y = y') * P(Y=y')/ count(1, Y = y') + count(0, Y = y')
    => X: did this person actually survive

5. Get the Y = y that gives the highest probability
'''
# Step 0: Implement count( X = survival, Y = age)
# X = {0.0, 1.0}; Y = {0.0, 10.0, 20.0 ... 90.0}
def count ( survival, age ):
    value = 0
    for passenger in passengers_with_age:
        if passenger[1] == survival and ( passenger[5] >= age and passenger[5] <= age + 9 ):
            value += 1
    return value

# Step 1: Divide into 10 different age groups:
age_groups = [0, 0, 0, 0, 0, 0, 0, 0, 0]
age_probs = []
for passenger in passengers_with_age:
    if passenger[5] <= 9:
        age_groups[0] += 1
    elif passenger[5] <= 19:
        age_groups[1] += 1
    elif passenger[5] <= 29:
        age_groups[2] += 1
    elif passenger[5] <= 39:
        age_groups[3] += 1
    elif passenger[5] <= 49:
        age_groups[4] += 1
    elif passenger[5] <= 59:
        age_groups[5] += 1
    elif passenger[5] <= 69:
        age_groups[6] += 1
    elif passenger[5] <= 79:
        age_groups[7] += 1
    elif passenger[5] <= 89:
        age_groups[8] += 1
    else:
        print("Unable to classify age for this passenger.")

for age in age_groups:
    prob = age/len(passengers_with_age)
    age_probs.append(prob)

# Now we have P(Y = y) for all possible y's
# Step 2: count(X, Y = y) * P( Y = y ) for each y. Y = {0.0, 10.0, 20.0 ... 90.0}
def prob_survival( x, y ):
    # For terms in the numerator
    numerator_up = float(count(x, y) * age_probs[ int( y/10 ) ])
    denominator_up = float(count(1.0, y) + count(0.0, y))
    up = numerator_up / denominator_up
    # For terms in the denominator
    down = 0.0
    for age in range(0, 90, 10):
        age = float(age)
        numerator_down = float(count(x, age) * age_probs[ int( age/10 ) ])
        denominator_down = float(count(1.0, age) + count(0.0, age))

        down += numerator_down/denominator_down

    return up/down

def most_possible_age( passenger ):
    this_passenger_probs = []
    for age in range(0, 90, 10):
        age = float(age)
        prob = prob_survival(passenger[1], age)
        this_passenger_probs.append(prob)

    candidate = 0
    candidate_prob = 0.0
    counter = 0
    for prob in this_passenger_probs:
        if prob > candidate_prob:
            candidate_prob = prob
            candidate = counter
        counter += 1

    return float(candidate * 10.0)

'''
for passenger in all_passengers:
    if passenger[5] == '':
        passenger[5] = most_possible_age( passenger )
'''

survived = []
died = []

for age in range(0, 90, 10):
    yes = prob_survival(1.0, float(age) )
    no = prob_survival(0.0, float(age) )
    msg = ''
    if yes > no:
        msg = "survived with prob " + str(yes)
        survived.append([age, yes])
    else:
        msg = "didn't survive with prob " + str(no)
        died.append([age, no])
    print( str(age) + ": " + msg)

print("===================================================")
survived.sort(key=lambda x: x[1])
died.sort(key=lambda x: x[1])

predicted_age_if_survived = survived[ len(survived)-1 ][0]
predicted_age_if_died = died[ len(died)-1 ][0]

print("If the target survived, his/her predicted age is " + str(predicted_age_if_survived) )
print("If the target died, his/her predicted age is " + str(predicted_age_if_died) )

for passenger in all_passengers:
    if passenger[5] == '':
        if passenger[1] == 1.0: # survived
            passenger[5] = predicted_age_if_survived
        else:
            passenger[5] = predicted_age_if_died


with open('preprocessed.csv','wt') as file:
    for line in all_passengers:
        file.write(str(line))
        file.write('\n')
