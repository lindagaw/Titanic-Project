import csv
import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt

all_passengers = []
passengers_without_age = []
passengers_with_age = []

with open('train.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile)
    '''
    0-PassengerId, 1-Survived, 2-Pclass, 3-Name, 5-Sex, 6-Age, 7-SibSp, 8-Parch, 9-Ticket, 10-Fare, 11-Cabin, 12-Embarked
    '''
    for item in reader:
        for sub in item:
                sub = sub.strip()

        item[0] = int( item[0] )
        item[1] = float( item[1] )
        item[2] = float( item[2])
        item[6] = float( item[6])
        item[7] = float( item[7])
        all_passengers.append(item)


print(all_passengers[0])

## Percentage of Survival
lived = died = 0
## Percentage of Ticket Class
first = second = third = 0
## Percentage of Sex
female = male = 0
## Mean and SD of Age
ages = []
## Mean and SD of Sibs on ships
sibs = []
## Percentage of Embark
C = Q = S = 0



for passenger in all_passengers:
    if( passenger[1] == 1 ):
        lived += 1
    else:
        died += 1

    if( int(passenger[2]) == 1 ):
        first += 1
    elif( int(passenger[2]) == 2 ):
        second += 1
    else:
        third += 1

    if( passenger[4] == 'female' ):
        female += 1
    else:
        male += 1

    if( type(passenger[5]) == type(1) ):
        ages.append( passenger[5])
    if( type(passenger[6]) == type(1) ):
        sibs.append( passenger[6])

    if( passenger[11] == 'S' ):
        S += 1
    elif( passenger[11] == 'Q' ):
        Q += 1
    else:
        C += 1

## Percentage of Survival
def percent_survival():
    survival = float(lived)/( float(lived) + float(died) )
    print( str(lived) + " people survived and " + str(died) + " people died. "  )
    print( "Percentage of survival is " + str(survival))
    fig = plt.figure(1, figsize=(10, 8))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data_to_plot)
    ax.set_xticklabels(['Survived', "Didn't survived"])
## Percentage of Ticket Class
def percent_ticketclass():
    first_class = float(first)/( float(third) + float(second) + float(third) )
    second_class = float(second)/( float(third) + float(second) + float(third) )
    third_class = float(third)/( float(third) + float(second) + float(third) )
    print( str(first) + " people were in the first class. They are " + str(first_class) + " of the overall passengers." )
    print( str(second) + " people were in the second class. They are " + str(second_class) + " of the overall passengers." )
    print( str(third) + " people were in the third class. They are " + str(third_class) + " of the overall passengers." )
    fig = plt.figure(1, figsize=(10, 8))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data_to_plot)
    ax.set_xticklabels(['1st', "2nd", '3rd'])
## Percentage of Sex
def percent_sex():
    men = float(male)/(float(male) + float(female))
    women = float(female)/(float(male) + float(female))
    print( str(women) + " of them was female. There were " + str(women) + " of them.")
    print( str(men) + " of them was female. There were " + str(men) + " of them.")
## Mean and SD of Age
def age_mean_sd():
    print( "The mean of their ages is " + str( numpy.mean(ages) ) )
    print( "The SD of their ages is " + str( numpy.std(ages) ) )
## Mean and SD of Sibs on ships
def sibs_mean_sd():
    print( "The mean of their siblings/spouses on the ship is " + str(numpy.mean(sibs)) )
    print( "The SD of their siblings/spouses on the ship is " + str(numpy.std(sibs)) )
## Percentage of Embark
def percent_embark():
    Q_percent = float(Q)/(float(S) + float(C) + float(Q))
    S_percent = float(S)/(float(S) + float(C) + float(Q))
    C_percent = float(C)/(float(S) + float(C) + float(Q))
    print( str(C_percent) + " of them embarked the ship from Cherbourg.")
    print( str(Q_percent) + " of them embarked the ship from Queenstown.")
    print( str(S_percent) + " of them embarked the ship from Southampton.")

percent_survival()
percent_ticketclass()
percent_sex()
age_mean_sd()
sibs_mean_sd()
percent_embark()
