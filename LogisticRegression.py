import os
from collections import Counter
import time
import math
import csv
import random
import matplotlib.pyplot as plt

 # Return | x |
def pos(x):
    if x < 0: return - x
    return x

def trim (x):
    if      x > 10 :return 10
    elif    x < -10 :return -10
    else :return x

 # Goes into Sigmoid as x
def func(W, x):
    ret = 0
    for i in range(len(x)):
        ret += W[i]*x[i]

    return ret + W[-1]

 # Sigmoid Function 
def sigmoid(x):
  return 1/(1 + math.exp(-x))

 # Make a weight list size N , value x
def init_Weights(N ,x):
    Weights = []  
    for i in range(N+1):
        Weights.append(x)
    return Weights

 # Weight update formula 
  # b = b + alpha * (y – prediction) * prediction * (1 – prediction) * x
def update_weights(X ,y ,Weights ,pred, a):
    mul = a*(y - pred)*pred*(1- pred)
    for i in range(len(Weights)-1):
        w = Weights[i]
        Weights[i] = abs( w + mul*X[i])
    Weights[-1] =  a*(y - pred)*pred*(1- pred)
    return Weights


def makeDict(Messages):

    """
        == Dict for spam/ham-words , each one for subj and body
    """
    spam_dict = Counter()
    ham_dict = Counter()

    spam_dict_subj = Counter()
    ham_dict_subj = Counter()

    # Flospam holds a value of spam likelyhood for each word 
    flospam = Counter()
    flospam_subj = Counter()

    # Loop through every message and fill the dictionary
    for m in Messages:
        if m.Spam is True:
            for word in set(m.Subject):
                spam_dict[word] += 1
        else:
            for word in set(m.Subject):
                ham_dict[word] += 1


    TOTAL_MESS = len(Messages) # Messages in total

    print ('Spam Dictionary has : ', len(spam_dict) ,' entries')

    # Loop through every known word and fill out flospam dict
    for word in spam_dict:
        hc = ham_dict[word]
        sc = spam_dict[word]
        if hc == sc     : x = 0
        elif hc == 0    : x = 1000
        elif sc == 0    : x = -1
        else: 
            if hc > sc:
                x = - sc / (hc+sc)
            else:
                #print (sc ,'/' , hc+sc )
                x = sc / (hc+sc)
                #print (x )
        if sc > 500:
            flospam[word] = x

    # Fill out subj flospam aswell
    for word in spam_dict_subj:
        hc = ham_dict_subj[word]
        sc = spam_dict_subj[word]
        if hc == sc     : x = 0
        elif hc == 0    : x = 100
        elif sc == 0    : x = -1
        else: 
            if hc > sc:
                x = - sc / (hc+sc)
            else:
                #print (sc ,'/' , hc+sc )
                x = sc / (hc+sc)
                #print (x )

        flospam_subj[word] = x

    X_number = 100
    top_spam_words = flospam.most_common(int(X_number/2))
    top_spam_words_subj = flospam_subj.most_common(int(X_number/2))
    
    return flospam,flospam_subj,top_spam_words,top_spam_words_subj

def train(X ,dicts,X_number):


    flospam = dicts [0]
    flospam_subj = dicts [1]
    top_spam_words = flospam.most_common(int(X_number/2))
    top_spam_words_subj = flospam_subj.most_common(int(X_number/2))

    Weights = init_Weights(X_number+2 , 0)
    correct = 0
    accuracy = []
    trainidx = []
    for idx,test_message in enumerate(X):
        x1 = 0
        for word in test_message.Subject:
            x1 += flospam_subj[word]
        x2 = 0
        for word in test_message.Body:
            x2 += flospam[word]

        X = []
        X.append(x1)
        X.append(x2)
        for word in top_spam_words:
            w = word[0]
            if w in test_message.Body:
                X.append(1)
            else:
                X.append(0)
        for word in top_spam_words_subj:
            w = word[0]
            if w in test_message.Body:
                X.append(1)
            else:
                X.append(0)

        x = func(Weights, X)
        x = trim(x)
        
        res = sigmoid(x)

        Pred = False
        y = test_message.Spam
        if res >= 0.5 :
            Pred = True

        if Pred == y:
            correct +=1

        if idx%100 == 0:
            acc = correct/100
            accuracy.append(acc)
            trainidx.append(idx)
            correct = 0

         # Update Weights
        Weights = update_weights(X,y ,Weights ,res,0.01)
    
    
    plt.scatter(trainidx ,accuracy)
    plt.title('ScatterPlot of Accuracy over Training')
    plt.xlabel('Train Index')
    plt.ylabel('Accuracy')
    plt.show()
    return Weights

