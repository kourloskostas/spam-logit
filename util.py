import os
from collections import Counter
import time
import math
import csv
import random

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
def update_weights(X,Weights, y ,pred, a):
    mul = a*(y - pred)*pred*(1- pred)
    for i in range(len(Weights)-1):
        w = Weights[i]
        Weights[i] = abs( w + mul*X[i])
    Weights[-1] =  a*(y - pred)*pred*(1- pred)
    return Weights


def train(Train_Data):
    for test_message in Train_Data:
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

        x = util.func(Weights, X)
        x = util.trim(x)
        
        res = util.sigmoid(x)
    
        Pred = False
        y = test_message.Spam
        if res >= 0.5 :
            Pred = True

        Weights = update_weights(X,Weights, y ,res,0.01)
    return Weights

