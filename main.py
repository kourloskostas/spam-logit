import os
from collections import Counter
import time
import math
import csv
import random
from IO import parseMails , readFiles
import util
import LogisticRegression

 # Message Class 
class Message:
    def __init__(self, Subject , Body , Spam):
        self.Subject = Subject
        self.Body = Body
        self.Spam = Spam


files = readFiles(root ='pu_corpora_public/')    
Messages = parseMails(files = files)  

flospam,flospam_subj,top_spam_words,top_spam_words_subj  = LogisticRegression.makeDict(Messages)

# Shuffle and split into test and train data
random.shuffle(Messages)
Train_Data = Messages[:6000]
Test_Data = Messages[6000:]

Weights = LogisticRegression.train(Train_Data ,[flospam,flospam_subj,top_spam_words,top_spam_words_subj],100)



"""
    Testing
"""
correct = 0
idx = 0
for test_message in Test_Data:
    
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

    #x = w1*x1 + w2*x2
    x = util.func(Weights, X)
    #x = trim(x)

    res = util.sigmoid(util.trim(x))
    
    Pred = False
    y = test_message.Spam
    if res > 0.5 :
        Pred = True
    
    if Pred == y:
        correct +=1
  

    #print (Weights)
    idx +=1

    print (test_message.Spam , ' / ' , Pred)

    print ("Found ", correct , "/" , idx , " - :" , correct*100/idx , '%')
    
print (Weights)
