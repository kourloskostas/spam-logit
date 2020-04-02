import os
from collections import Counter
import time
import math
import csv
import random

 # Message Class 
class Message:
    def __init__(self, Subject , Body , Spam):
        self.Subject = Subject
        self.Body = Body
        self.Spam = Spam

 # Read files from every folder and return a list containing them
def readFiles(root):
    files = [] # List of files to be returned
    for r, d, f in os.walk(root):# r=root, d=directories, f = files
        for file in f:
            if '.txt' in file:
                files.append(os.path.join(r, file))
    return files

 # Input : a list of filenames 
  # Output : a list of Message objects contained within these files
def parseMails(files):
    Messages = [] # Messages list to be returned
    for f in files:
        fname = f.split('/')[-1]
        with open(f,"r") as fopen:
            Subject = fopen.readline().rstrip().split(' ')[1:]
            Body = []
            lamb = (lambda x:not "legit" in x)
            Spam = lamb(fname)
            for line in fopen:
                for word in line.rstrip().split(' '):
                    Body.append(word)  
            Messages.append(Message(Subject ,Body ,Spam ))

    return Messages