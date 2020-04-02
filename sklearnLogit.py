import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

 # Read dataframe and drop NaN
df = pd.read_csv('mail_list.csv', delimiter=',',names=["Subj" , "Body" , "Spam"])
df.dropna()

 # Vectorize 
vectorizer = TfidfVectorizer()
subj_mat = vectorizer.fit_transform(df["Subj"].values.astype('U'))
body_mat = vectorizer.fit_transform(df["Body"].values.astype('U'))
joined_matrix = subj_mat + body_mat

print (joined_matrix)

 # Split df to train and test
X_train, X_test, Y_train, y_test = train_test_split(joined_matrix,df["Spam"])

 # Logit Clasifier + Train Model
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)

predictions = classifier.predict(X_test)
print ("Made ", len(predictions) , "predictions")
print(accuracy_score(y_test,predictions))




"""
xt = []

for x in X_train:
    s = 0
    for y in x:
        s += y

    xt.append(s)


plt.scatter(xt,Y_train)
plt.title("Logistic Regression")
plt.xlabel('Mean Vecfloat')
plt.ylabel('Spam (1:Spam, 0:Ham)')
plt.show()
"""
"""
with open("mail_list.csv" ,"w+" ) as f:
    csv_writer = csv.writer(f , delimiter=",")


    for m in Messages:
        row = []
        subj = ''
        body = ''
        for w in m.Subject:
            subj = subj + ' ' + w

        for w in m.Subject:
            body = body + ' ' + w
        
        spam = m.Spam

        if spam is True : spam = 1
        else: spam = 0

        row.append(subj)
        row.append(body)
        row.append(spam)

        csv_writer.writerow(row)
"""