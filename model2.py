# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

data = pd.read_csv('./advertising.csv')

from sklearn.model_selection import train_test_split
X = data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
y = data['Clicked on Ad']
X_train,X_test,y_Train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)

from sklearn.linear_model import LogisticRegression

lg = LogisticRegression()
lg.fit(X_train,y_Train)


# Saving model to disk
pickle.dump(lg, open('model2.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model2.pkl','rb'))
print(model.predict([[50.52, 31, 72270.88, 171.62, 1]]))



# predictions = lg.predict(X_test)

# from sklearn.metrics import classification_report

# print(classification_report(y_test,predictions))