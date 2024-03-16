import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string
from sklearn.metrics import accuracy_score
from joblib import dump,load
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
import warnings 
warnings.filterwarnings('ignore')

# read data
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")
df=pd.read_csv("manual_testing.csv")
x = df["text"]
y = df["class"]
# spliting train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30,random_state=123)
LR = LogisticRegression()
LR.fit(xv_train,y_train)
# model testing
def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    

    return print("\n\nLR Prediction: {} ".format(output_lable(pred_LR[0])))

news = str(input())
manual_testing(news)
