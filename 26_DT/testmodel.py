# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 20:40:26 2021

@author: vikas
"""


import numpy as np
import pickle
modelcall = pickle.load(open('Model.sav', 'rb'))

ch=1
while(ch==1):
    sl = float(input("Sepel Length"))
    sw = float(input("Sepel Width"))
    pl = float(input("Petal Length"))
    pw = float(input("Petal Width"))
    inp = np.array([sl,sw,pl,pw]).reshape(1,-1)
    pred = modelcall.predict(inp)
    print("Predicted Output is -> ", pred)
    ch = int(input("Enter 1 to continue"))
print("Thank you")

