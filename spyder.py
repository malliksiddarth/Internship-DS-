# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



ironman=pd.read_csv('TSLA Historical Data.csv', usecols=[0,1,2,3,4])



POHL_avg=ironman[['Price','Open','High','Low']].mean(axis=1)


udv=np.arange(1,len(ironman)+1,1)

plt.plot(udv,POHL_avg,'r',label='graph')