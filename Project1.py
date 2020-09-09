# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 09:51:13 2020

@author: Mallik siddarth
"""

import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("advertising.csv")
data.head()


fig,axs=plt.subplots(1,3, sharey=True)
data.plot(kind='scatter', x='TV', y='Sales', ax=axs[0], figsize=(15,8))
data.plot(kind='scatter', x='Newspaper', y='Sales', ax=axs[1])
data.plot(kind='scatter', x='Radio', y='Sales', ax=axs[2])


feature_cols=['Radio']
x=data[feature_cols]
y=data.Sales


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x,y)


print(lr.intercept_)
print(lr.coef_)

result=12.235721966369235+0.12443166*60 
print(result)


X_new = pd.DataFrame({'Radio': [data.TV.min(),data.TV.max()]})
X_new.head()


preds = lr.predict(X_new)
preds


data.plot(kind = 'scatter',x = 'Radio',y ='Sales')

plt.plot(X_new,preds,c ='red',linewidth =2)


import statsmodels.formula.api as smf
lm = smf.ols(formula = 'Sales ~ Radio ',data = data).fit()
lm.conf_int()


lm.pvalues
lm.rsquared

feature_cols = ['TV','Radio','Newspaper']
X = data[feature_cols]
Y = data.Sales

lr = LinearRegression()
lr.fit(X,Y)


print(lr.intercept_)
print(lr.coef_)



lm = smf.ols(formula = 'Sales  ~  Radio+TV+Newspaper',data =data).fit()
lm.conf_int()
lm.summary()