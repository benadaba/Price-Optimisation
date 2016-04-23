# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 21:12:54 2016

@author: Bernard
"""

# -*- coding: utf-8 -*-

import pandas as pd
import os
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics
import statsmodels.formula.api as smf


os.chdir("C:\TREES")


"""
Data Engineering and Analysis
"""
#Load the dataset

AH_data = pd.read_csv('optimum_pricePOS.csv')
print(AH_data)

#drop all NaN values
data_clean1 = AH_data.dropna()

data_clean1["Isweekday"]= data_clean1["Isweekday"]
data_clean1["Isweekend"]= data_clean1["Isweekend"]
data_clean1["Isworkday"]= data_clean1["Isworkday"]
data_clean1["IsBankHoliday"]= data_clean1["IsBankHoliday"]
data_clean1["ProductSalePrice"]= data_clean1["ProductSalePrice"]
data_clean1["Profit"]= data_clean1["Profit"]
data_clean1["OrderItemQuantity"]= data_clean1["OrderItemQuantity"]
data_clean1["SupPurchasePrice"]= data_clean1["SupPurchasePrice"]




###
#data management
#recoding values
#first we define a dictionary of what we want to do
recode1 = {"YES":1, "NO":0}

#if weekday code as 1 if not weekend code as 0
data_clean1["IsweekdayRC"]= data_clean1["Isweekday"].map(recode1)
c1 = data_clean1["IsweekdayRC"].value_counts(sort = False)
print(c1)

#if weekend code as 1 if not weekend code as 0
data_clean1["IsweekendRC"]= data_clean1["Isweekend"].map(recode1)
c2 = data_clean1["IsweekendRC"].value_counts(sort = False)
print(c2)

#if workday code as 1 if not weekend code as 0
data_clean1["IsworkdayRC"]= data_clean1["Isworkday"].map(recode1)
c3 = data_clean1["IsworkdayRC"].value_counts(sort = False)
print(c3)

#if bankHoliday code as 1 if not weekend code as 0
data_clean1["IsBankHolidayRC"]= data_clean1["IsBankHoliday"].map(recode1)
c4 = data_clean1["IsBankHolidayRC"].value_counts(sort = False)
print(c4)

# count values in profit
c5 = data_clean1["Profit"].value_counts(sort = False)
print(c5)


# convert to numeric format
data_clean1['IsweekdayRC'] = pd.to_numeric(data_clean1['IsweekdayRC'], errors='coerce')
data_clean1['IsweekendRC'] = pd.to_numeric(data_clean1['IsweekendRC'], errors='coerce')
data_clean1['IsworkdayRC'] = pd.to_numeric(data_clean1['IsworkdayRC'], errors='coerce')
data_clean1['IsBankHolidayRC'] = pd.to_numeric(data_clean1['IsBankHolidayRC'], errors='coerce')
data_clean1['ProductSalePrice'] = pd.to_numeric(data_clean1['ProductSalePrice'], errors='coerce')
data_clean1['OrderItemQuantity'] = pd.to_numeric(data_clean1['OrderItemQuantity'], errors='coerce')
data_clean1['SupPurchasePrice'] = pd.to_numeric(data_clean1['SupPurchasePrice'], errors='coerce')
data_clean1['Profit'] = pd.to_numeric(data_clean1['Profit'], errors='coerce')


# bivariate bar graph to visualize the relationship between profit and the predictor variables
#ProductSalePrice vs Profit
sns.factorplot(x="ProductSalePrice", y="Profit", data=data_clean1, kind="bar", ci=None, size=10)
plt.xlabel('ProductSalePrice')
plt.ylabel('Profit')

# bivariate bar graph IsweekendRC vs Profit
sns.factorplot(x="IsweekendRC", y="Profit", data=data_clean1, kind="bar", ci=None, size=5)
plt.xlabel('IsweekendRC')
plt.ylabel('Profit')

# bivariate bar graph IsweekdayRC vs Profit
sns.factorplot(x="IsweekdayRC", y="Profit", data=data_clean1, kind="bar", ci=None, size=5)
plt.xlabel('IsweekdayRC')
plt.ylabel('Profit')


# bivariate bar graph IsBankHolidayRC vs Profit
sns.factorplot(x="IsBankHolidayRC", y="Profit", data=data_clean1, kind="bar", ci=None, size=5)
plt.xlabel('IsBankHolidayRC')
plt.ylabel('Profit')


# bivariate bar graph ProductSalePrice vs Profit
sns.factorplot(x="OrderItemQuantity", y="Profit", data=data_clean1, kind="bar", ci=None, size=5)
plt.xlabel('OrderItemQuantity')
plt.ylabel('Profit')

# bivariate bar graph SupPurchasePrice vs Profit
sns.factorplot(x="SupPurchasePrice", y="Profit", data=data_clean1, kind="bar", ci=None, size=5)
plt.xlabel('SupPurchasePrice')
plt.ylabel('Profit')

#check mean, standard deivation and maximum ,adn min values of quantitative varibles
data_clean1['ProductSalePrice'].describe()

data_clean1['OrderItemQuantity'].describe()

data_clean1['SupPurchasePrice'] .describe()

#ProductSalePrice run the  scatterplots together to get both linear and 2rd order fit lines for x121_2012
scat1 = sns.regplot(x="ProductSalePrice", y="Profit", scatter=True, order=2, data=data_clean1)
plt.xlabel('ProductSalePrice')
plt.ylabel('Profit')

##the resultant relationship from the scatterplot is not linear so i will test significans using regression
# center quantitative IVs for regression analysis
data_clean1['ProductSalePrice_c'] = (data_clean1['ProductSalePrice'] - data_clean1['ProductSalePrice'].mean())
data_clean1['OrderItemQuantity_c'] = (data_clean1['OrderItemQuantity'] - data_clean1['OrderItemQuantity'].mean())
data_clean1['SupPurchasePrice_c'] = (data_clean1['SupPurchasePrice'] - data_clean1['SupPurchasePrice'].mean())
data_clean1[["ProductSalePrice_c", "OrderItemQuantity_c", "SupPurchasePrice_c"]].describe()

# linear regression analysis
reg1 = smf.ols('Profit ~ ProductSalePrice_c', data=data_clean1).fit()
print (reg1.summary())

#using only Product sale price as the predictor
#predictors = data_clean1[['ProductSalePrice']]

#including all other predictors
predictors = data_clean1[['IsweekdayRC','IsweekendRC','IsworkdayRC','IsBankHolidayRC',
'ProductSalePrice','SupPurchasePrice','OrderItemQuantity']]



data_clean1.dtypes

targets = data_clean1.Profit

pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets, test_size=0.4)


pred_train.shape
pred_test.shape

tar_train.shape
tar_test.shape


#Build model on training data
classifier=DecisionTreeClassifier()
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test,predictions)
sklearn.metrics.accuracy_score(tar_test, predictions)


#Displaying the decision tree
from sklearn import tree
#from StringIO import StringIO
from io import StringIO
#from StringIO import StringIO 
from IPython.display import Image
out = StringIO()
tree.export_graphviz(classifier, out_file=out)
import pydotplus
graph=pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png())

#Save the decision tree as an image
with open('optimumPriceALLPREDICTORS.png', 'wb') as f:

    f.write(graph.create_png())



