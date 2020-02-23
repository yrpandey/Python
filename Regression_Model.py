# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 18:11:49 2020

@author: Yograj
"""

import pandas as pd
import numpy as np
import seaborn as sns

sns.set(rc={'figure.figsize':(11.7,8.27)})
cars_data=pd.read_csv('cars_sampled.csv')
cars_data.describe()
cars=cars_data.copy()
cars.info()
cars.describe()
pd.set_option('display.float_format',lambda x:'%.3f'%x)
cars.describe()
#To Display maximum Set of column
pd.set_option('display.max_columns',500)
cars.describe()

#Droping unwanted column
col=['name','dateCrawled','dateCreated','postalCode','lastSeen']
cars=cars.drop(columns=col,axis=1)

#Removing duplicate record
cars.drop_duplicates(keep='first',inplace=True)

#Data Cleaning

#no of  missing values in each column
cars.isnull().sum()
#variable of year of registraion
yearwise_count=cars['yearOfRegistration'].value_counts().sort_index()
sum(cars['yearOfRegistration']>2018)
sum(cars['yearOfRegistration']<1950)
sns.regplot(x='yearOfRegistration',y='price',scatter=True,fit_reg=False,data=cars)
# working range 1950,2018

#variable price
price_count=cars['price'].value_counts().sort_index()
sns.distplot(cars['price'])
cars['price'].describe()
sns.boxplot(y=cars['price'])
sum(cars['price']>150000)
sum(cars['price']<100)
#working range 100 and 150000


#variable powerPS
power_count=cars['powerPS'].value_counts().sort_index()
sns.distplot(cars['powerPS'])
cars['powerPS'].describe()
sns.boxplot(y=cars['powerPS'])
sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False,data=cars)
sum(cars['powerPS']>500)
sum(cars['powerPS']<10)
#working range 10 and 150

# ======================================
# Working range of data
# ======================================

# Working range of data

cars=cars[
        (cars.yearOfRegistration<=2018)
        &(cars.yearOfRegistration>=1950)
        &(cars.price>=100)&(cars.price<=150000)
        &(cars.powerPS>=10)&(cars.powerPS<=500)
        ]
#~6700 records are dropped
# Further  to simplyfy- variable reduction
#combining yearOfRegistration and monthOfRegistration
cars['monthOfRegistration']/=12

# Creating new variable Age by adding yearOfRegistration and monthOfRegistration
cars['Age']=(2018-cars['yearOfRegistration'])+cars['monthOfRegistration']
cars['Age']=round(cars['Age'],2)
cars['Age'].describe()

# Dropping yearOfRegistration and monthOfRegistration
cars=cars.drop(columns=['yearOfRegistration','monthOfRegistration'],axis=1)

#visualizing parameters
# Age
sns.distplot(cars['Age'])
sns.boxplot(y=cars['Age'])
# price
sns.distplot(cars['price'])
sns.boxplot(y=cars['price'])
# powerPS
sns.distplot(cars['powerPS'])
sns.boxplot(y=cars['powerPS'])

#Visualizing parameters after narrowing working range
#Age vs price
sns.regplot(x='Age',y='price',scatter=True,fit_reg=False,data=cars)
#car priced higher are newer
#with increase Age price decreases
#Howerver some cars are priced higher with increas age

#powerPS vs price
sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False,data=cars)

#variable seller
cars['seller'].value_counts()
pd.crosstab(cars['seller'],columns='counts',normalize=True)
sns.countplot(x='seller',data=cars)
# fewer cars have 'commercial'=>insignificant

#variable offerType
cars['offerType'].value_counts()
sns.countplot(x='offerType',data=cars)
# All cars have 'offer'=>insignificant

#variable abtest
cars['abtest'].value_counts()
pd.crosstab(cars['abtest'],columns='counts',normalize=True)
sns.countplot(x='abtest',data=cars)
#Equaly distributed

sns.boxplot(x='abtest',y='price',data=cars)
#For every price value there is almost 50-50 distribution
#doesnot affect price=>insignificant

#variable vehicleType
cars['vehicleType'].value_counts()
pd.crosstab(cars['vehicleType'],columns='counts',normalize=True)
sns.countplot(x='vehicleType',data=cars)
sns.boxplot(x='vehicleType',y='price',data=cars)
#8 types- limousine,small cars and station wagons max freq
#vehicleType affects price

#variable gearbox
cars['gearbox'].value_counts()
pd.crosstab(cars['gearbox'],columns='count',normalize=True)
sns.countplot(x='gearbox',data=cars)
sns.boxplot(x='gearbox',y='price',data=cars)
#gearbox affects price

#variable model
cars['model'].value_counts()
pd.crosstab(cars['model'],columns='count',normalize=True)
sns.countplot(x='model',data=cars)
sns.boxplot(x='model',y='price',data=cars)
#cars are distributed over many models
#considering in modeling

#variable kilometer
cars['kilometer'].value_counts()
pd.crosstab(cars['kilometer'],columns='count',normalize=True)
sns.countplot(x='kilometer',data=cars)
sns.boxplot(x='kilometer',y='price',data=cars)
cars['kilometer'].describe()
sns.distplot(cars['kilometer'],bins=8,kde=False)
sns.regplot(x='kilometer',y='price',scatter=True,fit_reg=False,data=cars)
#considering in modeling


#variable fuelType
cars['fuelType'].value_counts()
pd.crosstab(cars['fuelType'],columns='count',normalize=True)
sns.countplot(x='fuelType',data=cars)
sns.boxplot(x='fuelType',y='price',data=cars)
#fuelType affects price

#variable brand
cars['brand'].value_counts()
pd.crosstab(cars['brand'],columns='count',normalize=True)
sns.countplot(x='brand',data=cars)
sns.boxplot(x='brand',y='price',data=cars)
#Cars are distruted over many brands
#considering in modeling


#variable notRepairedDamage
# yes- car is damaged but not rectified
# no- car was damaged but has been rectified
cars['notRepairedDamage'].value_counts()
pd.crosstab(cars['notRepairedDamage'],columns='count',normalize=True)
sns.countplot(x='notRepairedDamage',data=cars)
sns.boxplot(x='notRepairedDamage',y='price',data=cars)
#As expected the cars that require the damage to be repaired
#fall under lower price range

#=====================================================
#===========Removing  insignificant  variables
#===================================================

col=['seller','offerType','abtest']
cars=cars.drop(columns=col,axis=1)
cars_copy=cars.copy()

#=====================================================
#===========correlation
#===================================================

cars_select1=cars.select_dtypes(exclude=[object])
correlation=cars_select1.corr()
round(correlation)
cars_select1.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]

"""
#=====================================================
# we are going to build a linear regression and Random Forest model
 on two set of data.
 1.Data obtained by  omitting rows with any missing value
 2. Data obtained by imputing the missing values
#===================================================
"""
#=====================================================
#omitting missing values
#===================================================

cars_omit=cars.dropna(axis=0)

#converting categorical variables to dummy variables
cars_omit=pd.get_dummies(cars_omit,drop_first=True)

#=====================================================
#add som libraries
#===================================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


#=====================================================
#Model building with omitted data
#===================================================

#separating  input and output  features
x1=cars_omit.drop(['price'],axis='columns',inplace=False)
y1=cars_omit['price']

#plotting the variable price
prices=pd.DataFrame({"1. Before":y1,"2. After":np.log(y1)})
prices.hist()

#transforming price as a logarithmic value
y1=np.log(y1)

#transforming price as a logarithmic value
#splitting data into test and train
X_train,X_test,y_train,y_test=train_test_split(x1,y1,test_size=0.3,random_state=3)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
#=====================================================
#Baseline model for Omitted data
#===================================================
"""
 we are making  a base model by using test data mean value
 This is to set a benchmark and to a comparewit our regression model  
"""

# finding  the mean for test data value
based_pred=np.mean(y_test)
print(based_pred)

# repeating the same value till length of test data
based_pred=np.repeat(based_pred,len(y_test))

#Find RMSE root mean square error
base_root_mean_square_error=np.sqrt(mean_squared_error(y_test,based_pred))
print(base_root_mean_square_error)


#=====================================================
#lenear regression with Omitted data
#===================================================

# setting intercept as true
lgr=LinearRegression(fit_intercept=True)

#Model; X_Train=input feature y_train= output feature of train set of data
model_lin1=lgr.fit(X_train,y_train)

# Predecting model on test  set
car_predictions_lin1=lgr.predict(X_test)

#Computing MSE and RMSE
lin_mse1=mean_squared_error(y_test,car_predictions_lin1)
lin_rmse1=np.sqrt(lin_mse1)
print(lin_rmse1)

# Rsquared value
r2_lin_test1=model_lin1.score(X_test,y_test)
r2_lin_train1=model_lin1.score(X_train,y_train)
print(r2_lin_test1,r2_lin_train1)

#Regression diagnostics- Residual plot analysis
#Residual plot: difference between test data and prediction value
# or difference between predicted value and you actual value
residuals1=y_test-car_predictions_lin1
sns.regplot(x=car_predictions_lin1,y=residuals1,scatter=True,fit_reg=False,data=cars)
residuals1.describe()

#=====================================================
#Random forest with omited data
#===================================================
#Model parameters
rf=RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,min_samples_split=10,min_samples_leaf=4,random_state=1)

#Model
model_rf1=rf.fit(X_train,y_train)

#Predicting model on test set
cars_predictions_rf1=rf.predict(X_test)

#Computing MSE and RMSE
rf_mse1=mean_squared_error(y_test,cars_predictions_rf1)
rf_rmse1=np.sqrt(rf_mse1)
print(rf_rmse1)

# Rsquared value
r2_rf_test1=model_rf1.score(X_test,y_test)
r2_rf_train1=model_rf1.score(X_train,y_train)
print(r2_rf_test1,r2_rf_train1)

#=====================================================
#Model building with imputed data
#===================================================

cars_imputed=cars.apply(lambda x:x.fillna(x.median())\
                        if x.dtype=='float' else \
                        x.fillna(x.value_counts().index[0]))
cars_imputed.isnull().sum()

#Converting categorial variables to dummy variables
cars_imputed=pd.get_dummies(cars_imputed,drop_first=True)

#Separating input and output feature
x2=cars_imputed.drop(['price'],axis='columns',inplace=False)
y2=cars_imputed['price']

#Plotting the variables price
price=pd.DataFrame({"1. Before":y2,"2. After":np.log(y2)})
prices.hist()
y2=np.log(y2)
#splitting data into test and train
X_train1,X_test1,y_train1,y_test1=train_test_split(x2,y2,test_size=0.3,random_state=3)
print(X_train1.shape,X_test1.shape,y_train1.shape,y_test1.shape)

#=====================================================
#Baseline model for imputed data
#===================================================


# finding  the mean for test data value
based_pred=np.mean(y_test1)
print(based_pred)

# repeating the same value till length of test data
based_pred=np.repeat(based_pred,len(y_test1))

#Find RMSE root mean square error
base_root_mean_square_error_imputted=np.sqrt(mean_squared_error(y_test1,based_pred))
print(base_root_mean_square_error_imputted)

#=====================================================
#Linear regression with imputed data
#===================================================

# setting intercept as true
lgr2=LinearRegression(fit_intercept=True)

#Model; X_Train=input feature y_train= output feature of train set of data
model_lin2=lgr2.fit(X_train1,y_train1)

# Predecting model on test  set
car_predictions_lin2=lgr2.predict(X_test1)

#Computing MSE and RMSE
lin_mse2=mean_squared_error(y_test1,car_predictions_lin2)
lin_rmse2=np.sqrt(lin_mse2)
print(lin_rmse2)
# Rsquared value
r2_lin_test2=model_lin2.score(X_test1,y_test1)
r2_lin_train2=model_lin2.score(X_train1,y_train1)
print(r2_lin_test2,r2_lin_train2)
#=====================================================
#Random forest with imputed data
#===================================================

#Model parameters
rf2=RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,min_samples_split=10,min_samples_leaf=4,random_state=1)

#Model
model_rf2=rf2.fit(X_train1,y_train1)

#Predicting model on test set
cars_predictions_rf2=rf.predict(X_test1)

#Computing MSE and RMSE
rf_mse2=mean_squared_error(y_test1,cars_predictions_rf2)
rf_rmse2=np.sqrt(rf_mse2)
print(rf_rmse2)

# Rsquared value
r2_rf_test2=model_rf2.score(X_test1,y_test1)
r2_rf_train2=model_rf2.score(X_train1,y_train1)
print(r2_rf_test2,r2_rf_train2)





























