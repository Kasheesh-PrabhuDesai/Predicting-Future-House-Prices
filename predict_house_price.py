#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 23:06:35 2019

@author: kasheesh
"""
from pandas import DataFrame
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer



df = pd.read_csv('train.csv')

enc = OneHotEncoder(sparse=False)
imp = SimpleImputer(missing_values=np.nan,strategy='most_frequent')

df = DataFrame(imp.fit_transform(df),columns=df.columns)


def clean_data(data):
    X = data.drop(columns=['SalePrice','Id','PoolQC','Fence','MiscFeature'],axis=1)
    Y = DataFrame(data.SalePrice.values.reshape(-1,1),columns=['SalePrice'])
    '''
    X['LotFrontage']=X['LotFrontage'].fillna(method='bfill')
    X['Alley']=X['Alley'].fillna(value='None')
    X['FireplaceQu']=X['FireplaceQu'].fillna(value='None')
    X['GarageType']=X['GarageType'].fillna(value='None')
    X['GarageYrBlt']=X['GarageYrBlt'].fillna(method='bfill')
    X['GarageFinish']=X['GarageFinish'].fillna(value='None')
    X['GarageQual']=X['GarageQual'].fillna(value='None')
    X['GarageCond']=X['GarageCond'].fillna(value='None')
    X['BsmtQual'] = X['BsmtQual'].fillna(value='None')
    X['BsmtCond'] = X['BsmtCond'].fillna(value='None')
    X['BsmtExposure'] = X['BsmtExposure'].fillna(method='bfill')
    X['BsmtFinType1'] = X['BsmtFinType1'].fillna(method='bfill')
    X['BsmtFinType2'] = X['BsmtFinType2'].fillna(method='bfill')
    X['Electrical'] = X['Electrical'].fillna(method='bfill')
    X['MasVnrType'] = X['MasVnrType'].fillna(method='bfill')
    '''

    #X = X.fillna(method='bfill') 
    #X['Alley']=X['Alley'].fillna(value='None')
    
    
    return X,Y



def transform_data(data):
    data['MSZoning'] = enc.fit_transform(data.MSZoning.values.reshape(-1,1))
    data['Street'] = enc.fit_transform(data.Street.values.reshape(-1,1))
    data['LotShape'] = enc.fit_transform(data.LotShape.values.reshape(-1,1))
    data['LandContour'] = enc.fit_transform(data.LandContour.values.reshape(-1,1))
    data['Utilities'] = enc.fit_transform(data.Utilities.values.reshape(-1,1))
    data['LotConfig'] = enc.fit_transform(data.LotConfig.values.reshape(-1,1))
    data['LandSlope'] = enc.fit_transform(data.LandSlope.values.reshape(-1,1))
    data['Neighborhood'] = enc.fit_transform(data.Neighborhood.values.reshape(-1,1))
    data['Condition1'] = enc.fit_transform(data.Condition1.values.reshape(-1,1))
    data['Condition2'] = enc.fit_transform(data.Condition2.values.reshape(-1,1))
    data['BldgType'] = enc.fit_transform(data.BldgType.values.reshape(-1,1))
    data['HouseStyle'] = enc.fit_transform(data.HouseStyle.values.reshape(-1,1))
    data['RoofStyle'] = enc.fit_transform(data.RoofStyle.values.reshape(-1,1))
    data['RoofMatl'] = enc.fit_transform(data.RoofMatl.values.reshape(-1,1))
    data['Exterior1st'] = enc.fit_transform(data.Exterior1st.values.reshape(-1,1))
    data['Exterior2nd'] = enc.fit_transform(data.Exterior2nd.values.reshape(-1,1))
    data['ExterQual'] = enc.fit_transform(data.ExterQual.values.reshape(-1,1))
    data['ExterCond'] = enc.fit_transform(data.ExterCond.values.reshape(-1,1))
    data['Foundation'] = enc.fit_transform(data.Foundation.values.reshape(-1,1))
    data['BsmtQual'] = enc.fit_transform(data.BsmtQual.values.reshape(-1,1))
    data['BsmtCond'] = enc.fit_transform(data.BsmtCond.values.reshape(-1,1))
    data['BsmtExposure'] = enc.fit_transform(data.BsmtExposure.values.reshape(-1,1))
    data['BsmtFinType1'] = enc.fit_transform(data.BsmtFinType1.values.reshape(-1,1))
    data['BsmtFinType2'] = enc.fit_transform(data.BsmtFinType2.values.reshape(-1,1))
    data['Heating'] = enc.fit_transform(data.Heating.values.reshape(-1,1))
    data['HeatingQC'] = enc.fit_transform(data.HeatingQC.values.reshape(-1,1))
    data['CentralAir'] = enc.fit_transform(data.CentralAir.values.reshape(-1,1))
    data['Electrical'] = enc.fit_transform(data.Electrical.values.reshape(-1,1))
    data['KitchenQual'] = enc.fit_transform(data.KitchenQual.values.reshape(-1,1))
    data['Functional'] = enc.fit_transform(data.Functional.values.reshape(-1,1))
    data['FireplaceQu'] = enc.fit_transform(data.FireplaceQu.values.reshape(-1,1))
    data['GarageType'] = enc.fit_transform(data.GarageType.values.reshape(-1,1))
    data['GarageFinish'] = enc.fit_transform(data.GarageFinish.values.reshape(-1,1))
    data['GarageQual'] = enc.fit_transform(data.GarageQual.values.reshape(-1,1))
    data['GarageCond'] = enc.fit_transform(data.GarageCond.values.reshape(-1,1))
    data['PavedDrive'] = enc.fit_transform(data.PavedDrive.values.reshape(-1,1))
    data['Alley'] = enc.fit_transform(data.Alley.values.reshape(-1,1))
    data['SaleType'] = enc.fit_transform(data.SaleType.values.reshape(-1,1))
    data['SaleCondition'] = enc.fit_transform(data.SaleCondition.values.reshape(-1,1))
    data['MasVnrType'] = enc.fit_transform(data.MasVnrType.values.reshape(-1,1))
    
      
    return data


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,ExtraTreesClassifier,GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error

   
x,y = clean_data(df)


x_set = transform_data(x)

x_train,x_test,y_train,y_test = train_test_split(x_set,y,test_size=0.25)

clf = GradientBoostingRegressor(n_estimators=100,learning_rate=0.1)

clf.fit(x_train,y_train)

result = clf.predict(x_test).reshape(-1,1)



print(np.sqrt(mean_squared_log_error(y_test,result)))





test = pd.read_csv('test.csv')    
test_set = DataFrame(imp.fit_transform(test),columns=test.columns)

x_t = transform_data(test_set)
x_t = x_t.drop(['Id','PoolQC','MiscFeature','Fence'],axis=1)


result_test = clf.predict(x_t).reshape(-1,1)


id_df = DataFrame(test_set['Id'],columns=['Id'])

r_df = DataFrame(result_test,columns=['SalePrice'])   

f_df = pd.concat([id_df,r_df],axis=1) 

f_df.to_csv('predictions-30.07(1).csv')






