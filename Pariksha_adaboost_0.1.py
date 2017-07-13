
# coding: utf-8

# In[316]:

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV


# In[279]:

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')


# In[280]:

print data_train.shape
print data_test.shape


# ## EDA

# In[281]:

Counter(data_train.columns.str.contains('54500'))


# In[282]:

cols_54500 = [col for col in data_train.columns if '54500' in col]
cols_54499 = [col for col in data_train.columns if '54499' in col]
cols_54498 = [col for col in data_train.columns if '54498' in col]


# In[120]:

set(data_train.columns) - set(cols_54499 + cols_54500 + cols_54498)


# #### Mising Value Imputation

# In[283]:

# First replacing all the values having strings '?' and '#VALUE!' by NaN, in the training and test data set

print data_train.isnull().values.any()
data_train = data_train.replace('?', np.nan)
data_train = data_train.replace('#VALUE!', np.nan)
print data_train.isnull().values.any()


# In[284]:

print data_test.isnull().values.any()
data_test = data_test.replace('?', np.nan)
data_test = data_test.replace('#VALUE!', np.nan)
print data_test.isnull().values.any()


# In[285]:

# Drop all columns indicating time. Only the start time is given, if the end time was also given the difference could've 
# been used as a feature. Also all features having stats for test_10 have been dropped.

drop_cols = [col for col in data_train.columns if '_start_time' in col]
drop_cols = drop_cols + [col for col in data_train.columns if 'test_10' in col]
drop_cols.remove('test_10_total')


# In[286]:

drop_cols


# In[287]:

data_train = data_train.drop(drop_cols, axis=1)
data_test  = data_test.drop(drop_cols, axis=1)


# In[288]:

# Imputing missing values in each column by median value of the non-NaN values. '?' probably means that the test was not 
# taken by the candidate. Hence,imputing by 0 would affect the predictions becsause that would mean, he/she took the test and scored a 0

i = 0
for i in xrange(data_train.shape[1]):
    data_train.iloc[:,i] = pd.to_numeric(data_train.iloc[:,i])
    data_train.iloc[:,i] = data_train.iloc[:,i].fillna(np.nanmedian(data_train.iloc[:,i]))
#     data[item] = data[item].fillna(np.nanmedian(data[item]))


# In[289]:

print data_train.shape
print data_test.shape


# In[290]:

# Doing the same imputation for the test data

i = 0
for i in xrange(data_test.shape[1]):
    print '\r' + str(i),
    data_test.iloc[:,i] = pd.to_numeric(data_test.iloc[:,i])
    data_test.iloc[:,i] = data_test.iloc[:,i].fillna(np.nanmedian(data_test.iloc[:,i]))

print data_test.isnull().values.any()


# In[291]:

X_test.columns[pd.isnull(X_test).sum() > 0].tolist()


# In[258]:

X_train, X_test, y_train, y_test = train_test_split(data_train[list(set(data_train.columns)-set(drop_cols)-set(['test_10_total']))],data_train['test_10_total'],test_size=0.2 )


# In[292]:

data_train_X = data_train[list(set(data_train.columns)-set(drop_cols)-set(['test_10_total']))]
data_train_y = data_train['test_10_total']
data_test_X = data_test[list(set(data_test.columns)-set(drop_cols)-set(['test_10_total']))]
data_test_y = data_test['test_10_total']


# In[293]:

data_train_X.shape


# ##### There seems to ba a linear relationship between no_correct_ques, no_incorrect_ques and total score
# ##### For each correct queston for 54500 you get +3 and for every incorrect you get -1

# In[44]:

from sklearn.ensemble import RandomForestRegressor


# In[263]:

def std_error_estimate(y, y_pred):
    diff = y - y_pred
    error = np.sqrt(sum(diff**2)/len(y))
    return error


# #### Parameter Tuning

# In[319]:

model_rf = RandomForestRegressor(n_jobs=-1,n_estimators=500, oob_score = True)
param_grid = {
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 12, None]    
}

CV_rf = GridSearchCV(estimator=model_rf, param_grid=param_grid, cv= 5)
CV_rf.fit(data_train_X, data_train_y)


# In[321]:

print CV_rf.best_params_
# print CV_rfc.grid_scores_


# In[325]:

rf = RandomForestRegressor(n_jobs = -1,n_estimators=1000, max_features='log2', max_depth=None)
model_rf = rf.fit(data_train_X, data_train_y)


# In[327]:

pred_y = np.round(model_rf.predict(data_test_X))


# In[330]:

pd.DataFrame(pred_y).to_csv('C:/Users/a566280/Documents/Pariksha/submission_RF_simple.csv', index=False, header=None)


# ### Checking the performance by Cross Vallidation

# In[300]:

from sklearn.model_selection import KFold


# In[322]:

kf = KFold(n_splits=10)
kf.get_n_splits(data_train)

print kf
cross_val_score = []
i = 0

for train_index, test_index in kf.split(data_train):
    print '\r' + str(i),
    i+=1
#     print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = data_train_X.ix[train_index], data_train_X.ix[test_index]
    y_train, y_test = data_train_y.ix[train_index], data_train_y.ix[test_index]
    
    rf = RandomForestRegressor(n_jobs = -1,n_estimators=1000, max_features='log2')
    model_rf = rf.fit(X_train, y_train)
    
    pred_y = model_rf.predict(X_test)
    
    cross_val_score.append(std_error_estimate(np.array(y_test), np.round(pred_y)))


# In[324]:

np.mean(np.array(cross_val_score))
# cross_val_score


# ## Adaboost

# In[331]:

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor


# In[359]:

regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=None, max_features='log2'),
                          n_estimators=500, random_state=rng)
param_grid = {
    'loss': ['linear', 'square', 'exponential'],
    'learning_rate' : [0.05, 0.1, 0.3, 0.5, 1]
}

CV_rf = GridSearchCV(estimator=regr_2, param_grid=param_grid, cv= 5)
CV_rf.fit(data_train_X, data_train_y)


# In[360]:

print CV_rf.best_params_


# In[363]:

kf = KFold(n_splits=10)
kf.get_n_splits(data_train)

print kf
cross_val_score = []
i = 0
rng = np.random.RandomState(1)

for train_index, test_index in kf.split(data_train):
    print '\r' + str(i),
    i+=1
#     print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = data_train_X.ix[train_index], data_train_X.ix[test_index]
    y_train, y_test = data_train_y.ix[train_index], data_train_y.ix[test_index]
    
    regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=None, max_features='log2'),
                          n_estimators=1000, random_state=rng, loss='exponential', learning_rate=0.1)
    regr_2.fit(X_train, y_train)
    
    pred_y_adaboost = regr_2.predict(X_test)#model_rf.predict(X_test)
    
    cross_val_score.append(std_error_estimate(np.array(y_test), np.round(pred_y_adaboost)))


# In[364]:

np.mean(np.array(cross_val_score))


# In[365]:

pred_y_adaboost = regr_2.predict(data_test_X)


# In[366]:

pd.DataFrame(pred_y_adaboost).to_csv('C:/Users/a566280/Documents/Pariksha/submission_RF_adaboost_2.csv', index=False, header=None)

