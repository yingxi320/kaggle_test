"""
Contributions from:
DSEverything - Mean Mix - Math, Geo, Harmonic (LB 0.493)
https://www.kaggle.com/dongxu027/mean-mix-math-geo-harmonic-lb-0-493
JdPaletto - Surprised Yet? - Part2 - (LB: 0.503)
https://www.kaggle.com/jdpaletto/surprised-yet-part2-lb-0-503
hklee - weighted mean comparisons, LB 0.497, 1ST
https://www.kaggle.com/zeemeen/weighted-mean-comparisons-lb-0-497-1st

Also all comments for changes, encouragement, and forked scripts rock

Keep the Surprise Going
"""

import glob, re
import numpy as np
import pandas as pd
from sklearn import *
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

data = {
    'tra': pd.read_csv('../data/air_visit_data_new.csv'),
    'as': pd.read_csv('../data/air_store_info.csv'),
    'hs': pd.read_csv('../data/hpg_store_info.csv'),
    'ar': pd.read_csv('../data/air_reserve.csv'),
    'hr': pd.read_csv('../data/hpg_reserve.csv'),
    'id': pd.read_csv('../data/store_id_relation.csv'),
    'tes': pd.read_csv('../data/sample_submission.csv'),
    'hol': pd.read_csv('../data/date_info.csv').rename(columns={'calendar_date': 'visit_date'})
}

# location cluster
cluster = KMeans(n_clusters=5)
data['as']['loc_labels'] = cluster.fit_predict(data['as'][['longitude', 'latitude']])
loc_label_dit = dict(data['as']['loc_labels'].value_counts())
data['as']['loc_store_cnt'] = data['as']['loc_labels'].map(loc_label_dit)

data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])

for df in ['ar', 'hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days,
                                                       axis=1)
    tmp1 = data[df].groupby(['air_store_id', 'visit_datetime'], as_index=False)[
        ['reserve_datetime_diff', 'reserve_visitors']].sum().rename(
        columns={'visit_datetime': 'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors': 'rv1'})
    tmp2 = data[df].groupby(['air_store_id', 'visit_datetime'], as_index=False)[
        ['reserve_datetime_diff', 'reserve_visitors']].mean().rename(
        columns={'visit_datetime': 'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors': 'rv2'})

    tmp3 = data[df].groupby(['air_store_id', 'visit_datetime'], as_index=False).count()[
        ['air_store_id', 'visit_datetime', 'reserve_datetime']].rename(
        columns={'visit_datetime': 'visit_date', 'reserve_datetime': 'visit_counts'})

    data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id', 'visit_date'])

    data[df] = pd.merge(data[df], tmp3, on=['air_store_id', 'visit_date'], how='left')
    del tmp1; del tmp2; del tmp3

data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

###############################################
# start_date
start_date = data['tra'].groupby(['air_store_id'])['visit_date'].min()
start_date = start_date.reset_index().rename(columns={'visit_date': 'start_date'})
start_date['start_date'] = start_date['start_date']
################################################

data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date

unique_stores = data['tes']['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i] * len(unique_stores)}) for i in range(7)],
                   axis=0, ignore_index=True).reset_index(drop=True)

stores = data['tra'].groupby(['air_store_id', 'dow']).agg(
    {'visitors': [np.min, np.mean, np.median, np.max, np.std]}).reset_index()
stores.columns = ['air_store_id', 'dow', 'min_visitors', 'mean_visitors', 'median_visitors', 'max_visitors',
                  'std_visitors']

stores = pd.merge(stores, data['as'], how='left', on=['air_store_id'])
# NEW FEATURES FROM Georgii Vyshnia
stores['air_genre_name'] = stores['air_genre_name'].map(lambda x: str(str(x).replace('/', ' ')))
stores['air_area_name'] = stores['air_area_name'].map(lambda x: str(str(x).replace('-', ' ')))
lbl = preprocessing.LabelEncoder()

for i in range(4):
    stores['air_genre_name' + str(i)] = lbl.fit_transform(
        stores['air_genre_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' ')) > i else ''))
    stores['air_area_name' + str(i)] = lbl.fit_transform(
        stores['air_area_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' ')) > i else ''))

stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date
train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date'])
test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date'])

train = pd.merge(train, stores, how='left', on=['air_store_id', 'dow'])
test = pd.merge(test, stores, how='left', on=['air_store_id', 'dow'])

for df in ['ar', 'hr']:
    train = pd.merge(train, data[df], how='left', on=['air_store_id', 'visit_date'])
    test = pd.merge(test, data[df], how='left', on=['air_store_id', 'visit_date'])

train['id'] = train.apply(lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), axis=1)

train['set'] = 0
test['set'] = 1
train_test = pd.concat([train, test], axis=0)

store_with_no_visit_rate = pd.read_csv('../data/store_with_no_visitor_rate.csv')
train_test = pd.merge(train_test, store_with_no_visit_rate, on='air_store_id', how='left')
del store_with_no_visit_rate

train_test['total_reserv_sum'] = train_test['rv1_x'] + train_test['rv1_y']
train_test['total_reserv_mean'] = (train_test['rv2_x'] + train_test['rv2_y']) / 2
train_test['total_reserv_dt_diff_mean'] = (train_test['rs2_x'] + train_test['rs2_y']) / 2

train_test = pd.merge(train_test, start_date, on='air_store_id', how='left')
train_test['days_from_start_date'] = train_test.apply(lambda x: (x['visit_date'] - x['start_date']).days, axis=1)
# NEW FEATURES FROM JMBULL
train_test['date_int'] = train_test['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
train_test['var_max_lat'] = train_test['latitude'].max() - train_test['latitude']
train_test['var_max_long'] = train_test['longitude'].max() - train_test['longitude']

# NEW FEATURES FROM Georgii Vyshnia
train_test['lon_plus_lat'] = train_test['longitude'] + train_test['latitude']

lbl = preprocessing.LabelEncoder()
train_test['air_store_id2'] = lbl.fit_transform(train_test['air_store_id'])

train = train_test[train_test['set'] == 0]
test = train_test[train_test['set'] == 1]

train.drop('set', axis=1, inplace=True)
test.drop('set', axis=1, inplace=True)

col = [c for c in train.columns if c not in ['id', 'air_store_id', 'visit_date', 'visitors', 'start_date', 'day_of_week']]
print(col)
train = train.fillna(-1)
test = test.fillna(-1)

def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred) ** 0.5

from sklearn.utils import shuffle

train = shuffle(train, random_state=2018)

params = {
    'num_leaves': 60,
    'objective': 'regression',
    'min_data_in_leaf': 50,
    'learning_rate': 0.02,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'metric': 'rmse',
    'num_threads': 16
}

param_grid = {
    'num_leaves': [31, 40, 50, 60, 80],
    'objective': ['regression'],
    'min_data_in_leaf': [40, 50, 60, 70, 80, 90, 100],
    'learning_rate': [0.02, 0.03, 0.04, 0.05],
    'bagging_fraction': [0.7, 0.75, 0.8, 0.9],
    'bagging_freq': [1],
    'metric': ['rmse']
}

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
gbm = lgb.LGBMRegressor()
grid = GridSearchCV(gbm, cv=10, n_jobs=1, param_grid=param_grid, scoring=make_scorer(RMSLE), verbose=2)

X = train[col].values
y = np.log1p(train['visitors'].values)
grid.fit(X, y)

print('after search')
print(grid.best_params_, grid.best_score_)
