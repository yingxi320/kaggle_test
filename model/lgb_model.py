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

    data[df]['hour_gap'] = data[df]['visit_datetime'].sub(data[df]['reserve_datetime'])
    data[df]['hour_gap'] = data[df]['hour_gap'].apply(lambda x: x / np.timedelta64(1, 'h'))
    # separate reservation into 5 categories based on gap lenght
    data[df]['reserve_-12_h'] = np.where(data[df]['hour_gap'] <= 12,
                                        data[df]['reserve_visitors'], 0)
    data[df]['reserve_12_37_h'] = np.where((data[df]['hour_gap'] <= 37) & (data[df]['hour_gap'] > 12),
                                          data[df]['reserve_visitors'], 0)
    data[df]['reserve_37_59_h'] = np.where((data[df]['hour_gap'] <= 59) & (data[df]['hour_gap'] > 37),
                                          data[df]['reserve_visitors'], 0)
    data[df]['reserve_59_85_h'] = np.where((data[df]['hour_gap'] <= 85) & (data[df]['hour_gap'] > 59),
                                          data[df]['reserve_visitors'], 0)
    data[df]['reserve_85+_h'] = np.where((data[df]['hour_gap'] > 85),
                                        data[df]['reserve_visitors'], 0)
    # group by air_store_id and visit_date to enable joining with main table
    group_list = ['air_store_id', 'visit_datetime', 'reserve_visitors', 'reserve_-12_h',
                  'reserve_12_37_h', 'reserve_37_59_h', 'reserve_59_85_h', 'reserve_85+_h']
    reserve = data[df][group_list].groupby(['air_store_id', 'visit_datetime'], as_index=False).sum().rename(
        columns={'visit_datetime': 'visit_date'}
    )

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
    data[df] = pd.merge(data[df], reserve, on=['air_store_id', 'visit_date'], how='left')

    del tmp1; del tmp2; del tmp3; del reserve

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

from sklearn.model_selection import GridSearchCV


# from datetime import date
# dd = date(2017, 3, 16)
# validate = train[train['visit_date'] >= dd]
# train_set = train[train['visit_date'] < dd]
kf = KFold(n_splits=10)
test['visitors'] = 0
print('starting training lgb model')
MAX_ROUNDS = 8000
X = train[col].values
y = np.log1p(train['visitors'].values)
train_errors = []
valid_errors = []
idx = 1
for train_index, test_index in kf.split(X):
    print('-' * 50)
    print('fold %d' % idx)
    idx += 1
    X_tr, X_te = X[train_index], X[test_index]
    y_tr, y_te = y[train_index], y[test_index]
    evals_result = {}  # to record eval results for plotting
    dtrain = lgb.Dataset(X_tr, y_tr)
    dval = lgb.Dataset(X_te, y_te)
    bst = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        feature_name=col,evals_result=evals_result,
        valid_sets=[dtrain, dval], early_stopping_rounds=50, verbose_eval=50
    )

    # print('Plot metrics during training...')
    # ax = lgb.plot_metric(evals_result, metric='rmse')
    # plt.show()
    #
    # print('Plot feature importances...')
    # ax = lgb.plot_importance(bst, max_num_features=50)
    # plt.show()

    err_tr = RMSLE(y_tr, bst.predict(X_tr, num_iteration=bst.best_iteration or MAX_ROUNDS))
    err_te = RMSLE(y_te, bst.predict(X_te, num_iteration=bst.best_iteration or MAX_ROUNDS))
    print('RMSE train: ', err_tr)
    print('RMSE validate: ', err_te)
    train_errors.append(err_tr)
    valid_errors.append(err_te)
    preds = bst.predict(test[col], num_iteration=bst.best_iteration or MAX_ROUNDS)
    test['visitors'] += preds
print('Train RMSE mean: ', np.mean(train_errors))
print('Valid RMSE mean: ', np.mean(valid_errors))
test['visitors'] /= 10
#print('Validate RMSE: ', RMSLE(np.log1p(test['visitors'].values), test['preds']))
test['visitors'] = np.expm1(test['visitors']).clip(lower=0.)
sub1 = test[['id', 'visitors']].copy()
del train;
del data;

# from hklee
# https://www.kaggle.com/zeemeen/weighted-mean-comparisons-lb-0-497-1st/code
dfs = { re.search('/([^/\.]*)\.csv', fn).group(1):
    pd.read_csv(fn)for fn in glob.glob('../data/*.csv')}

for k, v in dfs.items(): locals()[k.split('\\')[-1]] = v

wkend_holidays = date_info.apply(
    (lambda x: (x.day_of_week == 'Sunday' or x.day_of_week == 'Saturday') and x.holiday_flg == 1), axis=1)
date_info.loc[wkend_holidays, 'holiday_flg'] = 0
date_info['weight'] = ((date_info.index + 1) / len(date_info)) ** 5

visit_data = air_visit_data.merge(date_info, left_on='visit_date', right_on='calendar_date', how='left')
visit_data.drop('calendar_date', axis=1, inplace=True)
visit_data['visitors'] = visit_data.visitors.map(pd.np.log1p)

wmean = lambda x: ((x.weight * x.visitors).sum() / x.weight.sum())
visitors = visit_data.groupby(['air_store_id', 'day_of_week', 'holiday_flg']).apply(wmean).reset_index()
visitors.rename(columns={0: 'visitors'}, inplace=True)  # cumbersome, should be better ways.

sample_submission['air_store_id'] = sample_submission.id.map(lambda x: '_'.join(x.split('_')[:-1]))
sample_submission['calendar_date'] = sample_submission.id.map(lambda x: x.split('_')[2])
sample_submission.drop('visitors', axis=1, inplace=True)
sample_submission = sample_submission.merge(date_info, on='calendar_date', how='left')
sample_submission = sample_submission.merge(visitors, on=[
    'air_store_id', 'day_of_week', 'holiday_flg'], how='left')

missings = sample_submission.visitors.isnull()
sample_submission.loc[missings, 'visitors'] = sample_submission[missings].merge(
    visitors[visitors.holiday_flg == 0], on=('air_store_id', 'day_of_week'),
    how='left')['visitors_y'].values

missings = sample_submission.visitors.isnull()
sample_submission.loc[missings, 'visitors'] = sample_submission[missings].merge(
    visitors[['air_store_id', 'visitors']].groupby('air_store_id').mean().reset_index(),
    on='air_store_id', how='left')['visitors_y'].values

sample_submission['visitors'] = sample_submission.visitors.map(pd.np.expm1)
sub2 = sample_submission[['id', 'visitors']].copy()
sub_merge = pd.merge(sub1, sub2, on='id', how='inner')

sub_merge['visitors'] = 0.7 * sub_merge['visitors_x'] + 0.3 * sub_merge['visitors_y'] * 1.1
sub_merge[['id', 'visitors']].to_csv('submission_1_25.csv', index=False)

# report
# [2800]	training's rmse: 0.444737	valid_1's rmse: 0.482011
# [2850]	training's rmse: 0.444233	valid_1's rmse: 0.481946
# [2900]	training's rmse: 0.443736	valid_1's rmse: 0.481904
# [2950]	training's rmse: 0.443187	valid_1's rmse: 0.481881
# [3000]	training's rmse: 0.442677	valid_1's rmse: 0.481857
# [3050]	training's rmse: 0.442165	valid_1's rmse: 0.48186
# Early stopping, best iteration is:
# [3008]	training's rmse: 0.442593	valid_1's rmse: 0.481846
# [LightGBM] [Info] Finished loading 3008 models

# RMSE train:  0.444704666105
# RMSE validate:  0.485510708879
# Train RMSE mean:  0.443368844842
# Valid RMSE mean:  0.481681036728
# Validate RMSE:  0.48886187901