import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

###################################################
print('-' * 50)
print('start to load data')
train = pd.read_csv('../data/air_visit_data.csv', parse_dates=['visit_date'])
test = pd.read_csv('../data/sample_submission.csv')
holiday = pd.read_csv('../data/date_info.csv').rename(columns={'calendar_date': 'visit_date'})
ar = pd.read_csv('../data/air_reserve.csv').rename(columns={'visit_datetime': 'visit_date'})
hr = pd.read_csv('../data/hpg_reserve.csv').rename(columns={'visit_datetime': 'visit_date'})
train['set'] = 0
test['set'] = 1
print('done')
#####################################################
print('-' * 50)
print('start to merge train and test')
start_date = train.groupby(['air_store_id'])['visit_date'].min()
test['air_store_id'] = test['id'].apply(lambda x: '_'.join(x.split('_')[:-1]))
test['visit_date'] = test['id'].apply(lambda x: x.split('_')[-1])
test['visit_date'] = pd.to_datetime(test['visit_date'])

test_start = test['visit_date'].min()

df = pd.concat([train, test], axis=0)

df['dow'] = df['visit_date'].dt.dayofweek
df['year'] = df['visit_date'].dt.year
df['month'] = df['visit_date'].dt.month
df['visit_date'] = df['visit_date'].dt.date

df['date_int'] = df['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)

print('done')

data_cols = ['date_int', 'dow', 'year', 'month']
# ################################################
# deal with holiday
print('-' * 50)
print('start to deal with holiday')
holiday['visit_date'] = holiday['visit_date'].apply(lambda x: pd.to_datetime(x).date())
holiday['visit_date'] = holiday['visit_date'].apply(lambda x: pd.to_datetime(x))
df = pd.merge(df, holiday, on='visit_date', how='left')
del train, test, holiday
holi_cols = ['holiday_flg']
print('done')
#############################################################
# deal store info
print('-' * 50)
print('start to deal with store info')
stores = pd.read_csv('../data/air_store_info.csv')

# store stats
stats = df[df['set'] == 0].groupby(['air_store_id', 'dow']).agg(
    {'visitors': [np.min, np.mean, np.median, np.max, np.size, np.std]}).reset_index()
stats.columns = ['air_store_id', 'dow', 'min_visitors', 'mean_visitors', 'median_visitors', 'max_visitors',
                  'count_observations', 'std_visitors']

# location cluster
cluster = KMeans(n_clusters=5)
stores['loc_labels'] = cluster.fit_predict(stores[['longitude', 'latitude']])
loc_label_dit = dict(stores['loc_labels'].value_counts())
stores['loc_store_cnt'] = stores['loc_labels'].map(loc_label_dit)

# NEW FEATURES FROM Georgii Vyshnia
stores['air_genre_name'] = stores['air_genre_name'].map(lambda x: str(str(x).replace('/', ' ')))
stores['air_area_name'] = stores['air_area_name'].map(lambda x: str(str(x).replace('-', ' ')))
lbl = LabelEncoder()

stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])
df = pd.merge(df, stores, on='air_store_id', how='left')
df = pd.merge(df, stats, on=['air_store_id', 'dow'], how='left')
stores_cols = ['air_genre_name', 'air_area_name', 'loc_labels', 'loc_store_cnt']
stats_cols = ['min_visitors', 'mean_visitors', 'median_visitors', 'max_visitors', 'count_observations', 'std_visitors']
del stores, stats
print('done')
#################################################################
#deal with air reserve
print('-' * 50)
print('start to deal with air reserve')
air_reserve = pd.read_csv('../data/air_reserve.csv')
air_reserve['visit_datetime'] = pd.to_datetime(air_reserve['visit_datetime'])
air_reserve['visit_datetime'] = air_reserve['visit_datetime'].dt.date
air_reserve['reserve_datetime'] = pd.to_datetime(air_reserve['reserve_datetime'])
air_reserve['reserve_datetime'] = air_reserve['reserve_datetime'].dt.date
air_reserve['reserve_datetime_diff'] = air_reserve.apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days,
                                                   axis=1)
tmp1 = air_reserve.groupby(['air_store_id', 'visit_datetime'], as_index=False)[
    ['reserve_datetime_diff', 'reserve_visitors']].sum().rename(
    columns={'visit_datetime': 'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors': 'rv1'})
tmp2 = air_reserve.groupby(['air_store_id', 'visit_datetime'], as_index=False)[
    ['reserve_datetime_diff', 'reserve_visitors']].mean().rename(
    columns={'visit_datetime': 'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors': 'rv2'})
air_reserve = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id', 'visit_date'])
df = pd.merge(df, air_reserve, on=['air_store_id', 'visit_date'], how='left')
del air_reserve
print('done')
#############################################################################
#deal with hpg reserve
print('-' * 50)
print('start to deal with hpg reserve')
hpg_reserve = pd.read_csv('../data/hpg_reserve.csv')
hpg_air_id = pd.read_csv('../data/store_id_relation.csv')
hpg_reserve = pd.merge(hpg_reserve, hpg_air_id, how='left', on=['hpg_store_id'])
hpg_reserve['visit_datetime'] = pd.to_datetime(hpg_reserve['visit_datetime'])
hpg_reserve['visit_datetime'] = hpg_reserve['visit_datetime'].dt.date
hpg_reserve['reserve_datetime'] = pd.to_datetime(hpg_reserve['reserve_datetime'])
hpg_reserve['reserve_datetime'] = hpg_reserve['reserve_datetime'].dt.date
hpg_reserve['reserve_datetime_diff'] = hpg_reserve.apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days,
                                                   axis=1)
tmp1 = hpg_reserve.groupby(['air_store_id', 'visit_datetime'], as_index=False)[
    ['reserve_datetime_diff', 'reserve_visitors']].sum().rename(
    columns={'visit_datetime': 'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors': 'rv1'})
tmp2 = hpg_reserve.groupby(['air_store_id', 'visit_datetime'], as_index=False)[
    ['reserve_datetime_diff', 'reserve_visitors']].mean().rename(
    columns={'visit_datetime': 'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors': 'rv2'})
hpg_reserve = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id', 'visit_date'])
df = pd.merge(df, hpg_reserve, on=['air_store_id', 'visit_date'], how='left')

df['total_reserv_sum'] = df['rv1_x'] + df['rv1_y']
df['total_reserv_mean'] = (df['rv2_x'] + df['rv2_y']) / 2
df['total_reserv_dt_diff_mean'] = (df['rs2_x'] + df['rs2_y']) / 2
reserve_cols = ['rs1_x', 'rv1_x', 'rs2_x',
       'rv2_x', 'rs1_y', 'rv1_y', 'rs2_y', 'rv2_y', 'total_reserv_sum',
       'total_reserv_mean', 'total_reserv_dt_diff_mean']
del hpg_reserve
print('done')
########################################################
print('start to build model')
train = df[df['set'] == 0]
test = df[df['set'] == 1]

train = train.fillna(train.median())
test = test.fillna(train.median())

def RMSLE(y, pred):
    return mean_squared_error(y, pred) ** 0.5

# from sklearn.utils import shuffle
#
# train = shuffle(train, random_state=21)

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
################################################################
# build single model
col = data_cols + stats_cols + reserve_cols + holi_cols
for i, j in [('data_cols', data_cols), ('stats_cols', stats_cols), ('reserve_cols', reserve_cols), ('holi_cols', holi_cols)]:
    import matplotlib.pyplot as plt
    import seaborn as sns
    corrmat = train[j + ['visitors']].corr()
    f, ax = plt.subplots(figsize=(20, 10))
    plt.xticks(rotation='90')
    sns.heatmap(corrmat, square=True, linewidths=.5, annot=True)
    plt.title(i)
    plt.show()

MAX_ROUNDS = 8000
X = train[col].values
y = np.log1p(train['visitors'].values)
X_test = test[col].values
kf = KFold(n_splits=10)
train_errors = []
valid_errors = []
idx = 1
for train_index, test_index in kf.split(X):
    print('-' * 50)
    print('fold %d' % idx)
    idx += 1
    X_tr, X_te = X[train_index], X[test_index]
    y_tr, y_te = y[train_index], y[test_index]
    dtrain = lgb.Dataset(X_tr, y_tr)
    dval = lgb.Dataset(X_te, y_te)
    bst = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        valid_sets=[dtrain, dval], early_stopping_rounds=50, verbose_eval=50
    )
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
# test['visitors'] /= 10
# test['visitors'] = np.expm1(test['visitors']).clip(lower=0.)
# sub1 = test[['id', 'visitors']].copy()
#################################################################
# kf = KFold(n_splits=10)
# test['visitors'] = 0
# print('starting training lgb model')
# MAX_ROUNDS = 8000
# X = train[col].values
# y = np.log1p(train['visitors'].values)
# train_errors = []
# valid_errors = []
# idx = 1
# for train_index, test_index in kf.split(X):
#     print('-' * 50)
#     print('fold %d' % idx)
#     idx += 1
#     X_tr, X_te = X[train_index], X[test_index]
#     y_tr, y_te = y[train_index], y[test_index]
#     dtrain = lgb.Dataset(X_tr, y_tr)
#     dval = lgb.Dataset(X_te, y_te)
#     bst = lgb.train(
#         params, dtrain, num_boost_round=MAX_ROUNDS,
#         valid_sets=[dtrain, dval], early_stopping_rounds=50, verbose_eval=50
#     )
#     err_tr = RMSLE(y_tr, bst.predict(X_tr, num_iteration=bst.best_iteration or MAX_ROUNDS))
#     err_te = RMSLE(y_te, bst.predict(X_te, num_iteration=bst.best_iteration or MAX_ROUNDS))
#     import matplotlib.pyplot as plt
#     for i in range(10):
#         plt.figure(figsize=(18, 9))
#         plt.scatter(list(range(len(y_te[i * 200: (i + 1) * 200]))), y_te[i * 200: (i + 1) * 200], c='r', label='true-v')
#         plt.scatter(list(range(len(y_te[i * 200: (i + 1) * 200]))), bst.predict(X_te, num_iteration=bst.best_iteration or MAX_ROUNDS)[i * 200: (i + 1) * 200], c='b', label='pred-v')
#         plt.legend(loc='upper right')
#         plt.grid(True)
#         plt.savefig('../result/%d_fold_%d.png' % (idx, i))
#         plt.show()
#     print('RMSE train: ', err_tr)
#     print('RMSE validate: ', err_te)
#     train_errors.append(err_tr)
#     valid_errors.append(err_te)
#     preds = bst.predict(test[col], num_iteration=bst.best_iteration or MAX_ROUNDS)
#     test['visitors'] += preds
# print('Train RMSE mean: ', np.mean(train_errors))
# print('Valid RMSE mean: ', np.mean(valid_errors))
# test['visitors'] /= 10
# test['visitors'] = np.expm1(test['visitors']).clip(lower=0.)
# sub1 = test[['id', 'visitors']].copy()

