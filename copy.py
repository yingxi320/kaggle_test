import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, date
import matplotlib.pyplot as plt

# extract infomation from store info
air_store_info = pd.read_csv('data/air_store_info.csv')
air_hpg_id = pd.read_csv('data/store_id_relation.csv')
# location cluster
cluster = KMeans(n_clusters=5)
air_store_info['loc_labels'] = cluster.fit_predict(air_store_info[['longitude', 'latitude']])
loc_label_dit = dict(air_store_info['loc_labels'].value_counts())
air_store_info['loc_store_cnt'] = air_store_info['loc_labels'].map(loc_label_dit)

# 1.0
#############################################################################
# extract infomation from trian and test file
train = pd.read_csv('data/air_visit_data_new.csv', parse_dates=['visit_date'])
test = pd.read_csv('data/sample_submission.csv')
test['visit_date'] = test['id'].map(lambda x: str(x).split('_')[2])
test['air_store_id'] = test['id'].map(lambda x: '_'.join(x.split('_')[:2]))
test['visit_date'] = pd.to_datetime(test['visit_date'])
test.drop('id', axis=1, inplace=True)
test['visitors'] = 0

def generate_days_for_date(df):
    df['day_of_week'] = df['visit_date'].dt.dayofweek
    df['week_of_year'] = df['visit_date'].apply(lambda x: x.isocalendar()[1])
    df['month'] = df['visit_date'].dt.month
    df['day'] = df['visit_date'].dt.day
    df['year'] = df['visit_date'].dt.year
    df['visit_date'] = df['visit_date'].dt.date
    return df
train = generate_days_for_date(train)
test = generate_days_for_date(test)

train['set'] = 0
test['set'] = 1

# start_date
start_date = train.groupby(['air_store_id'])['visit_date'].min()
start_date = start_date.reset_index().rename(columns={'visit_date': 'start_date'})
start_date['start_date'] = start_date['start_date']
start_date['start_date_int'] = start_date['start_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)

# is need to add max, min, std ? only with train set
store_sales = train.groupby(['air_store_id', 'day_of_week']).agg(
    {'visitors': [np.mean, np.median, np.std, np.min, np.max]}).reset_index()
store_sales.columns = ['air_store_id', 'day_of_week', 'mean_visitors', 'median_visitors'
                       , 'std_visitors', 'min_visitors', 'max_visitors']

air_store_info = pd.merge(store_sales, air_store_info, on='air_store_id', how='left')

air_store_info['air_genre_name'] = air_store_info['air_genre_name'].map(lambda x: str(str(x).replace('/', ' ')))
air_store_info['air_area_name'] = air_store_info['air_area_name'].map(lambda x: str(str(x).replace('-', ' ')))

air_store_info['genre_size'] = air_store_info['air_genre_name'].apply(lambda x: len(x.split(' ')))
air_store_info['area_size'] = air_store_info['air_area_name'].apply(lambda x: len(x.split(' ')))

print(air_store_info['genre_size'].unique())
print(air_store_info['area_size'].unique())

le = LabelEncoder()

for i in range(3):
    air_store_info['air_genre_name' + str(i)] = le.fit_transform(
        air_store_info['air_genre_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' ')) > i else ''))
for i in range(7):
    air_store_info['air_area_name' + str(i)] = le.fit_transform(
        air_store_info['air_area_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' ')) > i else ''))

air_store_info['air_genre_name'] = le.fit_transform(air_store_info['air_genre_name'])
air_store_info['air_area_name'] = le.fit_transform(air_store_info['air_area_name'])


holiday = pd.read_csv('data/date_info.csv').rename(columns={'calendar_date': 'visit_date'})
holiday['visit_date'] = pd.to_datetime(holiday['visit_date'])
holiday['visit_date'] = holiday['visit_date'].dt.date
holiday['is_previous_holiday'] = holiday['holiday_flg'].shift(1).fillna(0)
holiday['is_next_holiday'] = holiday['holiday_flg'].shift(-1).fillna(0)

holiday.drop('day_of_week', axis=1, inplace=True)
train = pd.merge(train, holiday, on='visit_date', how='left')
test = pd.merge(test, holiday, on='visit_date', how='left')

train = pd.merge(train, store_sales, how='left', on=['air_store_id', 'dow'])
test = pd.merge(test, store_sales, how='left', on=['air_store_id', 'dow'])

air_reserve = pd.read_csv('data/air_reserve.csv')
hpg_reserve = pd.read_csv('data/hpg_reserve.csv')
hpg_reserve = pd.merge(hpg_reserve, air_hpg_id, how='inner', on=['hpg_store_id'])
del air_hpg_id

def deal_with_reserve_info(df):
    df['visit_datetime'] = pd.to_datetime(df['visit_datetime'])
    df['visit_datetime'] = df['visit_datetime'].dt.date
    df['reserve_datetime'] = pd.to_datetime(df['reserve_datetime'])
    df['reserve_datetime'] = df['reserve_datetime'].dt.date

    df['hour_gap'] = df['visit_datetime'].sub(df['reserve_datetime'])
    df['hour_gap'] = df['hour_gap'].apply(lambda x: x / np.timedelta64(1, 'h'))
    # separate reservation into 5 categories based on gap lenght
    df['reserve_-12_h'] = np.where(df['hour_gap'] <= 12,
                                        df['reserve_visitors'], 0)
    df['reserve_12_37_h'] = np.where((df['hour_gap'] <= 37) & (df['hour_gap'] > 12),
                                          df['reserve_visitors'], 0)
    df['reserve_37_59_h'] = np.where((df['hour_gap'] <= 59) & (df['hour_gap'] > 37),
                                          df['reserve_visitors'], 0)
    df['reserve_59_85_h'] = np.where((df['hour_gap'] <= 85) & (df['hour_gap'] > 59),
                                          df['reserve_visitors'], 0)
    df['reserve_85+_h'] = np.where((df['hour_gap'] > 85),
                                        df['reserve_visitors'], 0)
    # group by air_store_id and visit_date to enable joining with main table
    group_list = ['air_store_id', 'visit_datetime', 'reserve_visitors', 'reserve_-12_h',
                  'reserve_12_37_h', 'reserve_37_59_h', 'reserve_59_85_h', 'reserve_85+_h']
    reserve = df[group_list].groupby(['air_store_id', 'visit_datetime'], as_index=False).sum().rename(
        columns={'visit_datetime': 'visit_date'}
    )

    df['reserve_datetime_diff'] = df.apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days,
                                                       axis=1)
    tmp1 = df.groupby(['air_store_id', 'visit_datetime'], as_index=False)[
        ['reserve_datetime_diff', 'reserve_visitors']].sum().rename(
        columns={'visit_datetime': 'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors': 'rv1'})
    tmp2 = df.groupby(['air_store_id', 'visit_datetime'], as_index=False)[
        ['reserve_datetime_diff', 'reserve_visitors']].mean().rename(
        columns={'visit_datetime': 'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors': 'rv2'})
    df = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id', 'visit_date'])

    df = pd.merge(df, reserve, on=['air_store_id', 'visit_date'], how='left')
    del reserve
    return df

air_reserve = deal_with_reserve_info(air_reserve)
hpg_reserve = deal_with_reserve_info(hpg_reserve)
data = pd.concat([train, test], axis=0)

store_ids = set(train['air_store_id'])

store_with_no_visitor_rate = []
for x in tqdm(list(store_ids)):
    tt = train[train['air_store_id'] == x]
    date_min = tt['visit_date'].min()
    date_max = tt['visit_date'].max()
    date_range = pd.date_range(date_min, date_max, freq='D')
    rate = len(set(date_range) - set(tt['visit_date'])) / len(set(date_range))
    store_with_no_visitor_rate.append(rate)

tt2 = pd.DataFrame({'air_store_id': list(store_ids), 'no_visit_rate': store_with_no_visitor_rate})
tt2.to_csv('processed/store_with_no_visitor_rate.csv', index=None)

del train; del test; del tt2

data = pd.merge(data, start_date, on='air_store_id', how='left')
data['days_from_start_date'] = data.apply(lambda x: (x['visit_date'] - x['start_date']).days, axis=1)

# should add this feature?
data['date_int'] = data['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)

data['latitude_plus_longitude'] = data['longitude'] + data['latitude']

data['reserve_visitors'] = data['reserve_visitors_x'] + data['reserve_visitors_y']
#air_reserve.drop(['reserve_visitors_x', 'reserve_visitors_y'], axis=1, inplace=True)
data['total_reserv_sum'] = data['rv1_x'] + data['rv1_y']
data['total_reserv_mean'] = (data['rv2_x'] + data['rv2_y']) / 2
data['total_reserv_dt_diff_mean'] = (data['rs2_x'] + data['rs2_y']) / 2

data['air_store_id_encode'] = le.fit_transform(data['air_store_id'])

# # is need to add max, min, std ? only with train set
# store_sales = data[data['set'] == 0].groupby(['air_store_id', 'day_of_week']).agg(
#     {'visitors': [np.mean, np.median, np.std, np.min, np.max]}).reset_index()
# store_sales.columns = ['air_store_id', 'day_of_week', 'mean_visitors', 'median_visitors'
#                        , 'std_visitors', 'min_visitors', 'max_visitors']
#
# # data = pd.merge(data, store_sales, how='left', on=['air_store_id', 'day_of_week'])
#
# data.to_csv('processed/data.csv', index=None)
# print('data: ', data.columns)
# # del data
# # 2.0
# ############################################################
# # extract infomation from store info
# air_store_info = pd.read_csv('data/air_store_info.csv')
# air_hpg_id = pd.read_csv('data/store_id_relation.csv')
# # location cluster
# cluster = KMeans(n_clusters=5)
# air_store_info['loc_labels'] = cluster.fit_predict(air_store_info[['longitude', 'latitude']])
# loc_label_dit = dict(air_store_info['loc_labels'].value_counts())
# air_store_info['loc_store_cnt'] = air_store_info['loc_labels'].map(loc_label_dit)
# air_store_info = pd.merge(air_store_info, air_hpg_id, on='air_store_id', how='left')
# air_store_info['air_and_hpg'] = ~air_store_info['hpg_store_id'].isnull()
# air_store_info['air_and_hpg'] = air_store_info['air_and_hpg'].astype('int')
# air_store_info = pd.merge(store_sales, air_store_info, on='air_store_id', how='left')
# air_store_info['latitude_plus_longitude'] = air_store_info['longitude'] + air_store_info['latitude']
#
# air_store_info['var_max_lat'] = air_store_info['latitude'].max() - air_store_info['latitude']
# air_store_info['var_max_long'] = air_store_info['longitude'].max() - air_store_info['longitude']
# # hpg_store_info = pd.read_csv('data/hpg_store_info.csv')
# # print(len(set(air_store_info['air_store_id'])), len(set(air_store_info['air_store_id']) - set(air_hpg_id['air_store_id'])))
# # print(len(set(hpg_store_info['hpg_store_id'])), len(set(hpg_store_info['hpg_store_id']) - set(air_hpg_id['hpg_store_id'])))
# # hpg_store_info = pd.merge(hpg_store_info, air_hpg_id, on='hpg_store_id', how='left')
# # air_store_info = pd.merge(air_store_info, hpg_store_info, on='air_store_id', how='left')
# # air_store_info.to_csv('processed/air_hpg_store_info.csv', index=None)
#
# air_store_info['air_genre_name'] = air_store_info['air_genre_name'].map(lambda x: str(str(x).replace('/', ' ')))
# air_store_info['air_area_name'] = air_store_info['air_area_name'].map(lambda x: str(str(x).replace('-', ' ')))
#
# air_store_info['genre_size'] = air_store_info['air_genre_name'].apply(lambda x: len(x.split(' ')))
# air_store_info['area_size'] = air_store_info['air_area_name'].apply(lambda x: len(x.split(' ')))
#
# print(air_store_info['genre_size'].unique())
# print(air_store_info['area_size'].unique())
#
# le = LabelEncoder()
#
# for i in range(3):
#     air_store_info['air_genre_name' + str(i)] = le.fit_transform(
#         air_store_info['air_genre_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' ')) > i else ''))
# for i in range(7):
#     air_store_info['air_area_name' + str(i)] = le.fit_transform(
#         air_store_info['air_area_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' ')) > i else ''))
#
# air_store_info['air_genre_name'] = le.fit_transform(air_store_info['air_genre_name'])
# air_store_info['air_area_name'] = le.fit_transform(air_store_info['air_area_name'])
# air_store_info['air_store_id_encode'] = le.fit_transform(air_store_info['air_store_id'])
#
# air_store_info.to_csv('processed/air_store_info.csv', index=None)
#
# # print('store: ', air_store_info.columns)
# air_store_info.drop('hpg_store_id', axis=1, inplace=True)
# data = pd.merge(data, air_store_info, on=['air_store_id', 'day_of_week'], how='left')
# del air_store_info
#
# # 3.0
# ##############################################################
# # extract infomation from reserve info
# air_reserve = pd.read_csv('data/air_reserve.csv')
# hpg_reserve = pd.read_csv('data/hpg_reserve.csv')
# hpg_reserve = pd.merge(hpg_reserve, air_hpg_id, how='inner', on=['hpg_store_id'])
# del air_hpg_id
#
# def deal_with_reserve_info(df):
#     df['visit_datetime'] = pd.to_datetime(df['visit_datetime'])
#     df['visit_datetime'] = df['visit_datetime'].dt.date
#     df['reserve_datetime'] = pd.to_datetime(df['reserve_datetime'])
#     df['reserve_datetime'] = df['reserve_datetime'].dt.date
#
#     df['hour_gap'] = df['visit_datetime'].sub(df['reserve_datetime'])
#     df['hour_gap'] = df['hour_gap'].apply(lambda x: x / np.timedelta64(1, 'h'))
#     # separate reservation into 5 categories based on gap lenght
#     df['reserve_-12_h'] = np.where(df['hour_gap'] <= 12,
#                                         df['reserve_visitors'], 0)
#     df['reserve_12_37_h'] = np.where((df['hour_gap'] <= 37) & (df['hour_gap'] > 12),
#                                           df['reserve_visitors'], 0)
#     df['reserve_37_59_h'] = np.where((df['hour_gap'] <= 59) & (df['hour_gap'] > 37),
#                                           df['reserve_visitors'], 0)
#     df['reserve_59_85_h'] = np.where((df['hour_gap'] <= 85) & (df['hour_gap'] > 59),
#                                           df['reserve_visitors'], 0)
#     df['reserve_85+_h'] = np.where((df['hour_gap'] > 85),
#                                         df['reserve_visitors'], 0)
#     # group by air_store_id and visit_date to enable joining with main table
#     group_list = ['air_store_id', 'visit_datetime', 'reserve_visitors', 'reserve_-12_h',
#                   'reserve_12_37_h', 'reserve_37_59_h', 'reserve_59_85_h', 'reserve_85+_h']
#     reserve = df[group_list].groupby(['air_store_id', 'visit_datetime'], as_index=False).sum().rename(
#         columns={'visit_datetime': 'visit_date'}
#     )
#
#     df['reserve_datetime_diff'] = df.apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days,
#                                                        axis=1)
#     tmp1 = df.groupby(['air_store_id', 'visit_datetime'], as_index=False)[
#         ['reserve_datetime_diff', 'reserve_visitors']].sum().rename(
#         columns={'visit_datetime': 'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors': 'rv1'})
#     tmp2 = df.groupby(['air_store_id', 'visit_datetime'], as_index=False)[
#         ['reserve_datetime_diff', 'reserve_visitors']].mean().rename(
#         columns={'visit_datetime': 'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors': 'rv2'})
#     df = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id', 'visit_date'])
#
#     df = pd.merge(df, reserve, on=['air_store_id', 'visit_date'], how='left')
#     del reserve
#     return df
#
# air_reserve = deal_with_reserve_info(air_reserve)
# hpg_reserve = deal_with_reserve_info(hpg_reserve)
#
# air_reserve = pd.merge(air_reserve, hpg_reserve, on=['air_store_id', 'visit_date'], how='left')
# air_reserve['reserve_visitors'] = air_reserve['reserve_visitors_x'] + air_reserve['reserve_visitors_y']
# #air_reserve.drop(['reserve_visitors_x', 'reserve_visitors_y'], axis=1, inplace=True)
# air_reserve['total_reserv_sum'] = air_reserve['rv1_x'] + air_reserve['rv1_y']
# air_reserve['total_reserv_mean'] = (air_reserve['rv2_x'] + air_reserve['rv2_y']) / 2
# air_reserve['total_reserv_dt_diff_mean'] = (air_reserve['rs2_x'] + air_reserve['rs2_y']) / 2
# # cols = ['reserve_-12_h', 'reserve_12_37_h', 'reserve_37_59_h', 'reserve_59_85_h', 'reserve_85+_h']
# # for c in cols:
# #     air_reserve[c] = air_reserve['%s_x' % c] + air_reserve['%s_y' % c]
# #     air_reserve.drop('%s_x' % c, axis=1, inplace=True)
# #     air_reserve.drop('%s_y' % c, axis=1, inplace=True)
# print('reserve: ', air_reserve.columns)
# air_reserve.to_csv('processed/air_reserve_info.csv', index=None)
# data = pd.merge(data, air_reserve, on=['air_store_id', 'visit_date'], how='left')
# del air_reserve
# # 4.0
# ###################################################################
# # extract infomation from holiday info
# holiday = pd.read_csv('data/date_info.csv').rename(columns={'calendar_date': 'visit_date'})
# holiday['visit_date'] = pd.to_datetime(holiday['visit_date'])
# holiday['visit_date'] = holiday['visit_date'].dt.date
# holiday['is_previous_holiday'] = holiday['holiday_flg'].shift(1).fillna(0)
# holiday['is_next_holiday'] = holiday['holiday_flg'].shift(-1).fillna(0)
#
# print('holiday: ', holiday.columns)
# holiday.to_csv('processed/holiday_info.csv', index=None)
# holiday.drop('day_of_week', axis=1, inplace=True)
# data = pd.merge(data, holiday, on='visit_date', how='left')
# del holiday

# there just fillna with zero
data = data.fillna(-1)

train = data[data['set'] == 0]
test = data[data['set'] == 1]
del data
train.drop('set', axis=1, inplace=True)
test.drop('set', axis=1, inplace=True)

cols = [x for x in train.columns if x not in ['air_store_id', 'visitors', 'visit_date', 'start_date']]
train[['air_store_id'] + cols].head(100).to_csv('processed/train_lgb_1.csv', index=None)
print(cols)
X_train = train[cols].values
y = np.log1p(train['visitors'].values)
X_test = test[cols].values

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

train = shuffle(train, random_state=2018)

kf = KFold(n_splits=10)

def RMSLE(y, pred):
    return mean_squared_error(y, pred) ** 0.5

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

num_rounds = 8000
idx = 1
for train_idx, test_idx in kf.split(X_train):
    print('-' * 50)
    print('fold %d' % idx)
    idx += 1
    X_tr, X_te = X_train[train_idx], X_train[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    evals_result = {}  # to record eval results for plotting
    dtrain = lgb.Dataset(X_tr, y_tr)
    dval = lgb.Dataset(X_te, y_te)
    bst = lgb.train(
        params, dtrain, num_boost_round=num_rounds,
        feature_name=cols, evals_result=evals_result,
        valid_sets=[dtrain, dval], early_stopping_rounds=50, verbose_eval=50
    )

    print('Plot metrics during training...')
    ax = lgb.plot_metric(evals_result, metric='rmse')
    plt.show()

    print('Plot feature importances...')
    ax = lgb.plot_importance(bst, max_num_features=30)
    plt.show()

    err_tr = RMSLE(y_tr, bst.predict(X_tr, num_iteration=bst.best_iteration or num_rounds))
    err_te = RMSLE(y_te, bst.predict(X_te, num_iteration=bst.best_iteration or num_rounds))
    print('RMSE train: ', err_tr)
    print('RMSE validate: ', err_te)