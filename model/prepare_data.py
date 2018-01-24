import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('../data/air_visit_data.csv', parse_dates=['visit_date'])
test = pd.read_csv('../data/sample_submission.csv')
holiday = pd.read_csv('../data/date_info.csv').rename(columns={'calendar_date': 'visit_date'})
ar = pd.read_csv('../data/air_reserve.csv').rename(columns={'visit_datetime': 'visit_date'})
hr = pd.read_csv('../data/hpg_reserve.csv').rename(columns={'visit_datetime': 'visit_date'})

start_date = train.groupby(['air_store_id'])['visit_date'].min()
test['air_store_id'] = test['id'].apply(lambda x: '_'.join(x.split('_')[:-1]))
test['visit_date'] = test['id'].apply(lambda x: x.split('_')[-1])
test['visit_date'] = pd.to_datetime(test['visit_date'])
test['id'] = range(1, len(test) + 1)
train['id'] = 0
test_start = test['visit_date'].min()

df = pd.concat([train, test], axis=0)

holiday['visit_date'] = holiday['visit_date'].apply(lambda x: pd.to_datetime(x).date())
holiday['visit_date'] = holiday['visit_date'].apply(lambda x: pd.to_datetime(x))
df = pd.merge(df, holiday, on='visit_date', how='left')
del train, test
print('loaded data')

# dates
date_range = pd.date_range(df['visit_date'].min(), df['visit_date'].max(), freq='D')
date_idx = [x for x in range(len(date_range))]
dt_to_idx = dict(map(reversed, enumerate(date_range)))
test_start_idx = dt_to_idx[test_start]
df['visit_date'] = df['visit_date'].map(dt_to_idx.get)
missing_dates = list(set(date_idx) - set(df['visit_date']))

# pivot and reindex
df = df.pivot_table(index=['air_store_id'], columns='visit_date')
fill = np.zeros([df.shape[0], len(missing_dates)])
fill[:] = np.nan
missing_df = pd.DataFrame(columns=missing_dates, data=fill)

holi = pd.concat([df['holiday_flg'].reset_index(), missing_df], axis=1).fillna(0)
holi = holi[['air_store_id'] + date_idx]
holi = holi[date_idx].values.astype(np.int8)

uid = pd.concat([df['id'].reset_index(), missing_df], axis=1).fillna(0)
uid = uid[['air_store_id'] + date_idx]

df = pd.concat([df['visitors'].reset_index(), missing_df], axis=1).fillna(0)
df = df[['air_store_id'] + date_idx]

if not os.path.isdir('data/processed'):
    os.makedirs('data/processed')

np.save('data/processed/holidayinfo.npy', holi.astype(np.int8))
np.save('data/processed/x_raw.npy', df[date_idx].values.astype(np.float16))
np.save('data/processed/id.npy', uid[date_idx].values.astype(np.int32))
print('pivoted')

df[date_idx] = np.log(np.maximum(df[date_idx].values, 0) + 1)
df[date_idx] = df[date_idx].astype(np.float16)
np.save('data/processed/x.npy', df[date_idx].values)

# non-temporal features
start_date = start_date.reset_index().rename(columns={'visit_date': 'start_date'})
start_date['start_date'] = start_date['start_date']
df = df.merge(start_date, how='left', on=['air_store_id'])
df['start_date'] = df['start_date'].map(lambda x: dt_to_idx.get(x, test_start_idx))
del start_date

data = {
    'as': pd.read_csv('../data/air_store_info.csv'),
    'hs': pd.read_csv('../data/hpg_store_info.csv'),
    'ar': pd.read_csv('../data/air_reserve.csv'),
    'hr': pd.read_csv('../data/hpg_reserve.csv'),
    'id': pd.read_csv('../data/store_id_relation.csv'),
    'hol': pd.read_csv('../data/date_info.csv').rename(columns={'calendar_date': 'visit_date'})
}

# for col in ['ar', 'hr']:
#     data[col]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
#     data[col]['visit_datetime'] = data[df]['visit_datetime'].dt.date
#     data[col]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
#     data[col]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
#     data[col]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days,
#                                                        axis=1)
#     tmp1 = data[col].groupby(['air_store_id', 'visit_datetime'], as_index=False)[
#         ['reserve_datetime_diff', 'reserve_visitors']].sum().rename(
#         columns={'visit_datetime': 'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors': 'rv1'})
#     tmp2 = data[col].groupby(['air_store_id', 'visit_datetime'], as_index=False)[
#         ['reserve_datetime_diff', 'reserve_visitors']].mean().rename(
#         columns={'visit_datetime': 'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors': 'rv2'})
#     data[col] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id', 'visit_date'])
#
# for col in ['ar', 'hr']:
#     df = pd.merge(train, data[col], how='left', on=['air_store_id', 'visit_date'])

stores = pd.read_csv('../data/air_store_info.csv')
stores['air_genre_name'] = stores['air_genre_name'].map(lambda x: str(str(x).replace('/', ' ')))
stores['air_area_name'] = stores['air_area_name'].map(lambda x: str(str(x).replace('-', ' ')))
lbl = LabelEncoder()
stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

df = df.merge(stores, how='left', on='air_store_id')
del stores

lbl = LabelEncoder()
df['air_store_id2'] = lbl.fit_transform(df['air_store_id'])

features = [
    ('air_store_id2', np.int16),
    ('air_area_name', np.int16),
    ('air_genre_name', np.int8),
    ('start_date', np.int16),
    ('latitude', np.float),
    ('longitude', np.float)
]

for feature, dtype in features:
    vals = df[feature].values.astype(dtype)
    np.save('data/processed/{}.npy'.format(feature), vals)
print('finished non-temporal features')

# lags
x = df[date_idx].values

x_lags = [1, 7, 14]
lag_data = np.zeros([x.shape[0], x.shape[1], len(x_lags)], dtype=np.float16)

for i, lag in enumerate(x_lags):
    lag_data[:, lag:, i] = x[:, :-lag]

np.save('data/processed/x_lags.npy', lag_data)
del lag_data

xy_lags = [16, 21, 28, 35, int(365/4), int(365/2), 365]
lag_data = np.zeros([x.shape[0], x.shape[1], len(xy_lags)], dtype=np.float16)

for i, lag in enumerate(xy_lags):
    lag_data[:, lag:, i] = x[:, :-lag]

np.save('data/processed/xy_lags.npy', lag_data)
del lag_data

