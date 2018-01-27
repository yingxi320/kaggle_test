import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, date

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

train['set'] = 0
test['set'] = 1

# start_date
start_date = train.groupby(['air_store_id'])['visit_date'].min()
start_date = start_date.reset_index().rename(columns={'visit_date': 'start_date'})
start_date['start_date'] = start_date['start_date']
start_date['start_date_int'] = start_date['start_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)

data = pd.concat([train, test])

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

data['day_of_week'] = data['visit_date'].dt.dayofweek
data['week_of_year'] = data['visit_date'].apply(lambda x: x.isocalendar()[1])
data['month'] = data['visit_date'].dt.month
data['day'] = data['visit_date'].dt.day
data['year'] = data['visit_date'].dt.year

data = pd.merge(data, start_date, on='air_store_id', how='left')
data['days_from_start_date'] = data.apply(lambda x: (x['visit_date'] - x['start_date']).days, axis=1)

# should add this feature?
data['date_int'] = data['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)

# is need to add max, min, std ?
store_sales = data.groupby(['air_store_id', 'day_of_week']).agg(
    {'visitors': [np.mean, np.median, np.std]}).reset_index()
store_sales.columns = ['air_store_id', 'day_of_week', 'mean_visitors', 'median_visitors'
                       , 'std_visitors']

data = pd.merge(data, store_sales, how='left', on=['air_store_id', 'day_of_week'])

data.to_csv('processed/data.csv', index=None)
print('data: ', data.columns)
del data
# 2.0
############################################################
# extract infomation from store info
air_store_info = pd.read_csv('data/air_store_info.csv')
air_hpg_id = pd.read_csv('data/store_id_relation.csv')
air_store_info = pd.merge(air_store_info, air_hpg_id, on='air_store_id', how='left')
air_store_info['air_and_hpg'] = ~air_store_info['hpg_store_id'].isnull()
air_store_info['air_and_hpg'] = air_store_info['air_and_hpg'].astype('int')
air_store_info['latitude_plus_longitude'] = air_store_info['longitude'] + air_store_info['latitude']
# hpg_store_info = pd.read_csv('data/hpg_store_info.csv')
# print(len(set(air_store_info['air_store_id'])), len(set(air_store_info['air_store_id']) - set(air_hpg_id['air_store_id'])))
# print(len(set(hpg_store_info['hpg_store_id'])), len(set(hpg_store_info['hpg_store_id']) - set(air_hpg_id['hpg_store_id'])))
# hpg_store_info = pd.merge(hpg_store_info, air_hpg_id, on='hpg_store_id', how='left')
# air_store_info = pd.merge(air_store_info, hpg_store_info, on='air_store_id', how='left')
# air_store_info.to_csv('processed/air_hpg_store_info.csv', index=None)

air_store_info['air_genre_name'] = air_store_info['air_genre_name'].map(lambda x: str(str(x).replace('/', ' ')))
air_store_info['air_area_name'] = air_store_info['air_area_name'].map(lambda x: str(str(x).replace('-', ' ')))

air_store_info['genre_size'] = air_store_info['air_genre_name'].apply(lambda x: len(x.split(' ')))
air_store_info['area_size'] = air_store_info['air_area_name'].apply(lambda x: len(x.split(' ')))

print(air_store_info['genre_size'].unique())
print(air_store_info['area_size'].unique())

le = LabelEncoder()
air_store_info['air_genre_name'] = le.fit_transform(air_store_info['air_genre_name'])
air_store_info['air_area_name'] = le.fit_transform(air_store_info['air_area_name'])
air_store_info['air_store_id_encode'] = le.fit_transform(air_store_info['air_store_id'])

for i in range(3):
    air_store_info['air_genre_name' + str(i)] = le.fit_transform(
        air_store_info['air_genre_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' ')) > i else ''))
for i in range(7):
    air_store_info['air_area_name' + str(i)] = le.fit_transform(
        air_store_info['air_area_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' ')) > i else ''))
print(air_store_info.head(5))
air_store_info.to_csv('processed/air_store_info.csv', index=None)

print('store: ', air_store_info.columns)
del air_store_info

# 3.0
##############################################################
# extract infomation from reserve info
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
    return reserve

air_reserve = deal_with_reserve_info(air_reserve)
hpg_reserve = deal_with_reserve_info(hpg_reserve)

air_reserve = pd.merge(air_reserve, hpg_reserve, on=['air_store_id', 'visit_date'], how='left')
air_reserve = air_reserve.fillna(0)
air_reserve['reserve_visitors'] = air_reserve['reserve_visitors_x'] + air_reserve['reserve_visitors_y']
air_reserve.drop(['reserve_visitors_x', 'reserve_visitors_y'], axis=1, inplace=True)
cols = ['reserve_-12_h', 'reserve_12_37_h', 'reserve_37_59_h', 'reserve_59_85_h', 'reserve_85+_h']
for c in cols:
    air_reserve[c] = air_reserve['%s_x' % c] + air_reserve['%s_y' % c]
    air_reserve.drop('%s_x' % c, axis=1, inplace=True)
    air_reserve.drop('%s_y' % c, axis=1, inplace=True)
print('reserve: ', air_reserve.columns)
air_reserve.to_csv('processed/air_reserve_info.csv', index=None)
# 4.0
###################################################################
# extract infomation from holiday info
holiday = pd.read_csv('data/date_info.csv').rename(columns={'calendar_date': 'visit_date'})
holiday['is_previous_holiday'] = holiday['holiday_flg'].shift(1).fillna(0)
holiday['is_next_holiday'] = holiday['holiday_flg'].shift(-1).fillna(0)

print('holiday: ', holiday.columns)
holiday.to_csv('processed/holiday_info.csv', index=None)
