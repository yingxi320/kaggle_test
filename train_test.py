import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

data = pd.read_csv('processed/data.csv')

store_with_no_visit_rate = pd.read_csv('processed/store_with_no_visitor_rate.csv')
data = pd.merge(data, store_with_no_visit_rate, on='air_store_id', how='left')
del store_with_no_visit_rate

air_store_info = pd.read_csv('processed/air_store_info.csv')
air_store_info.drop('hpg_store_id', axis=1, inplace=True)
data = pd.merge(data, air_store_info, on='air_store_id', how='left')
del air_store_info

air_reserve = pd.read_csv('processed/air_reserve_info.csv')
data = pd.merge(data, air_reserve, on=['air_store_id', 'visit_date'], how='left')
del air_reserve

holiday = pd.read_csv('processed/holiday_info.csv')
holiday.drop('day_of_week', axis=1, inplace=True)
data = pd.merge(data, holiday, on='visit_date', how='left')

total_sum = data.groupby(['air_store_id'], as_index=False).sum().rename(columns={'visitors': 'total_visitors'})
data = pd.merge(data, total_sum[['air_store_id', 'total_visitors']], on='air_store_id', how='left')
del total_sum

visitors_groupby_week = data.groupby(['air_store_id', 'day_of_week'], as_index=False).sum().rename(columns={'visitors': 'week_visitors'})
data = pd.merge(data, visitors_groupby_week[['air_store_id', 'week_visitors']], on='air_store_id', how='left')

del visitors_groupby_week
visitors_groupby_holi = data.groupby(['air_store_id', 'holiday_flg'], as_index=False).sum().rename(columns={'visitors': 'holi_visitors'})
data = pd.merge(data, visitors_groupby_holi[['air_store_id', 'holi_visitors']], on='air_store_id', how='left')
del visitors_groupby_holi
data['week_visitors_rate'] = data.apply(lambda x: x['week_visitors'] / x['total_visitors'], axis=1)
data['holi_visitors_rate'] = data.apply(lambda x: x['holi_visitors'] / x['total_visitors'], axis=1)

data.drop(['total_visitors', 'week_visitors', 'holi_visitors'], axis=1, inplace=True)


del holiday

# there just fillna with zero
data = data.fillna(0)

train = data[data['set'] == 0]
test = data[data['set'] == 1]
del data
train.drop('set', axis=1, inplace=True)
test.drop('set', axis=1, inplace=True)

cols = [x for x in train.columns if x not in ['air_store_id', 'visitors', 'visit_date', 'start_date']]

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
