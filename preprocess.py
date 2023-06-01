import pickle

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima
#pd.set_option('display.max_rows', None)

def parse_data(j_str):
    #j_str = json.loads(j_str)
    user_glucose_data = pd.DataFrame(j_str['glucose'])
    user_meal_data = pd.DataFrame(j_str['meal_events'])
    user_insulin_data = pd.DataFrame(j_str['insulin_events'])

    print(user_insulin_data)

    if not user_glucose_data.empty:
        user_glucose_data['datetime'] = pd.to_datetime(user_glucose_data['datetime'], format='%d.%m.%Y %H:%M')
    else:
        print('Нет данных о глюкозе')
        return pd.DataFrame()

    if not user_meal_data.empty:
        user_meal_data['datetime'] = pd.to_datetime(user_meal_data['datetime'], format='%d.%m.%Y %H:%M')
        user_meal_data = user_meal_data.set_index('datetime')
        user_meal_data.index = user_meal_data.index.floor('5min')
        user_meal_data = user_meal_data[~user_meal_data.index.duplicated(keep='first')]
    else:
        user_meal_data = pd.DataFrame(columns=['datetime', 'k', 'b', 'j', 'u'])
        user_meal_data = user_meal_data.set_index('datetime')

    if not user_insulin_data.empty:
        user_insulin_data['datetime'] = pd.to_datetime(user_insulin_data['datetime'], format='%d.%m.%Y %H:%M')
        user_insulin_data = user_insulin_data.set_index('datetime')
        user_insulin_data.index = user_insulin_data.index.floor('5min')
        user_insulin_data = user_insulin_data[~user_insulin_data.index.duplicated(keep='first')]
    else:
        user_insulin_data = pd.DataFrame(columns=['datetime','count', 'type'])
        user_insulin_data = user_insulin_data.set_index('datetime')

    user_glucose_data = user_glucose_data.set_index('datetime')
    user_glucose_data.index = user_glucose_data.index.floor('5min')
    user_glucose_data = user_glucose_data[~user_glucose_data.index.duplicated(keep='first')]




    data = pd.merge(user_glucose_data, user_meal_data, how='left', left_index=True, right_index=True).fillna(0)
    data = pd.merge(data, user_insulin_data, how='left', left_index=True, right_index=True).fillna(0)
    data = data.reset_index(drop=True)
    return data

def read_data():
    data = pd.read_csv('data/1.2.7.csv', delimiter = ';', names = ['date', 'time', 'Gl', 'info'], encoding = 'cp1251')[1:]
    data['date_time'] = pd.to_datetime(data['date'] + ' ' + data['time'], format='%d.%m.%Y %H:%M')
    data['Gl'] = data['Gl'].replace('-', method='ffill')
    data['Gl'] = data['Gl'].str.replace(',', '.').astype(float)
    data = data.drop(['date', 'time', 'info'], axis=1)
    data = data.set_index('date_time')
    data.index = data.index.floor('5min')
    data = data[~data.index.duplicated(keep='first')]
    # ------------------------------------загрузка данных о времени приема пищи и значениях кбжу продукта
    meal_data = pd.read_csv('data/1.1.7.csv', delimiter=';', names=['date', 'time', 'k', 'b', 'j', 'u'], encoding = 'cp1251')[1:]
    meal_data['date_time'] = pd.to_datetime(meal_data['date'] + ' ' + meal_data['time'], format='%d.%m.%Y %H:%M')
    meal_data = meal_data.drop(['date', 'time'], axis=1)
    meal_data = meal_data.set_index('date_time')
    meal_data[['k', 'b', 'j', 'u']] = meal_data[['k', 'b', 'j', 'u']].apply(lambda x: x.str.replace(',', '.').astype(float))
    meal_data.index = meal_data.index.floor('5min')
    meal_data = meal_data[~meal_data.index.duplicated(keep='first')]
    # ------------------------------------загрузка данных о времени приема препаратов
    insulin_data = pd.read_csv('data/1.3.7.csv', delimiter=';', names=['date', 'time', 'count', 'type'], encoding = 'cp1251')[1 : ]
    insulin_data['date_time'] = pd.to_datetime(insulin_data['date'] + ' ' + insulin_data['time'], format='%d.%m.%Y %H:%M')
    insulin_data = insulin_data.drop(['date', 'time'], axis=1)
    insulin_data = insulin_data.set_index('date_time')
    insulin_data['count'] = insulin_data['count'].str.replace(',', '.').astype(float)
    insulin_data['type'] = insulin_data['type'].str.replace('новорапид', '1')
    insulin_data['type'] = insulin_data['type'].str.replace('туджео', '2')
    insulin_data['type'] = insulin_data['type'].astype(float)
    insulin_data.index = insulin_data.index.floor('5min')
    insulin_data = insulin_data[~insulin_data.index.duplicated(keep='first')]

    data = pd.merge(data, meal_data, how='left', left_index=True, right_index=True).fillna(0)
    data = pd.merge(data, insulin_data, how='left', left_index=True, right_index=True).fillna(0).reset_index(drop=True)
    #data['Gl'] =  StandardScaler().fit_transform(data['Gl'].values.reshape(-1, 1))
    return data

def clustering_meal_data(data, start_size, history_size):
    cluster = pd.DataFrame()
    for i in data.index[start_size:-history_size]:
        if data.loc[i]['k'].any():
            start = i - start_size
            end = i + history_size
            sub_ser = data.loc[start : end].reset_index(drop = True)
            cluster = pd.concat([cluster, sub_ser])
    cluster = cluster.reset_index(drop = True)
    return cluster

def clustering_insulin_data(data, start_size, history_size):
    cluster = pd.DataFrame()
    for i in data.index[start_size:-history_size]:
        if data.loc[i]['count'].any():
            start = i - start_size
            end = i + history_size
            sub_ser = data.loc[start : end].reset_index(drop = True)
            cluster = pd.concat([cluster, sub_ser])
    cluster = cluster.reset_index(drop = True)
    return cluster

def clustering_night_data(data, start_size, history_size):
    size = start_size + history_size
    union = pd.DataFrame()
    i = size
    while i in range(size, len(data) - size):
        chunk = data.iloc[ i - size : i + size]
        if chunk[['k','b','j','u','count','type']].any().any():
            i += 1
        else:
            i+=size
            union = pd.concat([union, chunk], axis = 0)
    union = union.reset_index(drop=True)
    #plt.plot(union['Gl'], 'red')
    #plt.show()
    return union

def fit_meal_model(endog, exog, size, user_id, path = ''):
    model = SARIMAX(endog, exog, order=(5, 0, 4), seasonal_order=(2, 0, 1, size))
    model_fit = model.fit(maxiter = 50)

    path_to_folder = os.path.join(path, 'models')
    path_to_file = os.path.join(path_to_folder, str(user_id) + '_meal_model.pkl')

    with open(path_to_file, 'wb') as f:
      pickle.dump(model_fit, f)

def fit_insulin_model(endog, exog, size, user_id, path = ''):
    model = SARIMAX(endog, exog, order=(7, 0, 4), seasonal_order=(2, 0, 1, size))
    model_fit = model.fit(maxiter = 50)

    path_to_folder = os.path.join(path, 'models')
    path_to_file = os.path.join(path_to_folder, str(user_id) + '_insulin_model.pkl')
    with open(path_to_file, 'wb') as f:
      pickle.dump(model_fit, f)

def fit_night_model(endog, size, user_id, path = ''):
    #model_t = auto_arima(endog, seasonal = False, trace = True)
    #order = model_t.order
    #print(order)
    model = SARIMAX(endog, order = (5,0,2))
    model_fit = model.fit(maxiter = 100)

    path_to_folder = os.path.join(path, 'models')
    path_to_file = os.path.join(path_to_folder, str(user_id) + '_night_model.pkl')
    with open(path_to_file, 'wb') as f:
      pickle.dump(model_fit, f)

def meal_predict(prev_data, exog, start_size, history_size, prediction_size, model_meal):
    last_ind = len(model_meal.model.endog)
    new_index = pd.RangeIndex(last_ind, last_ind + len(prev_data))
    prev_data.index = new_index
    model_meal = model_meal.extend(prev_data['Gl'], exog=prev_data[['k', 'b', 'j', 'u']])
    Start = len(prev_data)
    End = len(prev_data) + prediction_size
    p = model_meal.predict(start=Start, end=End - 1, exog=exog)
    return p

def night_predict(prev_data, start_size, history_size, prediction_size, model_night):
    last_ind = len(model_night.model.endog)
    new_index = pd.RangeIndex(last_ind, last_ind + len(prev_data))
    prev_data_1 = prev_data
    prev_data_1.index = new_index
    model_meta = model_night.extend(prev_data_1['Gl'])
    Start = len(prev_data_1)
    End = len(prev_data_1) + prediction_size
    p = model_meta.predict(start=Start, end=End - 1)
    return p

def predict_future(prev_data, exog, start_size, history_size, prediction_size, model_meal, model_night):
    past_event_index = prev_data.loc[prev_data['k'] != 0].index.to_list()
    if past_event_index:
        past_event_index = past_event_index[-1]
    else:
        past_event_index = 0
    p = pd.DataFrame()
    if past_event_index > prev_data.index[0] + start_size + 15:
        prev_data = prev_data.loc[past_event_index - start_size:]
        p = meal_predict(prev_data, exog, start_size, history_size, prediction_size, model_meal)
    else:
        p1 = night_predict(prev_data, start_size, history_size, prediction_size, model_night)

        exog = exog.assign(Gl = p1 .values)
        exog = exog[['Gl', 'k', 'b', 'j', 'u']]

        last_ind = prev_data.index[-1] + 1
        new_index = pd.RangeIndex(last_ind, last_ind + len(exog))
        exog.index = new_index
        future_event_index = exog.loc[exog['k'] != 0].index.to_list()

        if future_event_index:
            future_event_index = future_event_index[0]
        else:
            # print("GGG")
            print(p1)
            return p1

        meal_prediction_size = prediction_size - (future_event_index - exog.index[0])

        res = pd.concat([prev_data, exog], axis=0).fillna(0)
        history = res.loc[future_event_index - start_size: future_event_index - 1]
        regressors = res.loc[future_event_index: future_event_index + meal_prediction_size][['k', 'b', 'j', 'u']]

        p = meal_predict(history, regressors, start_size, history_size, meal_prediction_size, model_meal)
        p = pd.concat([p1.reset_index(drop=True)[:-meal_prediction_size], p.reset_index(drop=True)], axis=0)
        print(p)
    return p

def RMAE(target, prediction):
    offset = 0
    for i, j in zip(target, prediction):
        offset += abs(i - j) / i
    return offset / len(target)

def fit(user_id, j_str, path = ''):
    path_to_folder = os.path.join(path, 'models')
    path_to_file = os.path.join(path_to_folder, str(user_id) + '_night_model.pkl')
    if os.path.exists(path_to_file):
        return True
    start_size = 10
    history_size = 19
    size = start_size + history_size + 1

    data = parse_data(j_str)
    meal_claster = clustering_meal_data(data, start_size, history_size)[:20 * size]
    insulin_claster = clustering_insulin_data(data, start_size, history_size)

    if(len(data) < 1000 or len(meal_claster) < 20 * size):
        return False
    meal_endog = meal_claster['Gl']
    #insulin_endog = insulin_claster['Gl']
    meal_exog = meal_claster[['k','b','j', 'u']]
    #insulin_exog = insulin_claster['count', 'type']

    fit_night_model(data['Gl'], size, user_id)
    fit_meal_model(meal_endog, meal_exog, size, user_id)
    #fit_insulin_model(insulin_endog, insulin_exog, size, user_id)
    return True

def predict(user_id, j_str, path = ''):
    start_size = 10
    history_size = 19
    size = start_size + history_size + 1
    prediction_size = 20
    path_to_folder = os.path.join(path, 'models')
    path_to_meal = os.path.join(path_to_folder, str(user_id) + '_meal_model.pkl')
    path_to_insulin = os.path.join(path_to_folder, str(user_id) + '_insulin_model.pkl')
    path_to_night = os.path.join(path_to_folder, str(user_id) + '_night_model.pkl')

    data = parse_data(j_str)

    if (len(data) < size):
        print("Недостаточно точек, нужно 30 точек для предсказания, в базе лежит : ", len(data) )
        return []

    data = data[ -size : ][['Gl', 'k', 'b', 'j', 'u']]
    empty_exog  = pd.DataFrame(dict.fromkeys(['k', 'b', 'j', 'u'], [0] * prediction_size))

    if not os.path.exists(path_to_night):
        print("Модель не обучена, использум 0 юзера")
        user_id = 0
        path_to_meal = os.path.join(path_to_folder, str(user_id) + '_meal_model.pkl')
        path_to_insulin = os.path.join(path_to_folder, str(user_id) + '_insulin_model.pkl')
        path_to_night = os.path.join(path_to_folder, str(user_id) + '_night_model.pkl')

    with open(path_to_meal, 'rb') as f:
        model_meal = pickle.load(f)

    print('meal_load')

    with open(path_to_night, 'rb') as f:
        model_night = pickle.load(f)
    p = predict_future(data, empty_exog, start_size, history_size, prediction_size, model_meal, model_night).to_list()
    return p