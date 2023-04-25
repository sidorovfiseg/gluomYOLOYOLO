import json


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima

import pickle

def RMAE(target, prediction):
    offset = 0
    for i, j in zip(target, prediction):
        offset += abs(i - j) / i
    return offset / len(target)

def read_data(user_id, path=''):
    data = pd.read_csv(path + str(user_id) + '_glucose.csv', delimiter=';', names=['date', 'time', 'Gl'])
    data['date_time'] = pd.to_datetime(data['date'] + ' ' + data['time'], format='%d.%m.%Y %H:%M')
    data['Gl'] = data['Gl'].str.replace(',', '.').astype(float)
    data = data.drop(['date', 'time'], axis=1)
    data = data.set_index('date_time')

    # ------------------------------------загрузка данных о времени приема пищи и значениях кбжу продукта
    meal_data = pd.read_csv(path + str(user_id) + '_meal.csv', delimiter=';',
                            names=['date', 'time', 'k', 'b', 'j', 'u'])

    meal_data['date_time'] = pd.to_datetime(meal_data['date'] + ' ' + meal_data['time'], format='%d.%m.%Y %H:%M')

    meal_data = meal_data.drop(['date', 'time'], axis=1)
    meal_data = meal_data.set_index('date_time')

    meal_data[['k', 'b', 'j', 'u']] = meal_data[['k', 'b', 'j', 'u']].apply(
        lambda x: x.str.replace(',', '.').astype(float))
    # -------------------------------------объединение данных о времени приема пищи и значениях кбжу продукта с данными уровня глюкозы
    data = pd.merge(data, meal_data, how='left', left_index=True, right_index=True)
    return data

def fit_model_save(user_id, path=''):
    data = read_data(user_id)
    # -------------------------------------подготовка регрессоров
    data = data.reset_index(drop=True)[:-100]
    regressors = data[['k','b', 'j', 'u']].fillna(0)

    endog = data['Gl']
    exog = regressors
    # -------------------------------------подбор модели
    #model_t = auto_arima(endog, exogenous=exog, start_p=12, start_q=3, max_p=60, max_q=5, max_d=4, seasonal=True,trace=True)

    model = SARIMAX(endog, exog=exog,order = (10,1,2))
    model_fit = model.fit(num_epochs = 10)
    with open(str(user_id) + '_model.pickle', 'wb') as f:
        pickle.dump(model_fit, f)


#fit_model_save(1)

def predict(user_id, j_str):
    j_str = json.dumps(j_str)
    j_str = json.loads(j_str)

    user_glucose_data = pd.DataFrame(j_str['glucose'])
    user_meal_data = pd.DataFrame(j_str['events'])
    user_glucose_data['datetime'] = pd.to_datetime(user_glucose_data['datetime'])
    user_meal_data['datetime'] = pd.to_datetime(user_meal_data['datetime'])
    user_glucose_data = user_glucose_data.set_index('datetime')
    user_meal_data = user_meal_data.set_index('datetime')
    data = pd.merge(user_glucose_data, user_meal_data, how='left', left_index=True, right_index=True).fillna(0)
    data = data.reset_index(drop = True)
    user_glucose_data = data['Gl'].astype(float)
    user_meal_data =  data[['k', 'b' , 'j' , 'u']].astype(float)

    with open(str(user_id) + '_model.pickle', 'rb') as f:
        model_fit = pickle.load(f)

    last_ind = len(model_fit.model.endog)

    new_index = pd.RangeIndex(last_ind  ,last_ind + len(user_glucose_data))

    user_glucose_data.index = new_index
    user_meal_data.index = new_index

    model_fit = model_fit.append(user_glucose_data, exog = user_meal_data)

    Start = last_ind + len(user_glucose_data)
    End = Start + 12

    df = pd.DataFrame({'k': [0] * 12,'b': [0] * 12,'g': [0] * 12,'u': [0] * 12})

    p = model_fit.predict(start=Start, end = End - 1, exog = df)

    predictions = pd.to_numeric(p, errors='coerce')

    return predictions


def parse_pred_data(data):
    glucose_df = pd.DataFrame(data['glucose'], columns=['date_time', 'Gl'])
    events_df = pd.DataFrame(data['events'], columns=['date_time', 'k', 'b', 'j', 'u'])

    glucose_df = glucose_df.set_index('date_time')
    events_df = events_df.set_index('date_time')

    data = pd.merge(glucose_df, events_df, how='left', left_index=True, right_index=True).fillna(0)
    data = data.reset_index(drop = True)
    print(data)
    return data



#predict(2,'')

#plt.legend()
#plt.show()