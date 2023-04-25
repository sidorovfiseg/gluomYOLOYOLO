import os
import json
import pickle

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima

def read_data(user_id, path = ''):
    path_to_folder = os.path.join(path, 'user_data')
    path_to_file = os.path.join(path_to_folder, str(user_id) + '_glucose.csv')
    data = pd.read_csv(path_to_file, delimiter=';', names=['date', 'time', 'Gl'])
    data['date_time'] = pd.to_datetime(data['date'] + ' ' + data['time'], format='%d.%m.%Y %H:%M')
    data['Gl'] = data['Gl'].str.replace(',', '.').astype(float)
    data = data.drop(['date', 'time'], axis=1)
    data = data.set_index('date_time')
    #загрузка данных о времени приема пищи и значениях кбжу продукта
    meal_data = pd.read_csv(path + str(user_id) + '_meal.csv', delimiter=';',
                            names=['date', 'time', 'k', 'b', 'j', 'u'])

    meal_data['date_time'] = pd.to_datetime(meal_data['date'] + ' ' + meal_data['time'], format='%d.%m.%Y %H:%M')

    meal_data = meal_data.drop(['date', 'time'], axis=1)
    meal_data = meal_data.set_index('date_time')

    meal_data[['k', 'b', 'j', 'u']] = meal_data[['k', 'b', 'j', 'u']].apply(
        lambda x: x.str.replace(',', '.').astype(float))
    #объединение данных о времени приема пищи и значениях кбжу продукта с данными уровня глюкозы
    data = pd.merge(data, meal_data, how='left', left_index=True, right_index=True)
    return data


def fit_model_save(user_id, path=''):
    data = read_data(user_id, path)
    #подготовка регрессоров
    data = data.reset_index(drop = True)
    regressors = data[['k','b', 'j', 'u']].fillna(0)

    endog = data['Gl']
    exog = regressors
    #подбор модели
    #model_t = auto_arima(endog, exogenous=exog, start_p=12, start_q=3, max_p=60, max_q=5, max_d=4, seasonal=True,trace=True)
    model = SARIMAX(endog, exog=exog,order = (10,1,2))
    model_fit = model.fit(num_epochs = 150)

    path_to_folder = os.path.join(path, 'user_models')
    path_to_file = os.path.join(path_to_folder, str(user_id) + '_model.pickle')

    with open(path_to_file, 'wb') as f:
        pickle.dump(model_fit, f)
def parse_prediction_data(j_str):
    #j_str = json.loads(j_str)
    user_glucose_data = pd.DataFrame(j_str['glucose'])
    user_meal_data = pd.DataFrame(j_str['events'])

    if (user_glucose_data.empty):
        return pd.DataFrame()

    if(user_meal_data.empty):
        user_meal_data = pd.DataFrame(columns=['datetime','k','b','j','u'])

    user_glucose_data['datetime'] = pd.to_datetime(user_glucose_data['datetime'])
    if not user_meal_data.empty:
        user_meal_data['datetime'] = pd.to_datetime(user_meal_data['datetime'])

    user_glucose_data = user_glucose_data.set_index('datetime')
    user_meal_data = user_meal_data.set_index('datetime')

    data = pd.merge(user_glucose_data, user_meal_data, how='left', left_index=True, right_index=True).fillna(0)
    data = data.reset_index(drop=True)
    return data


def predict(user_id, j_str, path = ''):

    data = parse_prediction_data(j_str)

    if(data.empty):
        return []

    user_glucose_data = data['Gl'].astype(float)
    user_meal_data =  data[['k', 'b' , 'j' , 'u']].astype(float)

    path_to_folder = os.path.join(path, 'user_models')
    path_to_file = os.path.join(path_to_folder, str(1) + '_model.pickle')

    with open(path_to_file, 'rb') as f:
        model_fit = pickle.load(f)
    # update input indexes
    last_ind = len(model_fit.model.endog)
    new_index = pd.RangeIndex(last_ind  ,last_ind + len(user_glucose_data))
    user_glucose_data.index = new_index
    user_meal_data.index = new_index
    # update model history
    model_fit = model_fit.append(user_glucose_data, exog = user_meal_data)
    prediction_size = 12
    Start = last_ind + len(user_glucose_data)
    End = Start + prediction_size
    empty = pd.DataFrame(dict.fromkeys(['k', 'b', 'g', 'u'], [0] * prediction_size))
    # prediction
    p = model_fit.predict(start=Start, end = End - 1, exog = empty)
    predictions = pd.to_numeric(p, errors='coerce')
    return predictions



if __name__ == '__main__':
    js = {
        "user_id": 10001,

        "glucose": [
            {"datetime": "2021-09-20 00:00:05", "Gl": 4.9},
            {"datetime": "2021-09-20 00:05:10", "Gl": 4.7},
            {"datetime": "2021-09-20 00:10:20", "Gl": 4.4},
            {"datetime": "2021-09-20 00:15:30", "Gl": 4.3},
            {"datetime": "2021-09-20 00:20:35", "Gl": 4.1},
            {"datetime": "2021-09-20 00:25:45", "Gl": 4.2},
            {"datetime": "2021-09-20 00:30:55", "Gl": 4.0},
            {"datetime": "2021-09-20 00:35:00", "Gl": 4.2},
            {"datetime": "2021-09-20 00:40:10", "Gl": 4.1},
            {"datetime": "2021-09-20 00:45:20", "Gl": 4.7}
        ],

        "events": [
            {"datetime": "2021-09-20 00:20:35", "k": 1.2, "b": 3.4, "j": 5.6, "u": 7.8},
            {"datetime": "2021-09-20 00:45:20", "k": 2.2, "b": 4.5, "j": 6.7, "u": 8.9}
        ]

    }

    print(predict(2, js))