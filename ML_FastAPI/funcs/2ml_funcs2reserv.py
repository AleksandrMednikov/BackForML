import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import SARIMAX, AutoReg
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima.utils import ndiffs
import datetime
from dateutil.relativedelta import *


# from google.colab import files
# import io

def oleg_func_linear(file, user_key,
                     promo1, promo2, promo3, promo4,
                     out_of_stock1, out_of_stock2, out_of_stock3, out_of_stock4,
                     epidemic1, epidemic2, epidemic3, epidemic4, file_type,
                     forecast_periods=24):

    """

    :param file: файл с наличием страниц
    :param user_key: фиктивная переменая, данная функция делалась как GUI для конкретного человека,
    однако если предположить что ей попытаются несколько людей пользоватся через сервер, то эта переменая
    будет их разделять
    :param promo1:
    :param promo2:
    :param promo3:
    :param promo4:
    :param out_of_stock1:
    :param out_of_stock2:
    :param out_of_stock3:
    :param out_of_stock4:
    :param epidemic1:
    :param epidemic2:
    :param epidemic3:
    :param epidemic4:
    :param forecast_periods:число месяцев
    :return:
    """

    if file_type == "xlsx":
        data = pd.read_excel(file, sheet_name='Третичные продажи AlfaRm')# читаем данные
    if file_type == "csv":
        data = pd.read_csv(file, sheet_name='Третичные продажи AlfaRm')  # читаем данные

    data = data.fillna(0)# заполняем пропуски нулями
    data = data.query('sku == "FLUIFORT granules"')  # выбираем бренд
    data['packs'] = data['packs'].astype('int')  # меняем тип данных в packs
    data.index = pd.to_datetime(data['year'].astype(str) + '-' + data['month'].astype(str),
                                format='%Y-%m')  # делаем даты индексами
    data.sort_index(inplace=True)  # сортируем индексы/даты на случай если где-то они идут не по порядку
    data = data.drop(['sku', 'year', 'month', 'period_key', 'last_month_of_quart'],
                     axis=1)  # с таким  дропом лучшие метрики

    # подготовка признаков max_lag, rolling_mean_size. сделал 4 как в оригинале
    data['month'] = data.index.month
    data['year'] = data.index.year
    for lag in range(1, 4 + 1):
        data['lag_{}'.format(lag)] = data['packs'].shift(lag)
    data['rolling_mean'] = data['packs'].shift().rolling(4).mean()

    data['out_of_stock'] = data['out_of_stock'].astype('int')  # changing data types
    data['promo'] = data['promo'].astype('int')  # changing data types

    data = data.drop(['word_stat'], axis=1)

    # Разбивка на выборки
    train, test = train_test_split(data, shuffle=False, test_size=0.1)
    train = train.dropna()
    features_train = train.drop(['packs'], axis=1)
    target_train = train['packs']
    features_test = test.drop(['packs'], axis=1)
    target_test = test['packs']

    # сохраняем копии выборок
    features_lr_train = features_train.copy()
    features_lr_test = features_test.copy()

    # обучение и предсказание модели
    model = LinearRegression()
    model.fit(features_lr_train, target_train)
    predictions_lr = model.predict(features_lr_test)
    # метрики
    mae_lr_train = mean_absolute_error(target_train, model.predict(features_lr_train))
    mae_lr_test = mean_absolute_error(target_test, predictions_lr)

    # результаты для клиента, точность прогноза на валидации
    result_for_client_month = f'Точность на 1-месячном отрезке {(1 - mae_lr_test / target_test.mean()):.1%}'
    result_for_client_allval = f'Точность на всем валидационном отрезке {(1 - abs(sum(target_test) - sum(predictions_lr)) / sum(target_test)):.1%}'

    sns.set(rc={"figure.figsize": (12, 7)})

    # функция для сохранения графиков сравнения target / prediction
    def plot_predict(df_prediction, target_train, save_name):
        plt.figure(figsize=(9, 3))
        plt.plot(target_train, label='true')
        plt.plot(temp_df, label='pred')
        plt.xlabel('date')
        plt.ylabel('value')
        plt.title('Original / Predicted series')
        plt.legend()
        plt.savefig("results/"+save_name)

    # сравним предсказания с реальными данными на обучении
    temp_df = pd.DataFrame(model.predict(features_train), index=target_train.index)
    plot_predict(temp_df, target_train, f'user_{user_key}_lr_train.png')
    # сравним предсказания на тестовой выборке с реальными значениями
    temp_df = pd.DataFrame(model.predict(features_test), index=target_test.index)
    plot_predict(temp_df, target_test, f'user_{user_key}_lr_test.png')

    #  ПОСТАНОВЛЕНИЕ ПРОГНОЗОВ МОДЕЛЬЮ ARIMA
    epidemic = data['epidemic']

    p = 2  # number of autoregressive terms (AR order)
    d = 1  # number of nonseasonal differences (differencing order)
    q = 2  # number of moving-average terms (MA order)

    P = 0  # Seasonal autoregressive order
    D = 1  # Seasonal difference order
    Q = 0  # Seasonal moving average
    m = 12  # number of time steps for a single seasonal period

    # Creating SARIMAX model
    my_order = (p, d, q)
    my_seasonal_order = (P, D, Q, m)

    # define model
    mymodel = SARIMAX(epidemic, order=my_order, seasonal_order=my_seasonal_order)
    # mymodel = ARIMA(data, order=my_order)
    modelfit = mymodel.fit()
    predictions_arima = modelfit.predict(dynamic=False)

    # прогнозирование
    predictions_arima = modelfit.predict(dynamic=False)

    epidemic_forecast = modelfit.predict(1, len(epidemic) + forecast_periods - 1)

    plt.figure(figsize=(9, 3))
    plt.plot(epidemic)
    plt.plot(epidemic_forecast)
    plt.xlabel('date')
    plt.ylabel('value')
    plt.title(f'Прогноз на {forecast_periods} месяцев')
    plt.savefig(f'results/user_{user_key}_arima.png')

    # лучшую модель обучаем на всей выборке
    data = data.dropna()
    features = data.drop(['packs'], axis=1)
    target = data['packs']
    model = LinearRegression()
    model.fit(features, target)

    # сравним предсказания с реальными данными на обучении
    temp_df = pd.DataFrame(model.predict(features), index=target.index)
    plot_predict(temp_df, target, f'user_{user_key}_lr_real.png')

    investment = pd.read_excel(file, sheet_name='Флуифорт_инв', index_col='Дата')
    investment.columns = ['budget']

    df = data.tail(4)

    def next_month_prediction(df, model, out_of_stock=0, promo=0, epidemic=None, invest=None):

        next_month = df.index.max() + relativedelta(months=+1)

        if epidemic is None:
            epidemic = epidemic_forecast.loc[next_month]

        if next_month.month in [1, 2, 3, 10, 11, 12]:
            high_season = 1
        else:
            high_season = 0

        if next_month.month in [7, 8]:
            epidemic = 0

        if invest is None:
            try:
                invest = investment.loc[next_month, 'budget']
            except:
                invest = 0

        df.loc[next_month] = [
            None,
            out_of_stock,
            promo,
            high_season,
            epidemic,
            invest,
            next_month.month,
            next_month.year,
            df.loc[df.index.max() + relativedelta(months=0), 'packs'],
            df.loc[df.index.max() + relativedelta(months=-1), 'packs'],
            df.loc[df.index.max() + relativedelta(months=-2), 'packs'],
            df.loc[df.index.max() + relativedelta(months=-3), 'packs'],
            (df.loc[df.index.max() + relativedelta(months=0), 'packs'] +
             df.loc[df.index.max() + relativedelta(months=-1), 'packs'] +
             df.loc[df.index.max() + relativedelta(months=-2), 'packs'] +
             df.loc[df.index.max() + relativedelta(months=-3), 'packs']) / 4
        ]

        df.loc[df.index.max(), 'packs'] = model.predict(df.tail(1).drop(['packs'], axis=1))
        return df


    # прогноз на 1й месяц
    df = next_month_prediction(df, model, out_of_stock=out_of_stock1, promo=promo1, epidemic=epidemic1)

    # прогноз на 2й месяц
    df = next_month_prediction(df, model, out_of_stock=out_of_stock2, promo=promo2, epidemic=epidemic2)

    # прогноз на 3й месяц
    df = next_month_prediction(df, model, out_of_stock=out_of_stock3, promo=promo3, epidemic=epidemic3)

    # прогноз на 4й месяц
    df = next_month_prediction(df, model, out_of_stock=out_of_stock4, promo=promo4, epidemic=epidemic4)

    for i in range(1, forecast_periods - 3):
        df = next_month_prediction(df, model)

    # график история + прогноз
    plt.figure(figsize=(9, 3))
    plt.plot(data['packs'], label='true')
    plt.plot(df['packs'][-forecast_periods:], label='prediction')
    plt.xlabel('date')
    plt.ylabel('value')
    plt.title('Original / Predicted series')
    plt.legend()
    plt.savefig(f'results/user_{user_key}_final.png')

    forecast = pd.DataFrame(df['packs'].tail(forecast_periods))
    forecast.columns = ['FLUIFORT granules']
    forecast.to_csv(f'results/forecast_{user_key}.csv')

"""
Разблочте чтобы протестировать работу функции

oleg_func_linear('prognozes.xlsx', user_key=1,
                 promo1=0, promo2=0, promo3=0, promo4=0,
                 out_of_stock1=0, out_of_stock2=0, out_of_stock3=0, out_of_stock4=0,
                 epidemic1=700, epidemic2=844, epidemic3=728, epidemic4=570,
                 forecast_periods=24)
"""