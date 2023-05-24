#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from math import sqrt
from numpy import concatenate
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report


plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False



def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg



def compare_predict_real(model, test_X, test_y,dataset,scaler):


    test_y = test_y.reshape((len(test_y), 2))

    temp_index = np.arange(500)

    import win32api, win32con
    import PyQt5
    # from IPython import get_ipython
    # get_ipython().run_line_magic('matplotlib', 'notebook')

    from IPython import get_ipython
    ipy = get_ipython()
    if ipy is not None:
        ipy.run_line_magic('matplotlib', 'inline')

    plt.figure(figsize=(100, 30))
    plt.axis([0, 500, 800, 1000])
    plt.ion()
    plt.legend()
    plt.grid()
    xs = [0, 0]
    ys = [1, 1]
    ys_0 = [1, 1]


    for i in range(350):
        temp_1 = test_X[i:i + 1, :]
        y_predict = model.predict(temp_1)
        temp_1 = temp_1.reshape(temp_1.shape[0], temp_1.shape[2])
        temp_2 = test_y[i:i + 1, :]

        inv_y_predict = concatenate((temp_1[:, 21:27], y_predict[:, [0]]), axis=1)
        inv_y_predict = scaler.inverse_transform(inv_y_predict)
        inv_y_predict = inv_y_predict[:, -1]

        inv_y = concatenate((temp_1[:, 21:27], temp_2[:, [0]]), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:, -1]

        if i == 0:
            xs[1] = dataset.index[i]
            ys[1] = inv_y_predict
            ys_0[1] = inv_y
        else:
            xs[0] = xs[1]
            ys[0] = ys[1]
            ys_0[0] = ys_0[1]

        if i == 100:
            inv_y = inv_y + 15

        xs[1] = temp_index[i]
        ys[1] = inv_y_predict
        ys_0[1] = inv_y

        plt.plot(xs, ys, color='red', label='预测分解炉温度')
        plt.plot(xs, ys_0, color='green', label='实际分解炉温度')

        if (inv_y - inv_y_predict) > 15:
            win32api.MessageBox(0, "分解炉温度异常，请检查！\n分解炉当前温度：" + str(inv_y) + "\n模型预测温度：" + str(inv_y_predict),
                                "重要提醒", win32con.MB_OK)

        if i == 0:
            plt.legend()

        plt.pause(0.1)
        plt.xlabel('测试集时间点')
        plt.ylabel('分解炉温度')
        plt.title('海螺产线分解炉温度预测模型')
        plt.savefig('海螺产线分解炉温度预测模型')

def model_train(scaled):

    reframed = series_to_supervised(scaled, 3, 2)

    values = reframed.values
    train = values[:2500, :]
    test = values[2500:, :]


    train_X, train_y = train[:, :27], train[:, [27, 34]]
    test_X, test_y = test[:, :27], test[:, [27, 34]]


    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X, train_y, test_X, test_y)
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(2))
    model.compile(loss='mae', optimizer='adam')

    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)

    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    # plt.title('LSTM', fontsize='12')
    # plt.ylabel('loss', fontsize='10')
    # plt.xlabel('epoch', fontsize='10')
    # plt.legend()
    # plt.show()

    return model, test_X, test_y

def parse_data():

    dataset = pd.read_csv('selectPoint.csv', index_col=0,
                          usecols=['time', '分解炉温度', '二次风温', '生料喂料速度t/h', '煤粉喂料速度t/h', '喂煤电机转速', '喂煤秤负载', '喂料阀门开度%'],
                          encoding='gbk')

    dataset.index.name = 'date'

    dataset = pd.DataFrame(dataset, columns=['二次风温', '生料喂料速度t/h', '煤粉喂料速度t/h', '喂煤电机转速', '喂煤秤负载', '喂料阀门开度%', '分解炉温度'])

    values = dataset.values

    values = values.astype('float32')

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # print(scaled)
    return scaled, dataset, scaler

def main():
    scaled, dataset, scaler = parse_data()
    model, test_X, test_y = model_train(scaled)
    compare_predict_real(model, test_X, test_y, dataset, scaler)

if __name__ == '__main__':
    main()


