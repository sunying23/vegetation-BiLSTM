import numpy as np
import os
import random
import math
import csv
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, Callback
import matplotlib.pyplot as plt2



def load_data(csv_data, seq_len, cellsNum, valpercent):
    sequence_length = seq_len + 1
    result = []
    #print(i)
    tempData = csv_data
    for index in range(len(tempData) - seq_len):
        # if tempData[index + seq_len, 0] <= 1 and tempData[index, 0] > 0:
        #     result.append(tempData[index: index + sequence_length])
        result.append(tempData[index: index + sequence_length])

    del tempData
    del csv_data
    result = np.array(result)
    x_train = result[:, :-1, 1:]
    y_train = result[:, -1, 0]
    y_train = y_train.reshape(len(y_train), 1)
    # x_val = result[int(row):, :-1, 1:]
    # y_val = result[int(row):, -1, 0]

    return [x_train, y_train]


def build_model(layers):
    d = 0.5
    model = Sequential()
    model.add(Bidirectional(LSTM(200, input_shape=(layers[1], layers[0]), return_sequences=True)))
    model.add(Dropout(d))
    model.add(BatchNormalization())
    #model.add(Bidirectional(LSTM(200, input_shape=(layers[1], layers[0]), return_sequences=True)))
    #model.add(Dropout(d))
    #model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(200, return_sequences=False)))
    model.add(Dense(1))
    model.add(Activation('softsign'))
    model.compile(loss='mse', optimizer='adam')
    return model


def learning_rate_schedule(epoch):
    lr_base = 1e-2
    epochs = 50
    lr_power = 0.9
    lr = lr_base * ((1 - float(epoch) / epochs) ** lr_power)
    print('learning rate:', lr)
    return float(lr)

def main():
    csv_dir = '/media/ubuntu/a1f55ab9-3d8e-4d5e-bb92-609fddf79c72/LDZ/LSTM/NDVI/Croplands/Data/2009/csv_pixel'
    window = 90
    model = build_model([5, window])
    #model.summary()
    model = load_model("/media/ubuntu/a1f55ab9-3d8e-4d5e-bb92-609fddf79c72/LDZ/LSTM/NDVI/Croplands/Models/2009_DL/080.h5")
    model.summary()
    #model.load_weights("/media/ubuntu/a1f55ab9-3d8e-4d5e-bb92-609fddf79c72/LDZ/LSTM/Needleleaf/Models_BiLSTM/All_200.h5", by_name=True)

    csvData = np.zeros((455, 6))
    cellsCount = 0
    for filename in os.listdir(csv_dir):
        csv_fname = csv_dir + '/' + filename
        csv_fread = open(csv_fname, "rb")
        tempData = np.loadtxt(csv_fread, delimiter=",", skiprows=1)
        csvData[:, :] = tempData[:, [1,2,3,4,5,6]]
        csv_fread.close()
        cellsCount += 1
        print(cellsCount)

        # 500
        csvData[:, 1] = csvData[:, 1] / 2.381100000000000136e+02
        csvData[:, 2] = csvData[:, 2] / 1.042900000000000000e+05
        csvData[:, 3] = csvData[:, 3] / 4.037599999999999909e+02
        csvData[:, 4] = csvData[:, 4] / 3.170299999999999727e+02
        csvData[:, 5] = csvData[:, 5] / 3.038899999999999864e+02
        #csvData[:, 5] = csvData[:, 5] / 1.624500000000000099e+01

        x_train, y_train = load_data(csvData, window, 1, 0)
        print("x_test", x_train.shape)
        print("y_test", y_train.shape)

        p = model.predict(x_train)
        output = np.hstack((y_train, p))
        out_fname = '/media/ubuntu/a1f55ab9-3d8e-4d5e-bb92-609fddf79c72/LDZ/LSTM/NDVI/Croplands/Pred/Croplands_2009_DL/all/' + filename
        headers = ['GT', 'Pred']
        with open(out_fname, 'w', newline='')as f:
            f_csv = csv.writer(f)
            f_csv.writerow(headers)
            f_csv.writerows(output)


if __name__ == '__main__':
    main()
