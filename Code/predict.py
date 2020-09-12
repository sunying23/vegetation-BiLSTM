import numpy as np
import os
import random
import math
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, Callback
import matplotlib.pyplot as plt2



def load_data(csv_data, seq_len, cellsNum, valpercent):
    sequence_length = seq_len + 1
    result = []
    #print(i)
    tempData = csv_data
    for index in range(len(tempData) - sequence_length):
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
    model.add(LSTM(200, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(200, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(200, return_sequences=False))
    model.add(Dense(1))
    model.add(Activation('softsign'))
    model.compile(loss='mse', optimizer='adam')
    return model


def learning_rate_schedule(epoch):
    lr_base = 1e-2
    epochs = 100
    lr_power = 0.9
    lr = lr_base * ((1 - float(epoch) / epochs) ** lr_power)
    print('learning rate:', lr)
    return float(lr)

def main():
    csv_dir = '/media/ubuntu/a1f55ab9-3d8e-4d5e-bb92-609fddf79c72/LDZ/LSTM/NDVI/Croplands/Data/test_2009'
    cellsNum = 20
    #csv_data = get_data(csv_dir, cellsNum)

    window = 90
    model = build_model([6, window])
    filepath = r"G:\MODIS\Models\{epoch:03d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, period=1)
    reduce_lr = LearningRateScheduler(learning_rate_schedule)
    model.summary()

    model.load_weights("/media/ubuntu/a1f55ab9-3d8e-4d5e-bb92-609fddf79c72/LDZ/LSTM/NDVI/Croplands/Models/100.hdf5")

    csvData = np.zeros((365, 7))
    cellsCount = 0
    for filename in os.listdir(csv_dir):
        csv_fname = csv_dir + '/' + filename
        csv_fread = open(csv_fname, "rb")
        tempData = np.loadtxt(csv_fread, delimiter=",", skiprows=1)
        csvData[:, :] = tempData[:, 1:]
        csv_fread.close()
        cellsCount += 1
        print(cellsCount)

        # 500
        csvData[:, 1] = csvData[:, 1] / 1.668500103553136071e+01
        csvData[:, 2] = csvData[:, 2] / 1.042839482421875000e+05
        csvData[:, 3] = csvData[:, 3] / 4.032111768722534180e+02
        csvData[:, 4] = csvData[:, 4] / 3.162059631347656250e+02
        csvData[:, 5] = csvData[:, 5] / 3.025696716308593750e+02
        csvData[:, 6] = csvData[:, 6] / 1.622155064511331446e+01

        x_train, y_train = load_data(csvData, window, 1, 0)
        print("x_test", x_train.shape)
        print("y_test", y_train.shape)

        p = model.predict(x_train)
        output = np.hstack((y_train, p))
        out_fname = '/media/ubuntu/a1f55ab9-3d8e-4d5e-bb92-609fddf79c72/LDZ/LSTM/NDVI/Croplands/Cropland_Pred_2009/LSTM/' + filename
        headers = ['GT', 'Pred']
        with open(out_fname, 'w', newline='')as f:
            f_csv = csv.writer(f)
            f_csv.writerow(headers)
            f_csv.writerows(output)


if __name__ == '__main__':
    main()
