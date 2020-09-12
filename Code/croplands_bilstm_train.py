import numpy as np
import os
import random
import math
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, Callback
import matplotlib.pyplot as plt2


def get_data(csv_dir, cellsNum):
    csvData = np.zeros((cellsNum, 455, 6))
    cellsCount = 0
    for filename in os.listdir(csv_dir):
        csv_fname = csv_dir + '/' + filename
        csv_fread = open(csv_fname, "rb")
        tempData = np.loadtxt(csv_fread, delimiter=",", skiprows=1)
        csvData[cellsCount, :, :] = tempData[:, [1,2,3,4,5,6]]
        csv_fread.close()
        cellsCount += 1
        print(cellsCount)
    #dataMax = [csvData[:, :, 1].max(), csvData[:, :, 2].max(), csvData[:, :, 3].max(), csvData[:, :, 4].max(),csvData[:, :, 5].max(), csvData[:, :, 6].max()]
    dataMax = [csvData[:, :, 1].max(), csvData[:, :, 2].max(), csvData[:, :, 3].max(), csvData[:, :, 4].max(),csvData[:, :, 5].max()]
    csvData[:, :, 1] = csvData[:, :, 1] / csvData[:, :, 1].max() #PRE
    csvData[:, :, 2] = csvData[:, :, 2] / csvData[:, :, 2].max() #PRS
    csvData[:, :, 3] = csvData[:, :, 3] / csvData[:, :, 3].max() #SSRA
    csvData[:, :, 4] = csvData[:, :, 4] / csvData[:, :, 4].max() #TMAX
    csvData[:, :, 5] = csvData[:, :, 5] / csvData[:, :, 5].max() #TMIN
    #csvData[:, :, 6] = csvData[:, :, 6] / csvData[:, :, 6].max() #DL

    return csvData, dataMax


def load_data(csv_data, seq_len, cellsNum, valpercent):
    sequence_length = seq_len + 1
    result = []
    for i in range(cellsNum):
        #print(i)
        tempData = csv_data[i, :, :]
        for index in range(len(tempData) - seq_len):
            # if tempData[index + seq_len, 0] <= 1 and tempData[index, 0] > 0:
            #     result.append(tempData[index: index + sequence_length])
            result.append(tempData[index: index + sequence_length])
    del tempData
    del csv_data
    print('1')
    result = np.array(result)
    print('1')
    x_train = result[:, :-1, 1:]
    print('1')
    y_train = result[:, -1, 0]
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
    lr_power = 0.6
    lr = lr_base * ((1 - float(epoch) / epochs) ** lr_power)
    print('learning rate:', lr)
    return float(lr)

def main():
    csv_dir = '/media/ubuntu/a1f55ab9-3d8e-4d5e-bb92-609fddf79c72/LDZ/LSTM/NDVI/Croplands/Data/2009/train_500'
    cellsNum = 500
    csv_data, dataMax = get_data(csv_dir, cellsNum)
    dataMax = np.array(dataMax)
    np.savetxt('/media/ubuntu/a1f55ab9-3d8e-4d5e-bb92-609fddf79c72/LDZ/LSTM/NDVI/Croplands/Logs/dataMax_2009.txt', dataMax)
    print("finish reading")
    window = 90
    x_train, y_train = load_data(csv_data, window, cellsNum, 0)
    del csv_data
    print("x_train", x_train.shape)
    print("y_train", y_train.shape)
    # print("x_val", x_val.shape)
    # print("y_val", y_val.shape)
    model = build_model([5, window])
    filepath = '/media/ubuntu/a1f55ab9-3d8e-4d5e-bb92-609fddf79c72/LDZ/LSTM/NDVI/Croplands/Models/2009_DL/{epoch:03d}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, period=1)
    reduce_lr = LearningRateScheduler(learning_rate_schedule)
    #model.summary()
    #model.load_weights("/media/ubuntu/a1f55ab9-3d8e-4d5e-bb92-609fddf79c72/LDZ/LSTM/Evergreen/Models_1000/050.hdf5")
    history = model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=100,
        validation_split=0.1,
        verbose=1,
        #callbacks=[reduce_lr, checkpoint, TensorBoard(log_dir='./logs')],
        callbacks=[checkpoint, TensorBoard(log_dir='./logs')],
        shuffle=True
        )
    print(dataMax)
    model.save('/media/ubuntu/a1f55ab9-3d8e-4d5e-bb92-609fddf79c72/LDZ/LSTM/NDVI/Croplands/Models/2009_DL/NDVI_500.h5')
    lossy = history.history['loss']
    np_lossy = np.array(lossy).reshape((1, len(lossy)))
    np.savetxt('/media/ubuntu/a1f55ab9-3d8e-4d5e-bb92-609fddf79c72/LDZ/LSTM/NDVI/Croplands/Logs/loss_2009_DL.txt', np_lossy)



if __name__ == '__main__':
    main()
