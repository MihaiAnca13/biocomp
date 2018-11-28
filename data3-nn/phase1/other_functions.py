from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import random


def read_file(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            row = line.replace("\n", "").split(" ")
            for i in range(len(row)):
                row[i] = float(row[i])
            data.append(row)
    return data


def create_model(input_dim, layer_size):
    # create model
    model = Sequential()
    model.add(Dense(layer_size, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def prepare_data(data):
    x = [item[:7] for item in data]
    y = [item[7] for item in data]

    data_size = len(data)
    data = []

    for index in range(data_size):
        data.append([x[index], y[index]])

    random.shuffle(data)

    x_train = [item[0] for item in data]
    y_train = [item[1] for item in data]

    return np.array(x_train, dtype=np.float16), np.array(y_train, dtype=np.float16)
