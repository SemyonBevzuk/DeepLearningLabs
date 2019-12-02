import os
import re
import sys

from keras import Sequential
from keras.engine.saving import load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.utils import plot_model

sys.path.append('../../src/')

import datahandler as dh
import modelhandler as mh
import plthandler as ph
import reporthandler as rh


def run_serial_experiment(data, series_parameters, all_configurations):
    save_folder_model = os.path.join('..', 'models')
    save_folder_log = os.path.join('..', 'log')
    save_folder_graphs = os.path.join('..', 'img')

    for current_configuration in all_configurations:
        print("\n\t !Model: {}".format(current_configuration))
        series_parameters['label'] = current_configuration['label']
        series_parameters['layers'] = current_configuration['layers']
        mh.fit_and_save_model(data, series_parameters, save_folder_model, save_folder_log, save_folder_graphs)


def run_serial_data2():
    filename = 'data2.pickle'
    path = os.path.join('..', '..', 'data', filename)
    data = dh.get_matrix_data(path)
    dh.print_data_information(data)
    params = {'batch_size': 128,
              'num_epochs': 10}
    all_configurations = [
        dict(label='CNN_1_relu', layers=[
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 256, 'activation': 'relu'}
        ]),
        dict(label='CNN_2_relu_dropout', layers=[
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Dropout', 'rate': 0.2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 256, 'activation': 'relu'}
        ]),
        dict(label='CNN_3_relu', layers=[
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 64, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 256, 'activation': 'relu'}
        ]),
        dict(label='CNN_4_relu_dropout', layers=[
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Dropout', 'rate': 0.2},
            {'name': 'Conv2D', 'filters': 64, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Dropout', 'rate': 0.5},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 256, 'activation': 'relu'}
        ]),
        dict(label='CNN_5_relu', layers=[
            {'name': 'Conv2D', 'filters': 16, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 64, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 512, 'activation': 'relu'}
        ]),
        dict(label='CNN_6_relu_dropout', layers=[
            {'name': 'Conv2D', 'filters': 16, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Dropout', 'rate': 0.2},
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Dropout', 'rate': 0.2},
            {'name': 'Conv2D', 'filters': 64, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Dropout', 'rate': 0.5},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 512, 'activation': 'relu'}
        ]),

        dict(label='CNN_1_elu', layers=[
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 256, 'activation': 'elu'}
        ]),
        dict(label='CNN_2_elu_dropout', layers=[
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Dropout', 'rate': 0.2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 256, 'activation': 'elu'}
        ]),
        dict(label='CNN_3_elu', layers=[
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 64, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 256, 'activation': 'elu'}
        ]),
        dict(label='CNN_4_elu_dropout', layers=[
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Dropout', 'rate': 0.2},
            {'name': 'Conv2D', 'filters': 64, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Dropout', 'rate': 0.5},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 256, 'activation': 'elu'}
        ]),
        dict(label='CNN_5_elu', layers=[
            {'name': 'Conv2D', 'filters': 16, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 64, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 512, 'activation': 'elu'}
        ]),
        dict(label='CNN_6_elu_dropout', layers=[
            {'name': 'Conv2D', 'filters': 16, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Dropout', 'rate': 0.2},
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Dropout', 'rate': 0.2},
            {'name': 'Conv2D', 'filters': 64, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Dropout', 'rate': 0.5},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 512, 'activation': 'elu'}
        ]),

        dict(label='CNN_1_sigmoid', layers=[
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 256, 'activation': 'elu'}
        ]),
        dict(label='CNN_2_sigmoid_dropout', layers=[
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Dropout', 'rate': 0.2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 256, 'activation': 'sigmoid'}
        ]),
        dict(label='CNN_3_sigmoid', layers=[
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 64, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 256, 'activation': 'sigmoid'}
        ]),
        dict(label='CNN_4_sigmoid_dropout', layers=[
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Dropout', 'rate': 0.2},
            {'name': 'Conv2D', 'filters': 64, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Dropout', 'rate': 0.5},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 256, 'activation': 'sigmoid'}
        ]),
        dict(label='CNN_5_sigmoid', layers=[
            {'name': 'Conv2D', 'filters': 16, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 64, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 512, 'activation': 'sigmoid'}
        ]),
        dict(label='CNN_6_sigmoid_dropout', layers=[
            {'name': 'Conv2D', 'filters': 16, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Dropout', 'rate': 0.2},
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Dropout', 'rate': 0.2},
            {'name': 'Conv2D', 'filters': 64, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Dropout', 'rate': 0.5},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 512, 'activation': 'sigmoid'}
        ]),

        dict(label='CNN_9_relu', layers=[
            {'name': 'Conv2D', 'filters': 6, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'Conv2D', 'filters': 12, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'Conv2D', 'filters': 24, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 48, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 256, 'activation': 'relu'}
        ]),

        dict(label='CNN_9_elu', layers=[
            {'name': 'Conv2D', 'filters': 6, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'Conv2D', 'filters': 12, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'Conv2D', 'filters': 24, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 48, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 256, 'activation': 'elu'}
        ]),

        dict(label='CNN_9_sigmoid', layers=[
            {'name': 'Conv2D', 'filters': 6, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'Conv2D', 'filters': 12, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'Conv2D', 'filters': 24, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 48, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 256, 'activation': 'sigmoid'}
        ]),

        dict(label='CNN_10_relu', layers=[
            {'name': 'Conv2D', 'filters': 6, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'Conv2D', 'filters': 12, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'Conv2D', 'filters': 24, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 48, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 1024, 'activation': 'relu'}
        ]),

        dict(label='CNN_10_elu', layers=[
            {'name': 'Conv2D', 'filters': 6, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'Conv2D', 'filters': 12, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'Conv2D', 'filters': 24, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 48, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 1024, 'activation': 'elu'}
        ]),

        dict(label='CNN_10_sigmoid', layers=[
            {'name': 'Conv2D', 'filters': 6, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'Conv2D', 'filters': 12, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'Conv2D', 'filters': 24, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 48, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 1024, 'activation': 'sigmoid'}
        ]),

        dict(label='CNN_11_relu', layers=[
            {'name': 'Conv2D', 'filters': 6, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'Conv2D', 'filters': 12, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'Conv2D', 'filters': 24, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'Conv2D', 'filters': 48, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 96, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 1024, 'activation': 'relu'}
        ]),

        dict(label='CNN_11_elu', layers=[
            {'name': 'Conv2D', 'filters': 6, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'Conv2D', 'filters': 12, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'Conv2D', 'filters': 24, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'Conv2D', 'filters': 48, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 96, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 1024, 'activation': 'elu'}
        ]),

        dict(label='CNN_11_sigmoid', layers=[
            {'name': 'Conv2D', 'filters': 6, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'Conv2D', 'filters': 12, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'Conv2D', 'filters': 24, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'Conv2D', 'filters': 48, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 96, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 1024, 'activation': 'sigmoid'}
        ]),

        dict(label='CNN_12_relu', layers=[
            {'name': 'Conv2D', 'filters': 128, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 256, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 512, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 1024, 'activation': 'relu'}
        ]),

        dict(label='CNN_12_elu', layers=[
            {'name': 'Conv2D', 'filters': 128, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 256, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 512, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 1024, 'activation': 'elu'}
        ]),

        dict(label='CNN_12_sigmoid', layers=[
            {'name': 'Conv2D', 'filters': 128, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 256, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 512, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 1024, 'activation': 'sigmoid'}
        ]),

        dict(label='CNN_12_relu', layers=[
            {'name': 'Conv2D', 'filters': 24, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 48, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 1024, 'activation': 'relu'}
        ]),

        dict(label='CNN_12_elu', layers=[
            {'name': 'Conv2D', 'filters': 24, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 48, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 1024, 'activation': 'elu'}
        ]),

        dict(label='CNN_12_sigmoid', layers=[
            {'name': 'Conv2D', 'filters': 24, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 48, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 1024, 'activation': 'sigmoid'}
        ]),

        dict(label='CNN_13_relu', layers=[
            {'name': 'Conv2D', 'filters': 6, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'Conv2D', 'filters': 12, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'Conv2D', 'filters': 24, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 48, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 1024, 'activation': 'relu'},
            {'name': 'Dense', 'units': 256, 'activation': 'relu'}
        ]),

        dict(label='CNN_13_elu', layers=[
            {'name': 'Conv2D', 'filters': 6, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'Conv2D', 'filters': 12, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'Conv2D', 'filters': 24, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 48, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 1024, 'activation': 'elu'},
            {'name': 'Dense', 'units': 256, 'activation': 'elu'}
        ]),

        dict(label='CNN_13_sigmoid', layers=[
            {'name': 'Conv2D', 'filters': 6, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'Conv2D', 'filters': 12, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'Conv2D', 'filters': 24, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 48, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 1024, 'activation': 'sigmoid'},
            {'name': 'Dense', 'units': 256, 'activation': 'sigmoid'}
        ]),

        dict(label='CNN_14_relu', layers=[
            {'name': 'Conv2D', 'filters': 6, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'Conv2D', 'filters': 12, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'Conv2D', 'filters': 24, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 48, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 1024, 'activation': 'relu'},
            {'name': 'Dense', 'units': 512, 'activation': 'relu'},
            {'name': 'Dense', 'units': 256, 'activation': 'relu'}
        ]),

        dict(label='CNN_14_elu', layers=[
            {'name': 'Conv2D', 'filters': 6, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'Conv2D', 'filters': 12, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'Conv2D', 'filters': 24, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 48, 'kernel_size': 3, 'padding': 'same', 'activation': 'elu'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 1024, 'activation': 'elu'},
            {'name': 'Dense', 'units': 512, 'activation': 'elu'},
            {'name': 'Dense', 'units': 256, 'activation': 'elu'}
        ]),

        dict(label='CNN_14_sigmoid', layers=[
            {'name': 'Conv2D', 'filters': 6, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'Conv2D', 'filters': 12, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'Conv2D', 'filters': 24, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 48, 'kernel_size': 3, 'padding': 'same', 'activation': 'sigmoid'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 1024, 'activation': 'sigmoid'},
            {'name': 'Dense', 'units': 512, 'activation': 'sigmoid'},
            {'name': 'Dense', 'units': 256, 'activation': 'sigmoid'}
        ])
    ]
    run_serial_experiment(data, params, all_configurations)


def main():
    run_serial_data2()

    report_path = os.path.join('..', 'readme.md')
    save_folder_log = os.path.join('..', 'log')
    save_folder_img = os.path.join('..', 'img')
    save_folder_model = os.path.join('..', 'models')

    rh.add_result_table_to_report(report_path, save_folder_log)
    rh.add_graph_table_to_report(report_path, save_folder_img)
    rh.add_graph_model_table_to_report(report_path, save_folder_img)

    #mh.show_all_models(save_folder_model, save_folder_log)


if __name__ == "__main__":
    main()
