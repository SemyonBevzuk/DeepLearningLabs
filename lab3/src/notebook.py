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
        ]),

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
