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


def run_serial_experiment(data, params, all_layers):
    save_folder_model = os.path.join('..', 'models')
    save_folder_log = os.path.join('..', 'log')
    save_folder_graphs = os.path.join('..', 'img')

    models = []
    for layers in all_layers:
        print("\n\t !Model: {}".format(layers))
        params['layers'] = layers
        model_name = mh.fit_and_save_model(data, params, save_folder_model, save_folder_log, save_folder_graphs)
        model_name += '.h5'
        models.append(model_name)

    mh.show_models(save_folder_model, models, save_folder_log)


def run_serial_data2():
    filename = 'data2.pickle'
    path = os.path.join('..', '..', 'data', filename)
    data = dh.get_matrix_data(path)
    dh.print_data_information(data)
    params = {'batch_size': 128,
              'num_epochs': 5}
    all_layers = [
        [
            {'name': 'Conv2D', 'filters': 16, 'kernel_size': 3, 'padding': 'same'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Dropout', 'rate': 0.2},
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Dropout', 'rate': 0.5},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 300}
        ],
        [
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Dropout', 'rate': 0.2},
            {'name': 'Conv2D', 'filters': 64, 'kernel_size': 3, 'padding': 'same'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Dropout', 'rate': 0.5},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 300}
        ],
        [
            {'name': 'Conv2D', 'filters': 16, 'kernel_size': 3, 'padding': 'same'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Dropout', 'rate': 0.2},
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Dropout', 'rate': 0.4},
            {'name': 'Conv2D', 'filters': 64, 'kernel_size': 3, 'padding': 'same'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Dropout', 'rate': 0.4},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 300}
        ],
        [
            {'name': 'Conv2D', 'filters': 16, 'kernel_size': 3, 'padding': 'same'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Dropout', 'rate': 0.2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 300}
        ],
        [
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Dropout', 'rate': 0.2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 500}
        ],
        [
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Dropout', 'rate': 0.2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 300}
        ],
        [
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Dropout', 'rate': 0.2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 128}
        ],
        [
            {'name': 'Conv2D', 'filters': 16, 'kernel_size': 3, 'padding': 'same'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 300}
        ],
        [
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 64, 'kernel_size': 3, 'padding': 'same'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 300}
        ],
        [
            {'name': 'Conv2D', 'filters': 16, 'kernel_size': 3, 'padding': 'same'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Conv2D', 'filters': 64, 'kernel_size': 3, 'padding': 'same'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 300}
        ],
        [
            {'name': 'Conv2D', 'filters': 16, 'kernel_size': 3, 'padding': 'same'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 300}
        ],
        [
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 500}
        ],
        [
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 300}
        ],
        [
            {'name': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'padding': 'same'},
            {'name': 'MaxPool2D', 'pool_size': 2},
            {'name': 'Flatten'},
            {'name': 'Dense', 'units': 128}
        ]
    ]
    run_serial_experiment(data, params, all_layers)


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
