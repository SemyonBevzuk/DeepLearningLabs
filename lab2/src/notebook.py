from datetime import datetime
from keras import Input, Model, optimizers
from keras.layers import Dense
import json

import datahandler as dh
import modelhandler as mh
import plthandler as ph

import os


def run_serial_experiment(data, params, all_hidden_layer_sizes):
    save_folder_model = os.path.join('..', 'models')
    save_folder_log = os.path.join('..', 'log')
    save_folder_graphs = os.path.join('..', 'img')

    for hidden_layer_sizes in all_hidden_layer_sizes:
        print("\n\t !Model: {}".format(hidden_layer_sizes))
        params['hidden_layer_sizes'] = hidden_layer_sizes
        mh.fit_and_save_model(data, params, save_folder_model, save_folder_log, save_folder_graphs)

    models = []
    for elem in all_hidden_layer_sizes:
        model_name = 'FCNN'
        for l in elem:
            model_name += '_' + str(l)
        model_name += '.h5'
        models.append(model_name)
    mh.show_models(save_folder_model, models, save_folder_log)


def main():
    filename = 'data2.pickle'
    path = os.path.join('..', '..', 'data', filename)
    data = dh.get_data(path)
    dh.print_data_information(data)
    params = {'batch_size': 128,
              'num_epochs': 10}
    all_hidden_layer_sizes = [
        [768],
        [1536],
        [2304],
        [1536, 768],
        [2304, 768],
        [1536, 768, 384, 192, 96]
    ]
    run_serial_experiment(data, params, all_hidden_layer_sizes)


if __name__ == "__main__":
    main()
