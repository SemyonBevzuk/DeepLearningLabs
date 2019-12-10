import json
import os
import sys

from keras import Sequential
from keras.engine.saving import load_model
from datetime import datetime

from keras.layers import Dense

sys.path.append('../../src/')

import datahandler as dh
import modelhandler as mh
import plthandler as ph
import reporthandler as rh



def run_experiment_autoencoder_FCNN(regularization_parameter=0):
    save_folder_log = os.path.join('..', 'log')
    save_folder_img = os.path.join('..', 'img')
    save_folder_model = os.path.join('..', 'models')

    filename = 'data2.pickle'
    path = os.path.join('..', '..', 'data', filename)
    data = dh.get_matrix_data(path)
    # dh.print_data_information(data)

    params = {'batch_size': 128,
              'num_epochs': 10,
              'regularization_parameter': regularization_parameter}

    encoder, decoder, autoencoder = mh.create_autoencoder_FCNN(params['regularization_parameter'])
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    mh.fit_and_save_autoencoder(autoencoder, params, data, save_folder_log, save_folder_img, save_folder_model)

    best_FCNN_name = 'FCNN_1_sigmoid'
    best_FCNN_folder = os.path.join('..', '..', 'lab2')

    mh.fit_and_save_pretrained_FCNN(encoder, best_FCNN_name, params, data, save_folder_log, save_folder_img, save_folder_model)

    data = dh.get_vector_data(path)
    mh.save_statistic_from_best_model(best_FCNN_name, best_FCNN_folder, data, save_folder_log)


def run_experiment_autoencoder_CNN():
    save_folder_log = os.path.join('..', 'log')
    save_folder_img = os.path.join('..', 'img')
    save_folder_model = os.path.join('..', 'models')

    filename = 'data2.pickle'
    path = os.path.join('..', '..', 'data', filename)
    data = dh.get_matrix_data(path)
    #dh.print_data_information(data)

    params = {'batch_size': 128,
              'num_epochs': 10}

    encoder, decoder, autoencoder = mh.create_autoencoder_CNN()
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    mh.fit_and_save_autoencoder(autoencoder, params, data, save_folder_log, save_folder_img, save_folder_model)

    best_CNN_name = 'CNN_1_relu'
    best_CNN_folder = os.path.join('..', '..', 'lab3')

    mh.fit_and_save_pretrained_CNN(encoder, best_CNN_name, params, data, save_folder_log, save_folder_img, save_folder_model)

    mh.save_statistic_from_best_model(best_CNN_name, best_CNN_folder, data, save_folder_log)


def run_experiment_deep_autoencoder_FCNN(regularization_parameter=0):
    save_folder_log = os.path.join('..', 'log')
    save_folder_img = os.path.join('..', 'img')
    save_folder_model = os.path.join('..', 'models')

    filename = 'data2.pickle'
    path = os.path.join('..', '..', 'data', filename)
    data = dh.get_matrix_data(path)
    #dh.print_data_information(data)

    params = {'batch_size': 128,
              'num_epochs': 10,
              'regularization_parameter': regularization_parameter}

    encoder, decoder, autoencoder = mh.create_deep_autoencoder_FCNN(params['regularization_parameter'])
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    mh.fit_and_save_autoencoder(autoencoder, params, data, save_folder_log, save_folder_img, save_folder_model)

    best_FCNN_name = 'FCNN_6_sigmoid'
    best_FCNN_folder = os.path.join('..', '..', 'lab2')

    mh.fit_and_save_pretrained_FCNN(encoder, best_FCNN_name, params, data, save_folder_log, save_folder_img, save_folder_model)

    data = dh.get_vector_data(path)
    mh.save_statistic_from_best_model(best_FCNN_name, best_FCNN_folder, data, save_folder_log)


def run_experiment_deep_autoencoder_CNN(regularization_parameter=0):
    save_folder_log = os.path.join('..', 'log')
    save_folder_img = os.path.join('..', 'img')
    save_folder_model = os.path.join('..', 'models')

    filename = 'data2.pickle'
    path = os.path.join('..', '..', 'data', filename)
    data = dh.get_matrix_data(path)
    #dh.print_data_information(data)

    params = {'batch_size': 128,
              'num_epochs': 10,
              'regularization_parameter': regularization_parameter}

    encoder, decoder, autoencoder = mh.create_deep_autoencoder_CNN()
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    mh.fit_and_save_autoencoder(autoencoder, params, data, save_folder_log, save_folder_img, save_folder_model)

    best_CNN_name = 'CNN_10_relu'
    best_CNN_folder = os.path.join('..', '..', 'lab3')

    mh.fit_and_save_pretrained_CNN(encoder, best_CNN_name, params, data, save_folder_log, save_folder_img, save_folder_model)

    mh.save_statistic_from_best_model(best_CNN_name, best_CNN_folder, data, save_folder_log)


def run_experiment_stack_autoencoders_FCNN():
    save_folder_log = os.path.join('..', 'log')
    save_folder_img = os.path.join('..', 'img')
    save_folder_model = os.path.join('..', 'models')

    filename = 'data2.pickle'
    path = os.path.join('..', '..', 'data', filename)
    data = dh.get_matrix_data(path)
    # dh.print_data_information(data)

    params = {'batch_size': 128,
              'num_epochs': 10}

    encoders, autoencoders = mh.create_stack_autoencoders_FCNN()
    for autoencoder in autoencoders:
        autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    mh.fit_and_save_stack_autoencoders(autoencoders, encoders, params, data, save_folder_log, save_folder_img, save_folder_model)

    best_FCNN_name = 'FCNN_6_sigmoid'
    best_FCNN_folder = os.path.join('..', '..', 'lab2')

    mh.fit_and_save_pretrained_FCNN_from_stack_encoders(encoders, best_FCNN_name, params, data, save_folder_log, save_folder_img, save_folder_model)

    data = dh.get_vector_data(path)
    mh.save_statistic_from_best_model(best_FCNN_name, best_FCNN_folder, data, save_folder_log)

def main():
    report_path = os.path.join('..', 'readme.md')

    save_folder_log = os.path.join('..', 'log')
    save_folder_img = os.path.join('..', 'img')
    save_folder_model = os.path.join('..', 'models')

    #filename = 'data2.pickle'
    #path = os.path.join('..', '..', 'data', filename)
    #data = dh.get_matrix_data(path)
    #data['x_train'] = data['x_train'][:100]
    #data['y_train'] = data['y_train'][:100]
    #data['x_test'] = data['x_test'][:100]
    #data['y_test'] = data['y_test'][:100]

    #print('\tFCNN test 1 layer\n')
    #run_experiment_autoencoder_FCNN()
    #print('\tCNN test 1 layer\n')
    #run_experiment_autoencoder_CNN()


    #print('\tFCNN\n')
    #run_experiment_deep_autoencoder_FCNN()
    #print('\tFCNN with regularization_parameter\n')
    #run_experiment_deep_autoencoder_FCNN(regularization_parameter=0.0001)
    #print('\tCNN\n')
    #run_experiment_deep_autoencoder_CNN()
    print('\tFCNN stack autoencoder\n')
    run_experiment_stack_autoencoders_FCNN()

    rh.add_result_table_to_report(report_path, save_folder_log)
    #rh.add_graph_table_to_report(report_path, save_folder_img)
    #rh.add_graph_model_table_to_report(report_path, save_folder_img)

    #mh.show_all_models(save_folder_model, save_folder_log)


if __name__ == "__main__":
    main()
