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


def main():
    report_path = os.path.join('..', 'readme.md')
    save_folder_log = os.path.join('..', 'log')
    save_folder_img = os.path.join('..', 'img')
    save_folder_model = os.path.join('..', 'models')

    filename = 'data2.pickle'
    path = os.path.join('..', '..', 'data', filename)
    data = dh.get_matrix_data(path)
    dh.print_data_information(data)

    d_encoder, d_decoder, d_autoencoder = mh.create_dense_ae()
    d_autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    params = {'batch_size': 128,
                  'num_epochs': 10}
    statistics = {}

    time_start = datetime.now()
    log = d_autoencoder.fit( data['x_train'],  data['x_train'],
                      epochs=params['num_epochs'],
                      batch_size=params['batch_size'],
                      shuffle=True,
                      validation_data=(data['x_test'], data['x_test']),
                      verbose=2)
    delta_time = datetime.now() - time_start
    statistics['Time_train'] = delta_time.total_seconds()

    score_train = d_autoencoder.evaluate(data['x_train'], data['x_train'], verbose=0)
    statistics['Train_loss'] = score_train[0]
    statistics['Train_accuracy'] = score_train[1]

    score_test = d_autoencoder.evaluate(data['x_test'], data['x_test'], verbose=0)
    statistics['Test_loss'] = score_test[0]
    statistics['Test_accuracy'] = score_test[1]
    model_info = {'Parameters': params, 'Statistics': statistics}

    mh.save_model(d_autoencoder, save_folder_model)
    model_name = d_autoencoder.name
    filename = model_name + '.json'
    with open(os.path.join(save_folder_log, filename), 'w', encoding='utf-8') as file:
        json.dump(model_info, file)
    ph.save_accuracy_graph(log, model_name, save_folder_img)
    ph.save_loss_graph(log, model_name, save_folder_img)
    ph.save_model_graph(d_autoencoder, model_name, save_folder_img)


    statistics = {}
    model_d_encoder = Sequential()
    for layer in d_encoder.layers:
        model_d_encoder.add(layer)
    model_d_encoder.add(Dense(units=43, activation='softmax', kernel_initializer='he_normal'))
    model_d_encoder.name = 'FCNN_6_sigmoid_encoder'
    model_d_encoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    time_start = datetime.now()
    log = model_d_encoder.fit(data['x_train'], data['y_train'],
                        epochs=params['num_epochs'],
                        batch_size=params['batch_size'],
                        shuffle=True,
                        validation_data=(data['x_test'], data['y_test']),
                        verbose=2)
    delta_time = datetime.now() - time_start
    statistics['Time_train'] = delta_time.total_seconds()

    score_train = model_d_encoder.evaluate(data['x_train'], data['y_train'], verbose=0)
    statistics['Train_loss'] = score_train[0]
    statistics['Train_accuracy'] = score_train[1]

    score_test = model_d_encoder.evaluate(data['x_test'], data['y_test'], verbose=0)
    statistics['Test_loss'] = score_test[0]
    statistics['Test_accuracy'] = score_test[1]
    model_info = {'Parameters': params, 'Statistics': statistics}

    mh.save_model(model_d_encoder, save_folder_model)
    model_name = model_d_encoder.name
    filename = model_name + '.json'
    with open(os.path.join(save_folder_log, filename), 'w', encoding='utf-8') as file:
        json.dump(model_info, file)
    ph.save_accuracy_graph(log, model_name, save_folder_img)
    ph.save_loss_graph(log, model_name, save_folder_img)
    ph.save_model_graph(model_d_encoder, model_name, save_folder_img)

    data = dh.get_vector_data(path)
    file = 'FCNN_6_sigmoid'
    old_model_folder = os.path.join('..', '..', 'lab2', 'models')
    old_model_log_folder = os.path.join('..', '..', 'lab2', 'log')
    old_model = load_model(os.path.join(old_model_folder, file + '.h5'))
    with open(os.path.join(old_model_log_folder, file + '.json'), 'r') as read_file:
        model_info = json.load(read_file)
    statistics = {}
    statistics['Time_train'] = model_info['Statistics']['Time_train']
    score_train = old_model.evaluate(data['x_train'], data['y_train'], verbose=0)
    statistics['Train_loss'] = score_train[0]
    statistics['Train_accuracy'] = score_train[1]

    score_test = old_model.evaluate(data['x_test'], data['y_test'], verbose=0)
    statistics['Test_loss'] = score_test[0]
    statistics['Test_accuracy'] = score_test[1]
    model_info = {'Parameters': params, 'Statistics': statistics}

    model_name = old_model.name
    filename = model_name + '.json'
    with open(os.path.join(save_folder_log, filename), 'w', encoding='utf-8') as file:
        json.dump(model_info, file)

    rh.add_result_table_to_report(report_path, save_folder_log)
    #rh.add_graph_table_to_report(report_path, save_folder_img)
    #rh.add_graph_model_table_to_report(report_path, save_folder_img)

    # mh.show_all_models(save_folder_model, save_folder_log)


if __name__ == "__main__":
    main()
