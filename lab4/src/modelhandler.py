import re
import sys
import json
import os

sys.path.append('../../src/')
import plthandler as ph

from datetime import datetime
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras import Sequential


def save_model(model, save_folder):
    filename = model.name + '.h5'
    path = os.path.join(save_folder, filename)
    model.save(path)


def fit_model(data, params):
    x_train = data['x_train']
    y_train = data['y_train']

    model = Sequential()
    model.add(Conv2D(filters=params['layers'][0]['filters'], kernel_size=params['layers'][0]['kernel_size'],
                     padding=params['layers'][0]['padding'], activation=params['layers'][0]['activation'],
                     kernel_initializer='he_normal', name=params['layers'][0]['activation']+'_1',
                     input_shape=(data['x_train'].shape[1], data['x_train'].shape[2], data['x_train'].shape[3])))
    for id, layer in enumerate(params['layers'][1:]):
        if layer['name'] == 'Conv2D':
            model.add(Conv2D(filters=layer['filters'], kernel_size=layer['kernel_size'],
                             padding=layer['padding'], activation=layer['activation'],
                             name=layer['activation']+'_'+str(id+2), kernel_initializer='he_normal'))
        if layer['name'] == 'MaxPool2D':
            model.add(MaxPool2D(pool_size=layer['pool_size'], name=str(id+2)))
        if layer['name'] == 'Flatten':
            model.add(Flatten(name=str(id+2)))
        if layer['name'] == 'Dense':
            model.add(Dense(units=layer['units'], activation=layer['activation'],
                            name=layer['activation']+'_'+str(id+2), kernel_initializer='he_normal'))
        if layer['name'] == 'Dropout':
            model.add(Dropout(rate=layer['rate'], name=str(id+2)))
    model.add(Dense(units=y_train.shape[1], activation='softmax', kernel_initializer='he_normal'))
    model.name = params['label']

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    log = model.fit(x_train, y_train, batch_size=params['batch_size'], epochs=params['num_epochs'],
                    validation_data=(data['x_test'], data['y_test']), shuffle=True, verbose=2)
    return (model, log)


def fit_and_save_model(data, params, save_folder_model, save_folder_log, save_folder_graphs):
    statistics = {}

    time_start = datetime.now()
    model, log = fit_model(data, params)
    delta_time = datetime.now() - time_start
    statistics['Time_train'] = delta_time.total_seconds()

    score_train = model.evaluate(data['x_train'], data['y_train'], verbose=0)
    statistics['Train_loss'] = score_train[0]
    statistics['Train_accuracy'] = score_train[1]

    score_test = model.evaluate(data['x_test'], data['y_test'], verbose=0)
    statistics['Test_loss'] = score_test[0]
    statistics['Test_accuracy'] = score_test[1]

    model_info = {'Parameters': params, 'Statistics': statistics}

    save_model(model, save_folder_model)

    model_name = model.name
    filename = model_name + '.json'
    with open(os.path.join(save_folder_log, filename), 'w', encoding='utf-8') as file:
        json.dump(model_info, file)

    ph.save_accuracy_graph(log, model_name, save_folder_graphs)
    ph.save_loss_graph(log, model_name, save_folder_graphs)
    ph.save_model_graph(model, model_name, save_folder_graphs)
    return model_name


def print_model_info(model, save_folder_log):
    filename = model.name + '.json'
    with open(os.path.join(save_folder_log, filename), 'r') as read_file:
        model_info = json.load(read_file)
    print(' Parameters:')
    print('batch_size = {}'.format(model_info['Parameters']['batch_size']))
    print('num_epochs = {}'.format(model_info['Parameters']['num_epochs']))
    print('layers = {}'.format(model_info['Parameters']['layers']))
    print(' Statistics:')
    print('Time_train = {}'.format(model_info['Statistics']['Time_train']))
    print('Train_loss = {}'.format(model_info['Statistics']['Train_loss']))
    print('Train_accuracy = {}'.format(model_info['Statistics']['Train_accuracy']))
    print('Test_loss = {}'.format(model_info['Statistics']['Test_loss']))
    print('Test_accuracy = {}'.format(model_info['Statistics']['Test_accuracy']))


def show_all_models(models_folder, save_folder_log):
    for file in os.listdir(models_folder):
        if file.endswith('.h5'):
            model = load_model(os.path.join(models_folder, file))
            print('\n\tModel: {}'.format(file))
            print('Number of layers: {}'.format(len(model.layers) - 1))
            print_model_info(model, save_folder_log)


def show_models(models_folder, models, save_folder_log):
    for file in os.listdir(models_folder):
        if file.endswith('.h5') and file in models:
            model = load_model(os.path.join(models_folder, file))
            print('\n\tModel: {}'.format(file))
            print('Number of layers: {}'.format(len(model.layers) - 1))
            print_model_info(model, save_folder_log)
