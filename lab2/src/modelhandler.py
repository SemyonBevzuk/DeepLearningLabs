from datetime import datetime
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras import optimizers

import json
import os


def get_model_name(model):
    filename = 'FCNN'
    layers = model.layers[1:-1]
    for layer in layers:
        config = layer.get_config()
        filename += '_' + str(config['units'])
    return filename


def save_model(model, save_folder):
    filename = get_model_name(model) + '.h5'
    path = os.path.join(save_folder, filename)
    model.save(path)


def fit_model(data, params):
    x_train = data['x_train']
    y_train = data['y_train']

    inp = Input(shape=(x_train.shape[1],))  # Our input is a 1D vector of size 32*32*3
    hidden_layer_prev = inp
    for hidden_layer_size in params['hidden_layer_sizes']:
        hidden_layer_prev = Dense(hidden_layer_size, activation='relu')(hidden_layer_prev)
    out = Dense(y_train.shape[1], activation='softmax')(hidden_layer_prev)  # Output softmax layer
    model = Model(inputs=inp, outputs=out)  # To define a model, just specify its input and output layers
    adam = optimizers.Adam(learning_rate=params['lr'], beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=params['batch_size'], epochs=params['num_epochs'], verbose=2)

    return model


def fit_and_save_model(data, params, save_folder_model, save_folder_log):
    statistics = {}

    time_start = datetime.now()
    model = fit_model(data, params)
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

    filename = get_model_name(model) + '.json'
    with open(os.path.join(save_folder_log, filename), 'w', encoding='utf-8') as file:
        json.dump(model_info, file)


def print_model_info(model, save_folder_log):
    filename = get_model_name(model) + '.json'
    with open(os.path.join(save_folder_log, filename), 'r') as read_file:
        model_info = json.load(read_file)
    print(' Parameters:')
    print('batch_size = {}'.format(model_info['Parameters']['batch_size']))
    print('lr = {}'.format(model_info['Parameters']['lr']))
    print('num_epochs = {}'.format(model_info['Parameters']['num_epochs']))
    print('hidden_layer_sizes = {}'.format(model_info['Parameters']['hidden_layer_sizes']))
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
