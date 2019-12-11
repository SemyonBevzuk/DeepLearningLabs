import re
import sys
import json
import os

sys.path.append('../../src/')
import plthandler as ph

from datetime import datetime
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout, Reshape, UpSampling2D
from keras import Sequential, regularizers


def save_model(model, save_folder):
    filename = model.name + '.h5'
    path = os.path.join(save_folder, filename)
    model.save(path)

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


def create_autoencoder_FCNN(regularization_parameter=0):
    postfix = ''
    if regularization_parameter != 0:
        postfix = '_' + str(regularization_parameter)

    input_img = Input(shape=(32, 32, 3))
    flat_img = Flatten()(input_img)
    encoded = Dense(128, activation='sigmoid', kernel_initializer='he_normal',
                  activity_regularizer=regularizers.l2(regularization_parameter))(flat_img)

    input_encoded = Input(shape=(128,))
    flat_decoded = Dense(32 * 32 * 3, activation='sigmoid', kernel_initializer='he_normal')(input_encoded)
    decoded = Reshape((32, 32, 3))(flat_decoded)

    encoder = Model(input_img, encoded, name='FCNN_1_sigmoid_encoder' + postfix)
    decoder = Model(input_encoded, decoded, name='FCNN_1_sigmoid_decoder' + postfix)
    autoencoder = Model(input_img, decoder(encoder(input_img)), name='FCNN_1_sigmoid_autoencoder' + postfix)
    return encoder, decoder, autoencoder


def create_deep_autoencoder_FCNN(regularization_parameter=0):
    postfix = ''
    if regularization_parameter != 0:
        postfix = '_' + str(regularization_parameter)

    input_img = Input(shape=(32, 32, 3))
    flat_img = Flatten()(input_img)
    layer = Dense(1024, activation='sigmoid', kernel_initializer='he_normal')(flat_img)
    layer = Dense(512, activation='sigmoid', kernel_initializer='he_normal')(layer)
    encoded = Dense(256, activation='sigmoid', kernel_initializer='he_normal',
                    activity_regularizer=regularizers.l2(regularization_parameter))(layer)

    input_encoded = Input(shape=(256,))
    layer = Dense(512, activation='sigmoid', kernel_initializer='he_normal')(input_encoded)
    layer = Dense(1024, activation='sigmoid', kernel_initializer='he_normal')(layer)
    flat_decoded = Dense(32 * 32 * 3, activation='sigmoid', kernel_initializer='he_normal')(layer)
    decoded = Reshape((32, 32, 3))(flat_decoded)

    encoder = Model(input_img, encoded, name='FCNN_6_sigmoid_encoder'+postfix)
    decoder = Model(input_encoded, decoded, name='FCNN_6_sigmoid_decoder'+postfix)
    autoencoder = Model(input_img, decoder(encoder(input_img)), name='FCNN_6_sigmoid_autoencoder'+postfix)
    return encoder, decoder, autoencoder


def fit_and_save_autoencoder(model_autoencoder, params, data, save_folder_log, save_folder_img, save_folder_model):
    statistics = {}
    time_start = datetime.now()
    log = model_autoencoder.fit(data['x_train'], data['x_train'],
                            epochs=params['num_epochs'],
                            batch_size=params['batch_size'],
                            shuffle=True,
                            validation_data=(data['x_test'], data['x_test']),
                            verbose=2)
    delta_time = datetime.now() - time_start
    statistics['Time_train'] = delta_time.total_seconds()

    score_train = model_autoencoder.evaluate(data['x_train'], data['x_train'], verbose=0)
    statistics['Train_loss'] = score_train[0]
    statistics['Train_accuracy'] = score_train[1]

    score_test = model_autoencoder.evaluate(data['x_test'], data['x_test'], verbose=0)
    statistics['Test_loss'] = score_test[0]
    statistics['Test_accuracy'] = score_test[1]
    model_info = {'Parameters': params, 'Statistics': statistics}

    save_model(model_autoencoder, save_folder_model)
    model_name = model_autoencoder.name
    filename = model_name + '.json'
    with open(os.path.join(save_folder_log, filename), 'w', encoding='utf-8') as file:
        json.dump(model_info, file)
    ph.save_accuracy_graph(log, model_name, save_folder_img)
    ph.save_loss_graph(log, model_name, save_folder_img)
    ph.save_model_graph(model_autoencoder, model_name, save_folder_img)

    return model_autoencoder


def fit_and_save_pretrained_FCNN(model_encoder, model_name, params, data, save_folder_log, save_folder_img, save_folder_model):
    postfix = ''
    if params['regularization_parameter'] != 0:
        postfix = '_' + str(params['regularization_parameter'])

    statistics = {}
    model_pretrained = Sequential()
    model_pretrained.name = model_name + '_pretraining' + postfix
    for layer in model_encoder.layers:
        model_pretrained.add(layer)
    model_pretrained.add(Dense(units=43, activation='softmax', kernel_initializer='he_normal'))
    model_pretrained.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    time_start = datetime.now()
    log = model_pretrained.fit(data['x_train'], data['y_train'],
                              epochs=params['num_epochs'],
                              batch_size=params['batch_size'],
                              shuffle=True,
                              validation_data=(data['x_test'], data['y_test']),
                              verbose=2)
    delta_time = datetime.now() - time_start
    statistics['Time_train'] = delta_time.total_seconds()

    score_train = model_pretrained.evaluate(data['x_train'], data['y_train'], verbose=0)
    statistics['Train_loss'] = score_train[0]
    statistics['Train_accuracy'] = score_train[1]

    score_test = model_pretrained.evaluate(data['x_test'], data['y_test'], verbose=0)
    statistics['Test_loss'] = score_test[0]
    statistics['Test_accuracy'] = score_test[1]
    model_info = {'Parameters': params, 'Statistics': statistics}

    save_model(model_pretrained, save_folder_model)
    model_name = model_pretrained.name
    filename = model_name + '.json'
    with open(os.path.join(save_folder_log, filename), 'w', encoding='utf-8') as file:
        json.dump(model_info, file)
    ph.save_accuracy_graph(log, model_name, save_folder_img)
    ph.save_loss_graph(log, model_name, save_folder_img)
    ph.save_model_graph(model_pretrained, model_name, save_folder_img)


def save_statistic_from_best_model(model_name, model_folder, data, save_folder_log):
    file = model_name
    old_model_folder = os.path.join(model_folder, 'models')
    old_model_log_folder = os.path.join(model_folder, 'log')
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
    model_info = {'Parameters': {'batch_size': 128,
                                 'num_epochs': 10},
                  'Statistics': statistics}

    model_name = old_model.name
    filename = model_name + '.json'
    with open(os.path.join(save_folder_log, filename), 'w', encoding='utf-8') as file:
        json.dump(model_info, file)


def create_autoencoder_CNN():
    input_img = Input(shape=(32, 32, 3))
    layer = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(input_img)
    encoded = MaxPool2D(pool_size=2)(layer)

    input_encoded = Input(shape=(16, 16, 32))
    layer = UpSampling2D((2, 2))(input_encoded)
    decoded = Conv2D(filters=3, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(layer)

    encoder = Model(input_img, encoded, name='CNN_1_relu_encoder')
    decoder = Model(input_encoded, decoded, name='CNN_1_relu_decoder')
    autoencoder = Model(input_img, decoder(encoder(input_img)), name='CNN_1_relu_autoencoder')
    return encoder, decoder, autoencoder


def create_deep_autoencoder_CNN():
    input_img = Input(shape=(32, 32, 3))
    layer = Conv2D(filters=6, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(input_img)
    layer = Conv2D(filters=12, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(layer)
    layer = Conv2D(filters=24, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(layer)
    layer = MaxPool2D(pool_size=2)(layer)
    layer = Conv2D(filters=48, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(layer)
    encoded = MaxPool2D(pool_size=2)(layer)

    input_encoded = Input(shape=(8, 8, 48))
    layer = UpSampling2D((2, 2))(input_encoded)
    layer = Conv2D(filters=24, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(layer)
    layer = UpSampling2D((2, 2))(layer)
    layer = Conv2D(filters=12, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(layer)
    layer = Conv2D(filters=6, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(layer)
    decoded = Conv2D(filters=3, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(layer)

    encoder = Model(input_img, encoded, name='CNN_10_relu_encoder')
    decoder = Model(input_encoded, decoded, name='CNN_10_relu_decoder')
    autoencoder = Model(input_img, decoder(encoder(input_img)), name='CNN_10_relu_autoencoder')
    return encoder, decoder, autoencoder


def fit_and_save_pretrained_CNN(model_encoder, model_name, params, data, save_folder_log, save_folder_img, save_folder_model):
    statistics = {}
    model_pretrained = Sequential()
    model_pretrained.name = model_name + '_pretraining'
    for layer in model_encoder.layers:
        model_pretrained.add(layer)
    model_pretrained.add(Flatten())
    model_pretrained.add(Dense(units=1024, activation='relu', kernel_initializer='he_normal')) # 256
    model_pretrained.add(Dense(units=43, activation='softmax', kernel_initializer='he_normal'))
    model_pretrained.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    time_start = datetime.now()
    log = model_pretrained.fit(data['x_train'], data['y_train'],
                              epochs=params['num_epochs'],
                              batch_size=params['batch_size'],
                              shuffle=True,
                              validation_data=(data['x_test'], data['y_test']),
                              verbose=2)
    delta_time = datetime.now() - time_start
    statistics['Time_train'] = delta_time.total_seconds()

    score_train = model_pretrained.evaluate(data['x_train'], data['y_train'], verbose=0)
    statistics['Train_loss'] = score_train[0]
    statistics['Train_accuracy'] = score_train[1]

    score_test = model_pretrained.evaluate(data['x_test'], data['y_test'], verbose=0)
    statistics['Test_loss'] = score_test[0]
    statistics['Test_accuracy'] = score_test[1]
    model_info = {'Parameters': params, 'Statistics': statistics}

    save_model(model_pretrained, save_folder_model)
    model_name = model_pretrained.name
    filename = model_name + '.json'
    with open(os.path.join(save_folder_log, filename), 'w', encoding='utf-8') as file:
        json.dump(model_info, file)
    ph.save_accuracy_graph(log, model_name, save_folder_img)
    ph.save_loss_graph(log, model_name, save_folder_img)
    ph.save_model_graph(model_pretrained, model_name, save_folder_img)



def create_stack_autoencoders_FCNN():
    input_img = Input(shape=(32, 32, 3))
    flat_img = Flatten()(input_img)

    encoded_1 = Dense(1024, activation='sigmoid', kernel_initializer='he_normal')(flat_img)
    input_encoded_1 = Input(shape=(1024,))
    flat_decoded = Dense(32 * 32 * 3, activation='sigmoid', kernel_initializer='he_normal')(input_encoded_1)
    decoded_1 = Reshape((32, 32, 3))(flat_decoded)
    encoder_1 = Model(input_img, encoded_1, name='FCNN_6_sigmoid_encoder_1_stack')
    decoder_1 = Model(input_encoded_1, decoded_1, name='FCNN_6_sigmoid_decoder_1_stack')
    autoencoder_1 = Model(input_img, decoder_1(encoder_1(input_img)), name='FCNN_6_sigmoid_autoencoder_1_stack')

    input_2 = Input(shape=(1024,))
    encoded_2 = Dense(512, activation='sigmoid', kernel_initializer='he_normal')(input_2)
    input_encoded_2 = Input(shape=(512,))
    decoded_2 = Dense(1024, activation='sigmoid', kernel_initializer='he_normal')(input_encoded_2)
    encoder_2 = Model(input_2, encoded_2, name='FCNN_6_sigmoid_encoder_2_stack')
    decoder_2 = Model(input_encoded_2, decoded_2, name='FCNN_6_sigmoid_decoder_2_stack')
    autoencoder_2 = Model(input_2, decoder_2(encoder_2(input_2)), name='FCNN_6_sigmoid_autoencoder_2_stack')

    input_3 = Input(shape=(512,))
    encoded_3 = Dense(256, activation='sigmoid', kernel_initializer='he_normal')(input_3)
    input_encoded_3 = Input(shape=(256,))
    decoded_3 = Dense(512, activation='sigmoid', kernel_initializer='he_normal')(input_encoded_3)
    encoder_3 = Model(input_3, encoded_3, name='FCNN_6_sigmoid_encoder_3_stack')
    decoder_3 = Model(input_encoded_3, decoded_3, name='FCNN_6_sigmoid_decoder_3_stack')
    autoencoder_3 = Model(input_3, decoder_3(encoder_3(input_3)), name='FCNN_6_sigmoid_autoencoder_3_stack')

    return [encoder_1, encoder_2, encoder_3], [autoencoder_1, autoencoder_2, autoencoder_3]


def fit_and_save_stack_autoencoders(autoencoders, encoders, params, data, save_folder_log, save_folder_img, save_folder_model):
    fit_and_save_autoencoder(autoencoders[0], params, data, save_folder_log, save_folder_img, save_folder_model)

    data_temp = {}
    data_temp['x_train'] = encoders[0].predict(data['x_train'])
    data_temp['x_test'] = encoders[0].predict(data['x_test'])
    fit_and_save_autoencoder(autoencoders[1], params, data_temp, save_folder_log, save_folder_img, save_folder_model)

    data_temp['x_train'] = encoders[1].predict(data_temp['x_train'])
    data_temp['x_test'] = encoders[1].predict(data_temp['x_test'])
    fit_and_save_autoencoder(autoencoders[2], params, data_temp, save_folder_log, save_folder_img, save_folder_model)


def fit_and_save_pretrained_FCNN_from_stack_encoders(encoders, model_name, params, data, save_folder_log, save_folder_img, save_folder_model):
    statistics = {}
    model_pretrained = Sequential()
    model_pretrained.name = model_name + '_pretraining_with_stack'
    for layer in encoders:
        model_pretrained.add(layer)
    model_pretrained.add(Dense(units=43, activation='softmax', kernel_initializer='he_normal'))
    model_pretrained.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    time_start = datetime.now()
    log = model_pretrained.fit(data['x_train'], data['y_train'],
                               epochs=params['num_epochs'],
                               batch_size=params['batch_size'],
                               shuffle=True,
                               validation_data=(data['x_test'], data['y_test']),
                               verbose=2)
    delta_time = datetime.now() - time_start
    statistics['Time_train'] = delta_time.total_seconds()

    score_train = model_pretrained.evaluate(data['x_train'], data['y_train'], verbose=0)
    statistics['Train_loss'] = score_train[0]
    statistics['Train_accuracy'] = score_train[1]

    score_test = model_pretrained.evaluate(data['x_test'], data['y_test'], verbose=0)
    statistics['Test_loss'] = score_test[0]
    statistics['Test_accuracy'] = score_test[1]
    model_info = {'Parameters': params, 'Statistics': statistics}

    save_model(model_pretrained, save_folder_model)
    model_name = model_pretrained.name
    filename = model_name + '.json'
    with open(os.path.join(save_folder_log, filename), 'w', encoding='utf-8') as file:
        json.dump(model_info, file)
    ph.save_accuracy_graph(log, model_name, save_folder_img)
    ph.save_loss_graph(log, model_name, save_folder_img)
    ph.save_model_graph(model_pretrained, model_name, save_folder_img)
