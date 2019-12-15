from keras.utils.np_utils import to_categorical
import numpy as np
import pickle
import re
import os
from skimage.transform import resize

def read_raw_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    data['y_train'] = to_categorical(data['y_train'], num_classes=43)
    data['y_validation'] = to_categorical(data['y_validation'], num_classes=43)
    data['y_test'] = to_categorical(data['y_test'], num_classes=43)
    data['x_train'] = data['x_train'].transpose(0, 2, 3, 1)
    data['x_validation'] = data['x_validation'].transpose(0, 2, 3, 1)
    data['x_test'] = data['x_test'].transpose(0, 2, 3, 1)

    #data['x_train'] = data['x_train'][:100]
    #data['y_train'] = data['y_train'][:100]
    #data['x_test'] = data['x_test'][:100]
    #data['y_test'] = data['y_test'][:100]
    return data


def get_vector_data(path):
    data = read_raw_data(path)
    data['x_train'] = data['x_train'].reshape(
        data['x_train'].shape[0],
        data['x_train'].shape[1] * data['x_train'].shape[2] * data['x_train'].shape[3])
    data['x_test'] = data['x_test'].reshape(
        data['x_test'].shape[0],
        data['x_test'].shape[1] * data['x_test'].shape[2] * data['x_test'].shape[3])
    return data


def get_matrix_data(path):
    data = read_raw_data(path)
    return data


def get_matrix_data_with_large_label(path, label_size):
    data = read_raw_data(path)
    data['y_train'] = np.concatenate((data['y_train'],
                                      np.zeros((data['y_train'].shape[0], label_size - data['y_train'].shape[1]))), axis=1)
    data['y_validation'] = np.concatenate((data['y_validation'],
                                           np.zeros((data['y_validation'].shape[0], label_size - data['y_validation'].shape[1]))), axis=1)
    data['y_test'] = np.concatenate((data['y_test'],
                                     np.zeros((data['y_test'].shape[0], label_size - data['y_test'].shape[1]))), axis=1)
    return data


def get_matrix_zoom_data_with_large_label(path, zoom, label_size):
    data = read_raw_data(path)

    imgs = []
    for img in data['x_train']:
        imgs.append(resize(img, (299, 299)))
    data['x_train'] = np.array(imgs)

    data['x_train'] = np.kron(data['x_train'], np.ones((zoom, zoom)))
    data['x_validation'] = np.kron(data['x_train'], np.ones((zoom, zoom)))
    data['x_test'] = np.kron(data['x_train'], np.ones((zoom, zoom)))

    data['y_train'] = np.concatenate((data['y_train'],
                                      np.zeros((data['y_train'].shape[0], label_size - data['y_train'].shape[1]))), axis=1)
    data['y_validation'] = np.concatenate((data['y_validation'],
                                           np.zeros((data['y_validation'].shape[0], label_size - data['y_validation'].shape[1]))), axis=1)
    data['y_test'] = np.concatenate((data['y_test'],
                                     np.zeros((data['y_test'].shape[0], label_size - data['y_test'].shape[1]))), axis=1)
    return data


def print_data_information(data):
    for i, j in data.items():
        if i == 'labels':
            print(i + ':', len(j))
        else:
            print(i + ':', j.shape)


def get_data_type(filename):
    result = int(re.findall(r'\d', filename)[0])
    if 0 <= result <= 3:
        return 'rgb'
    elif 4 <= result <= 8:
        return 'gray'
    else:
        return 'error'
