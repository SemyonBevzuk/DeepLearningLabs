from math import sqrt, ceil
from keras.utils.np_utils import to_categorical

import numpy as np
import pickle
import matplotlib.pyplot as plt
import re


def read_raw_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    data['y_train'] = to_categorical(data['y_train'], num_classes=43)
    data['y_validation'] = to_categorical(data['y_validation'], num_classes=43)
    data['y_test'] = to_categorical(data['y_test'], num_classes=43)
    data['x_train'] = data['x_train'].transpose(0, 2, 3, 1)
    data['x_validation'] = data['x_validation'].transpose(0, 2, 3, 1)
    data['x_test'] = data['x_test'].transpose(0, 2, 3, 1)

    return data


def get_data(path):
    data = read_raw_data(path)
    data['x_train'] = data['x_train'].reshape(
        data['x_train'].shape[0],
        data['x_train'].shape[1] * data['x_train'].shape[2] * data['x_train'].shape[3])
    data['x_test'] = data['x_test'].reshape(
        data['x_test'].shape[0],
        data['x_test'].shape[1] * data['x_test'].shape[2] * data['x_test'].shape[3])
    return data


def print_data_information(data):
    for i, j in data.items():
        if i == 'labels':
            print(i + ':', len(j))
        else:
            print(i + ':', j.shape)


def convert_to_grid(data, number_examples, type):
    if type == 'gray':
        data = data[:number_examples, :, :, 0]
        N, H, W = data.shape
    elif type == 'rgb':
        data = data[:number_examples, :, :, :]
        N, H, W, C = data.shape

    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + 1 * (grid_size - 1)
    grid_width = W * grid_size + 1 * (grid_size - 1)
    if type == 'gray':
        grid = np.zeros((grid_height, grid_width)) + 255
    elif type == 'rgb':
        grid = np.zeros((grid_height, grid_width, C)) + 255
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = data[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = 255.0 * (img - low) / (high - low)
                next_idx += 1
            x0 += W + 1
            x1 += W + 1
        y0 += H + 1
        y1 += H + 1

    return grid


def get_data_type(filename):
    result = int(re.findall(r'\d', filename)[0])
    if 0 <= result <= 3:
        return 'rgb'
    elif 4 <= result <= 8:
        return 'gray'
    else:
        return 'error'


def save_img_examples(data, data_type, number_examples, filename):
    fig = plt.figure()
    grid = convert_to_grid(data, number_examples, data_type)
    plt.imshow(grid.astype('uint8'), cmap='gray')
    plt.axis('off')
    plt.gcf().set_size_inches(15, 15)
    plt.title('Examples of data', fontsize=18)
    # plt.show()
    fig.savefig(filename)
    plt.close()
