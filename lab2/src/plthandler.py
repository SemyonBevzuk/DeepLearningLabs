from math import sqrt, ceil
import matplotlib.pyplot as plt
import numpy as np
import os

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


def save_accuracy_graph(log, model_name, save_folder_graphs):
    plt.rcParams['figure.figsize'] = (15.0, 6.0)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['font.family'] = 'Times New Roman'

    fig = plt.figure()
    plt.plot(log.history['accuracy'], '-o', linewidth=3.0)
    plt.plot(log.history['val_accuracy'], '-o', linewidth=3.0)
    plt.title(model_name, fontsize=22)
    plt.legend(['train', 'test'], loc='upper left', fontsize='xx-large')
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.tick_params(labelsize=18)

    # plt.show()
    filename = model_name + '_accuracy' + '.png'
    path = os.path.join(save_folder_graphs, filename)
    fig.savefig(path)
    plt.close()


def save_loss_graph(log, model_name, save_folder_graphs):
    plt.rcParams['figure.figsize'] = (15.0, 6.0)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['font.family'] = 'Times New Roman'

    fig = plt.figure()
    plt.plot(log.history['loss'], '-o', linewidth=3.0)
    plt.plot(log.history['val_loss'], '-o', linewidth=3.0)
    plt.title(model_name, fontsize=22)
    plt.legend(['train', 'test'], loc='upper left', fontsize='xx-large')
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.tick_params(labelsize=18)

    # plt.show()
    filename = model_name + '_loss' + '.png'
    path = os.path.join(save_folder_graphs, filename)
    fig.savefig(path)
    plt.close()
