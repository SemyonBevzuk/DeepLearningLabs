from keras.utils.np_utils import to_categorical
import pickle
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
