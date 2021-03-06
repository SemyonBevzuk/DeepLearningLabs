import json
import os
import sys

sys.path.append('../../src/')

import datahandler as dh
import modelhandler as mh
import plthandler as ph
import reporthandler as rh


NASNetMobile_OUTPUT_SIZE = 1000


def run_experiment_base_NASNetMobile():
    save_folder_log = os.path.join('..', 'log')
    save_folder_img = os.path.join('..', 'img')
    save_folder_model = os.path.join('..', 'models')

    filename = 'data2.pickle'
    path = os.path.join('..', '..', 'data', filename)
    data = dh.get_matrix_data_with_large_label(path, NASNetMobile_OUTPUT_SIZE)
    # dh.print_data_information(data)

    params = {'batch_size': 128,
              'num_epochs': 10,
              'label': 'base_NASNetMobile'}

    mh.fit_and_save_base_NASNetMobile(data, params, save_folder_model, save_folder_log, save_folder_img)


def run_experiment_NASNetMobile_with_fully_connected_layers():
    save_folder_log = os.path.join('..', 'log')
    save_folder_img = os.path.join('..', 'img')
    save_folder_model = os.path.join('..', 'models')

    filename = 'data2.pickle'
    path = os.path.join('..', '..', 'data', filename)
    data = dh.get_matrix_data(path)
    # dh.print_data_information(data)

    params = {'batch_size': 128,
              'num_epochs': 10,
              'label': 'base_NASNetMobile_with_classifier'}

    mh.fit_and_save_NASNetMobile_with_fully_connected_layers(data, params, save_folder_model, save_folder_log, save_folder_img)


def run_experiment_fit_NASNetMobile():
    save_folder_log = os.path.join('..', 'log')
    save_folder_img = os.path.join('..', 'img')
    save_folder_model = os.path.join('..', 'models')

    filename = 'data2.pickle'
    path = os.path.join('..', '..', 'data', filename)
    data = dh.get_matrix_data_with_large_label(path, NASNetMobile_OUTPUT_SIZE)
    # dh.print_data_information(data)

    params = {'batch_size': 128,
              'num_epochs': 10,
              'label': 'NASNetMobile'}

    mh.fit_and_save_NASNetMobile(data, params, save_folder_model, save_folder_log, save_folder_img)


def run_experiment_fit_NASNetMobile_with_classifier():
    save_folder_log = os.path.join('..', 'log')
    save_folder_img = os.path.join('..', 'img')
    save_folder_model = os.path.join('..', 'models')

    filename = 'data2.pickle'
    path = os.path.join('..', '..', 'data', filename)
    data = dh.get_matrix_data(path)
    # dh.print_data_information(data)

    params = {'batch_size': 128,
              'num_epochs': 10,
              'label': 'NASNetMobile_with_classifier'}

    mh.fit_and_save_NASNetMobile_with_classifier(data, params, save_folder_model, save_folder_log, save_folder_img)


def run_experiment_base_NASNetMobile_zoom_data():
    save_folder_log = os.path.join('..', 'log')
    save_folder_img = os.path.join('..', 'img')
    save_folder_model = os.path.join('..', 'models')

    filename = 'data2.pickle'
    path = os.path.join('..', '..', 'data', filename)
    data = dh.get_matrix_data_with_large_label(path, NASNetMobile_OUTPUT_SIZE)
    # dh.print_data_information(data)

    params = {'batch_size': 128,
              'num_epochs': 10,
              'label': 'base_NASNetMobile_zoom_data',
              'img_size': 256}

    mh.fit_and_save_base_NASNetMobile_zoom_data(data, params, save_folder_model, save_folder_log, save_folder_img)


def run_experiment_NASNetMobile_with_fully_connected_layers_zoom_data():
    save_folder_log = os.path.join('..', 'log')
    save_folder_img = os.path.join('..', 'img')
    save_folder_model = os.path.join('..', 'models')

    filename = 'data2.pickle'
    path = os.path.join('..', '..', 'data', filename)
    data = dh.get_matrix_data(path)
    # dh.print_data_information(data)

    params = {'batch_size': 128,
              'num_epochs': 10,
              'label': 'base_NASNetMobile_with_classifier_zoom_data',
              'img_size': 256}

    mh.fit_and_save_NASNetMobile_with_fully_connected_layers_zoom_data(data, params, save_folder_model, save_folder_log,
                                                             save_folder_img)


def run_experiment_fit_NASNetMobile_zoom_data():
    save_folder_log = os.path.join('..', 'log')
    save_folder_img = os.path.join('..', 'img')
    save_folder_model = os.path.join('..', 'models')

    filename = 'data2.pickle'
    path = os.path.join('..', '..', 'data', filename)
    data = dh.get_matrix_data_with_large_label(path, NASNetMobile_OUTPUT_SIZE)
    # dh.print_data_information(data)

    params = {'batch_size': 128,
              'num_epochs': 10,
              'label': 'NASNetMobile_zoom_data',
              'img_size': 256}

    mh.fit_and_save_NASNetMobile_zoom_data(data, params, save_folder_model, save_folder_log, save_folder_img)


def run_experiment_fit_NASNetMobile_with_classifier_zoom_data():
    save_folder_log = os.path.join('..', 'log')
    save_folder_img = os.path.join('..', 'img')
    save_folder_model = os.path.join('..', 'models')

    filename = 'data2.pickle'
    path = os.path.join('..', '..', 'data', filename)
    data = dh.get_matrix_data(path)
    # dh.print_data_information(data)

    params = {'batch_size': 128,
              'num_epochs': 10,
              'label': 'NASNetMobile_with_classifier_zoom_data',
              'img_size': 256}

    mh.fit_and_save_NASNetMobile_with_classifier_zoom_data(data, params, save_folder_model, save_folder_log, save_folder_img)


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


    print('\n\tBase NASNetMobile\n')
    run_experiment_base_NASNetMobile()

    print('\n\tNASNetMobile_512_256_128_64\n')
    run_experiment_NASNetMobile_with_fully_connected_layers()

    print('\n\tNASNetMobile\n')
    run_experiment_fit_NASNetMobile()

    print('\n\tNASNetMobile with my classifier\n')
    run_experiment_fit_NASNetMobile_with_classifier()




    print('\n\tBase NASNetMobile zoom data \n')
    run_experiment_base_NASNetMobile_zoom_data()

    print('\n\tBase NASNetMobile with my classifier zoom data \n')
    run_experiment_NASNetMobile_with_fully_connected_layers_zoom_data()

    #print('\n\tNASNetMobile zoom data \n')
    #run_experiment_fit_NASNetMobile_zoom_data()

    #print('\n\tNASNetMobile with my classifier zoom data \n')
    #run_experiment_fit_NASNetMobile_with_classifier_zoom_data()


    rh.add_result_table_to_report(report_path, save_folder_log)
    #rh.add_graph_table_to_report(report_path, save_folder_img)
    #rh.add_graph_model_table_to_report(report_path, save_folder_img)

    #mh.show_all_models(save_folder_model, save_folder_log)


if __name__ == "__main__":
    main()

