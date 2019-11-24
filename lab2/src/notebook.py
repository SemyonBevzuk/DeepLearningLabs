import os
import sys

from keras.engine.saving import load_model

sys.path.append('../../src/')

import datahandler as dh
import modelhandler as mh
import plthandler as ph
import reporthandler as rh


def run_serial_experiment(data, series_parameters, all_configurations):
    save_folder_model = os.path.join('..', 'models')
    save_folder_log = os.path.join('..', 'log')
    save_folder_graphs = os.path.join('..', 'img')

    for current_configuration in all_configurations:
        print("\n\t !Model: {}".format(current_configuration))
        series_parameters['label'] = current_configuration['label']
        series_parameters['layers'] = current_configuration['layers']
        mh.fit_and_save_model(data, series_parameters, save_folder_model, save_folder_log, save_folder_graphs)


def run_serial_data2():
    filename = 'data2.pickle'
    path = os.path.join('..', '..', 'data', filename)
    data = dh.get_vector_data(path)
    dh.print_data_information(data)
    series_parameters = {'batch_size': 128,
                         'num_epochs': 10}

    all_configurations = [
        dict(label='FCNN_1_relu', layers=[
            {'units': 128, 'activation': 'relu'}
        ]),
        dict(label='FCNN_2_relu', layers=[
            {'units': 256, 'activation': 'relu'}
        ]),
        dict(label='FCNN_3_relu', layers=[
            {'units': 512, 'activation': 'relu'}
        ]),
        dict(label='FCNN_4_relu', layers=[
            {'units': 1024, 'activation': 'relu'}
        ]),
        dict(label='FCNN_5_relu', layers=[
            {'units': 1024, 'activation': 'relu'},
            {'units': 512, 'activation': 'relu'}
        ]),
        dict(label='FCNN_6_relu', layers=[
            {'units': 1024, 'activation': 'relu'},
            {'units': 512, 'activation': 'relu'},
            {'units': 256, 'activation': 'relu'}
        ]),

        dict(label='FCNN_1_elu', layers=[
            {'units': 128, 'activation': 'elu'}
        ]),
        dict(label='FCNN_2_elu', layers=[
            {'units': 256, 'activation': 'elu'}
        ]),
        dict(label='FCNN_3_elu', layers=[
            {'units': 512, 'activation': 'elu'}
        ]),
        dict(label='FCNN_4_elu', layers=[
            {'units': 1024, 'activation': 'elu'}
        ]),
        dict(label='FCNN_5_elu', layers=[
            {'units': 1024, 'activation': 'elu'},
            {'units': 512, 'activation': 'elu'}
        ]),
        dict(label='FCNN_6_elu', layers=[
            {'units': 1024, 'activation': 'elu'},
            {'units': 512, 'activation': 'elu'},
            {'units': 256, 'activation': 'elu'}
        ]),

        dict(label='FCNN_1_sigmoid', layers=[
            {'units': 128, 'activation': 'sigmoid'}
        ]),
        dict(label='FCNN_2_sigmoid', layers=[
            {'units': 256, 'activation': 'sigmoid'}
        ]),
        dict(label='FCNN_3_sigmoid', layers=[
            {'units': 512, 'activation': 'sigmoid'}
        ]),
        dict(label='FCNN_4_sigmoid', layers=[
            {'units': 1024, 'activation': 'sigmoid'}
        ]),
        dict(label='FCNN_5_sigmoid', layers=[
            {'units': 1024, 'activation': 'sigmoid'},
            {'units': 512, 'activation': 'sigmoid'}
        ]),
        dict(label='FCNN_6_sigmoid', layers=[
            {'units': 1024, 'activation': 'sigmoid'},
            {'units': 512, 'activation': 'sigmoid'},
            {'units': 256, 'activation': 'sigmoid'}
        ])
    ]

    run_serial_experiment(data, series_parameters, all_configurations)


def main():
    run_serial_data2()

    report_path = os.path.join('..', 'readme.md')
    save_folder_log = os.path.join('..', 'log')
    save_folder_img = os.path.join('..', 'img')
    save_folder_model = os.path.join('..', 'models')

    rh.add_result_table_to_report(report_path, save_folder_log)
    rh.add_graph_table_to_report(report_path, save_folder_img)
    rh.add_graph_model_table_to_report(report_path, save_folder_img)

    # mh.show_all_models(save_folder_model, save_folder_log)


if __name__ == "__main__":
    main()
