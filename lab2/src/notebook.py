import os
import sys
sys.path.append('../../src/')

import datahandler as dh
import modelhandler as mh
import plthandler as ph
import reporthandler as rh




def run_serial_experiment(data, params, all_hidden_layer_sizes):
    save_folder_model = os.path.join('..', 'models')
    save_folder_log = os.path.join('..', 'log')
    save_folder_graphs = os.path.join('..', 'img')

    for hidden_layer_sizes in all_hidden_layer_sizes:
        print("\n\t !Model: {}".format(hidden_layer_sizes))
        params['hidden_layer_sizes'] = hidden_layer_sizes
        mh.fit_and_save_model(data, params, save_folder_model, save_folder_log, save_folder_graphs)

    models = []
    for elem in all_hidden_layer_sizes:
        model_name = 'FCNN'
        for l in elem:
            model_name += '_' + str(l)
        model_name += '.h5'
        models.append(model_name)
    mh.show_models(save_folder_model, models, save_folder_log)


def run_serial_data2():
    filename = 'data2.pickle'
    path = os.path.join('..', '..', 'data', filename)
    data = dh.get_vector_data(path)
    dh.print_data_information(data)
    params = {'batch_size': 128,
              'num_epochs': 15}
    all_hidden_layer_sizes = [
        [128],
        [200],
        [256],
        [300],
        [512],
        [768],
        [1024],
        [1536],
        [2048],
        [2304],
        [768, 384],
        [1024, 512],
        [1536, 768],
        [2304, 768],
        [768, 384, 192],
        [768, 384, 96],
        [2048, 1024, 512, 256],
        [1536, 768, 384, 192, 96]
    ]

    run_serial_experiment(data, params, all_hidden_layer_sizes)


def main():
    #run_serial_data2()

    report_path = os.path.join('..', 'readme.md')
    save_folder_log = os.path.join('..', 'log')
    save_folder_img = os.path.join('..', 'img')
    save_folder_model = os.path.join('..', 'models')

    rh.add_result_table_to_report(report_path, save_folder_log)
    rh.add_graph_table_to_report(report_path, save_folder_img)

    # mh.show_all_models(save_folder_model, save_folder_log)


if __name__ == "__main__":
    main()
