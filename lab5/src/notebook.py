import json
import os
import sys

#print(os.getcwd())
os.chdir('/content/drive/My Drive/Colab Notebooks/DeepLearning_Folder/DeepLearningLabs/lab5/src/')
#print(os.getcwd())
sys.path.append('../../src/')
sys.path.append('/content/drive/My Drive/Colab Notebooks/DeepLearning_Folder/DeepLearningLabs/src/')
#print(sys.path)

import datahandler as dh
import modelhandler as mh
import plthandler as ph
import reporthandler as rh



def run_experiment_base_InceptionResNetV2():
    save_folder_log = os.path.join('..', 'log')
    save_folder_img = os.path.join('..', 'img')
    save_folder_model = os.path.join('..', 'models')

    filename = 'data2.pickle'
    path = os.path.join('..', '..', 'data', filename)
    data = dh.get_matrix_data(path)
    # dh.print_data_information(data)

    params = {'batch_size': 128,
              'num_epochs': 10,
              'label': 'base_NASNetMobile_512'}

    mh.fit_and_save_base_NASNetMobile(data, params, save_folder_model, save_folder_log, save_folder_img)


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

    print('\tInceptionResNetV2 stock\n')
    run_experiment_base_InceptionResNetV2()

    rh.add_result_table_to_report(report_path, save_folder_log)
    #rh.add_graph_table_to_report(report_path, save_folder_img)
    #rh.add_graph_model_table_to_report(report_path, save_folder_img)

    #mh.show_all_models(save_folder_model, save_folder_log)


if __name__ == "__main__":
    main()
