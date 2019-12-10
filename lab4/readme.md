# Лабораторная работа №4

# Описание директорий

## img
Содержит изображения для отчёта.
### graph_loss_accuracy
Графики зависимости accuracy и функции потерь на тренировочной и тестовой выборке в зависимости от эпохи.
### graph_model
Графы конфигураций сетей.

## log
Здесь лежат файлы .json со статистикой по разным конфигурациям.
Они содержат параметры сети (число слоёв, число нейронов, функции активации, параметр обучения, размер пачки, число эпох)
и статистику обучения (время, функцию потерь на тестовом и тренировочном наборе, точность на тестовом и тренировочном наборе)

## models
Здесь лежат файлы .h5 с конфигурацией сетей Keras для их последующей повторной загрузки.

## ../src
Общие скрипты для работы с данными и отчётом.
### datahandler.py
Содержит методы для чтения данные и конвертации их в векторную или матричную форму.
### plthandler.py
Содержит методы для отображения и сохранения графиков.
### reporthandler.py
Содержит методы для генерации таблиц в отчётах по логам экспериментов.

## src
Частные скрипты для работы с данными и фреймворком.
### modelhandler.py
Содержит методы для работы с сетями: запуск обучения, сбор статистики, сохранение и загрузка сетей.
### notebook.py
Является точкой входа. Блокнот для проведения экспериментов. Содержит метод для запуска серийного эксперимента с 
возможностью настройки конфигураций сетей.

## Структура сетей

[comment]: # (graph_model_table_start)

|         Model name          |                     Model graph                      |
| :-------------------------- | :--------------------------------------------------- |
| CNN_1_elu_model             | ![](img/graph_model/CNN_1_elu_model.png)             |
| CNN_1_relu_model            | ![](img/graph_model/CNN_1_relu_model.png)            |
| CNN_1_sigmoid_model         | ![](img/graph_model/CNN_1_sigmoid_model.png)         |
| CNN_2_elu_dropout_model     | ![](img/graph_model/CNN_2_elu_dropout_model.png)     |
| CNN_2_relu_dropout_model    | ![](img/graph_model/CNN_2_relu_dropout_model.png)    |
| CNN_2_sigmoid_dropout_model | ![](img/graph_model/CNN_2_sigmoid_dropout_model.png) |
| CNN_3_elu_model             | ![](img/graph_model/CNN_3_elu_model.png)             |
| CNN_3_relu_model            | ![](img/graph_model/CNN_3_relu_model.png)            |
| CNN_3_sigmoid_model         | ![](img/graph_model/CNN_3_sigmoid_model.png)         |
| CNN_4_elu_dropout_model     | ![](img/graph_model/CNN_4_elu_dropout_model.png)     |
| CNN_4_relu_dropout_model    | ![](img/graph_model/CNN_4_relu_dropout_model.png)    |
| CNN_4_sigmoid_dropout_model | ![](img/graph_model/CNN_4_sigmoid_dropout_model.png) |
| CNN_5_elu_model             | ![](img/graph_model/CNN_5_elu_model.png)             |
| CNN_5_relu_model            | ![](img/graph_model/CNN_5_relu_model.png)            |
| CNN_5_sigmoid_model         | ![](img/graph_model/CNN_5_sigmoid_model.png)         |
| CNN_6_elu_dropout_model     | ![](img/graph_model/CNN_6_elu_dropout_model.png)     |
| CNN_6_relu_dropout_model    | ![](img/graph_model/CNN_6_relu_dropout_model.png)    |
| CNN_6_sigmoid_dropout_model | ![](img/graph_model/CNN_6_sigmoid_dropout_model.png) |
| CNN_7_elu_model             | ![](img/graph_model/CNN_7_elu_model.png)             |
| CNN_7_relu_model            | ![](img/graph_model/CNN_7_relu_model.png)            |
| CNN_7_sigmoid_model         | ![](img/graph_model/CNN_7_sigmoid_model.png)         |
| CNN_8_elu_dropout_model     | ![](img/graph_model/CNN_8_elu_dropout_model.png)     |
| CNN_8_relu_dropout_model    | ![](img/graph_model/CNN_8_relu_dropout_model.png)    |
| CNN_8_sigmoid_dropout_model | ![](img/graph_model/CNN_8_sigmoid_dropout_model.png) |
| CNN_9_elu_model             | ![](img/graph_model/CNN_9_elu_model.png)             |
| CNN_9_relu_model            | ![](img/graph_model/CNN_9_relu_model.png)            |
| CNN_9_sigmoid_model         | ![](img/graph_model/CNN_9_sigmoid_model.png)         |
| CNN_10_elu_model            | ![](img/graph_model/CNN_10_elu_model.png)            |
| CNN_10_relu_model           | ![](img/graph_model/CNN_10_relu_model.png)           |
| CNN_10_sigmoid_model        | ![](img/graph_model/CNN_10_sigmoid_model.png)        |
| CNN_11_elu_model            | ![](img/graph_model/CNN_11_elu_model.png)            |
| CNN_11_relu_model           | ![](img/graph_model/CNN_11_relu_model.png)           |
| CNN_11_sigmoid_model        | ![](img/graph_model/CNN_11_sigmoid_model.png)        |
| CNN_12_elu_model            | ![](img/graph_model/CNN_12_elu_model.png)            |
| CNN_12_relu_model           | ![](img/graph_model/CNN_12_relu_model.png)           |
| CNN_12_sigmoid_model        | ![](img/graph_model/CNN_12_sigmoid_model.png)        |
| CNN_13_elu_model            | ![](img/graph_model/CNN_13_elu_model.png)            |
| CNN_13_relu_model           | ![](img/graph_model/CNN_13_relu_model.png)           |
| CNN_13_sigmoid_model        | ![](img/graph_model/CNN_13_sigmoid_model.png)        |
| CNN_14_elu_model            | ![](img/graph_model/CNN_14_elu_model.png)            |
| CNN_14_relu_model           | ![](img/graph_model/CNN_14_relu_model.png)           |
| CNN_14_sigmoid_model        | ![](img/graph_model/CNN_14_sigmoid_model.png)        |

[comment]: # (graph_model_table_end)

## Численные эксперименты
### Параметры

### Результаты
[comment]: # (result_table_start)

|       Model name       | Test accuracy | Test loss | Train accuracy | Train loss | Time_train (s) |
| :--------------------- | :-----------: | :-------: | :------------: | :--------: | :------------: |
| autoencoder            |    0.3343     |  0.0751   |     0.3353     |   0.0533   |    166.9747    |
| FCNN_6_sigmoid         |    0.8545     |  0.5885   |     0.9783     |   0.0777   |    689.7557    |
| FCNN_6_sigmoid_encoder |    0.7283     |  0.8965   |     0.8204     |   0.7031   |    73.4553     |

[comment]: # (result_table_end)

[comment]: # (graph_table_start)

|                            Accuracy                             |                            Loss                             |
| :-------------------------------------------------------------- | :---------------------------------------------------------- |
| ![](img/graph_loss_accuracy/CNN_1_elu_accuracy.png)             | ![](img/graph_loss_accuracy/CNN_1_elu_loss.png)             |
| ![](img/graph_loss_accuracy/CNN_1_relu_accuracy.png)            | ![](img/graph_loss_accuracy/CNN_1_relu_loss.png)            |
| ![](img/graph_loss_accuracy/CNN_1_sigmoid_accuracy.png)         | ![](img/graph_loss_accuracy/CNN_1_sigmoid_loss.png)         |
| ![](img/graph_loss_accuracy/CNN_2_elu_dropout_accuracy.png)     | ![](img/graph_loss_accuracy/CNN_2_elu_dropout_loss.png)     |
| ![](img/graph_loss_accuracy/CNN_2_relu_dropout_accuracy.png)    | ![](img/graph_loss_accuracy/CNN_2_relu_dropout_loss.png)    |
| ![](img/graph_loss_accuracy/CNN_2_sigmoid_dropout_accuracy.png) | ![](img/graph_loss_accuracy/CNN_2_sigmoid_dropout_loss.png) |
| ![](img/graph_loss_accuracy/CNN_3_elu_accuracy.png)             | ![](img/graph_loss_accuracy/CNN_3_elu_loss.png)             |
| ![](img/graph_loss_accuracy/CNN_3_relu_accuracy.png)            | ![](img/graph_loss_accuracy/CNN_3_relu_loss.png)            |
| ![](img/graph_loss_accuracy/CNN_3_sigmoid_accuracy.png)         | ![](img/graph_loss_accuracy/CNN_3_sigmoid_loss.png)         |
| ![](img/graph_loss_accuracy/CNN_4_elu_dropout_accuracy.png)     | ![](img/graph_loss_accuracy/CNN_4_elu_dropout_loss.png)     |
| ![](img/graph_loss_accuracy/CNN_4_relu_dropout_accuracy.png)    | ![](img/graph_loss_accuracy/CNN_4_relu_dropout_loss.png)    |
| ![](img/graph_loss_accuracy/CNN_4_sigmoid_dropout_accuracy.png) | ![](img/graph_loss_accuracy/CNN_4_sigmoid_dropout_loss.png) |
| ![](img/graph_loss_accuracy/CNN_5_elu_accuracy.png)             | ![](img/graph_loss_accuracy/CNN_5_elu_loss.png)             |
| ![](img/graph_loss_accuracy/CNN_5_relu_accuracy.png)            | ![](img/graph_loss_accuracy/CNN_5_relu_loss.png)            |
| ![](img/graph_loss_accuracy/CNN_5_sigmoid_accuracy.png)         | ![](img/graph_loss_accuracy/CNN_5_sigmoid_loss.png)         |
| ![](img/graph_loss_accuracy/CNN_6_elu_dropout_accuracy.png)     | ![](img/graph_loss_accuracy/CNN_6_elu_dropout_loss.png)     |
| ![](img/graph_loss_accuracy/CNN_6_relu_dropout_accuracy.png)    | ![](img/graph_loss_accuracy/CNN_6_relu_dropout_loss.png)    |
| ![](img/graph_loss_accuracy/CNN_6_sigmoid_dropout_accuracy.png) | ![](img/graph_loss_accuracy/CNN_6_sigmoid_dropout_loss.png) |
| ![](img/graph_loss_accuracy/CNN_7_elu_accuracy.png)             | ![](img/graph_loss_accuracy/CNN_7_elu_loss.png)             |
| ![](img/graph_loss_accuracy/CNN_7_relu_accuracy.png)            | ![](img/graph_loss_accuracy/CNN_7_relu_loss.png)            |
| ![](img/graph_loss_accuracy/CNN_7_sigmoid_accuracy.png)         | ![](img/graph_loss_accuracy/CNN_7_sigmoid_loss.png)         |
| ![](img/graph_loss_accuracy/CNN_8_elu_dropout_accuracy.png)     | ![](img/graph_loss_accuracy/CNN_8_elu_dropout_loss.png)     |
| ![](img/graph_loss_accuracy/CNN_8_relu_dropout_accuracy.png)    | ![](img/graph_loss_accuracy/CNN_8_relu_dropout_loss.png)    |
| ![](img/graph_loss_accuracy/CNN_8_sigmoid_dropout_accuracy.png) | ![](img/graph_loss_accuracy/CNN_8_sigmoid_dropout_loss.png) |
| ![](img/graph_loss_accuracy/CNN_9_elu_accuracy.png)             | ![](img/graph_loss_accuracy/CNN_9_elu_loss.png)             |
| ![](img/graph_loss_accuracy/CNN_9_relu_accuracy.png)            | ![](img/graph_loss_accuracy/CNN_9_relu_loss.png)            |
| ![](img/graph_loss_accuracy/CNN_9_sigmoid_accuracy.png)         | ![](img/graph_loss_accuracy/CNN_9_sigmoid_loss.png)         |
| ![](img/graph_loss_accuracy/CNN_10_elu_accuracy.png)            | ![](img/graph_loss_accuracy/CNN_10_elu_loss.png)            |
| ![](img/graph_loss_accuracy/CNN_10_relu_accuracy.png)           | ![](img/graph_loss_accuracy/CNN_10_relu_loss.png)           |
| ![](img/graph_loss_accuracy/CNN_10_sigmoid_accuracy.png)        | ![](img/graph_loss_accuracy/CNN_10_sigmoid_loss.png)        |
| ![](img/graph_loss_accuracy/CNN_11_elu_accuracy.png)            | ![](img/graph_loss_accuracy/CNN_11_elu_loss.png)            |
| ![](img/graph_loss_accuracy/CNN_11_relu_accuracy.png)           | ![](img/graph_loss_accuracy/CNN_11_relu_loss.png)           |
| ![](img/graph_loss_accuracy/CNN_11_sigmoid_accuracy.png)        | ![](img/graph_loss_accuracy/CNN_11_sigmoid_loss.png)        |
| ![](img/graph_loss_accuracy/CNN_12_elu_accuracy.png)            | ![](img/graph_loss_accuracy/CNN_12_elu_loss.png)            |
| ![](img/graph_loss_accuracy/CNN_12_relu_accuracy.png)           | ![](img/graph_loss_accuracy/CNN_12_relu_loss.png)           |
| ![](img/graph_loss_accuracy/CNN_12_sigmoid_accuracy.png)        | ![](img/graph_loss_accuracy/CNN_12_sigmoid_loss.png)        |
| ![](img/graph_loss_accuracy/CNN_13_elu_accuracy.png)            | ![](img/graph_loss_accuracy/CNN_13_elu_loss.png)            |
| ![](img/graph_loss_accuracy/CNN_13_relu_accuracy.png)           | ![](img/graph_loss_accuracy/CNN_13_relu_loss.png)           |
| ![](img/graph_loss_accuracy/CNN_13_sigmoid_accuracy.png)        | ![](img/graph_loss_accuracy/CNN_13_sigmoid_loss.png)        |
| ![](img/graph_loss_accuracy/CNN_14_elu_accuracy.png)            | ![](img/graph_loss_accuracy/CNN_14_elu_loss.png)            |
| ![](img/graph_loss_accuracy/CNN_14_relu_accuracy.png)           | ![](img/graph_loss_accuracy/CNN_14_relu_loss.png)           |
| ![](img/graph_loss_accuracy/CNN_14_sigmoid_accuracy.png)        | ![](img/graph_loss_accuracy/CNN_14_sigmoid_loss.png)        |

[comment]: # (graph_table_end)

### Анализ



