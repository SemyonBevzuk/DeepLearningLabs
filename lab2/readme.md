# Лабораторная работа №2

|           Model name           |                       Model graph                       |
| :----------------------------- | :------------------------------------------------------ |
| FCNN_128_model                 | ![](img/graph_model/FCNN_128_model.png)                 |
| FCNN_200_model                 | ![](img/graph_model/FCNN_200_model.png)                 |
| FCNN_256_model                 | ![](img/graph_model/FCNN_256_model.png)                 |
| FCNN_300_model                 | ![](img/graph_model/FCNN_300_model.png)                 |
| FCNN_512_model                 | ![](img/graph_model/FCNN_512_model.png)                 |
| FCNN_768_model                 | ![](img/graph_model/FCNN_768_model.png)                 |
| FCNN_1024_model                | ![](img/graph_model/FCNN_1024_model.png)                |
| FCNN_1536_model                | ![](img/graph_model/FCNN_1536_model.png)                |
| FCNN_2048_model                | ![](img/graph_model/FCNN_2048_model.png)                |
| FCNN_2304_model                | ![](img/graph_model/FCNN_2304_model.png)                |
| FCNN_768_384_model             | ![](img/graph_model/FCNN_768_384_model.png)             |
| FCNN_1024_512_model            | ![](img/graph_model/FCNN_1024_512_model.png)            |
| FCNN_1536_768_model            | ![](img/graph_model/FCNN_1536_768_model.png)            |
| FCNN_2304_768_model            | ![](img/graph_model/FCNN_2304_768_model.png)            |
| FCNN_768_384_96_model          | ![](img/graph_model/FCNN_768_384_96_model.png)          |
| FCNN_768_384_192_model         | ![](img/graph_model/FCNN_768_384_192_model.png)         |
| FCNN_2048_1024_512_256_model   | ![](img/graph_model/FCNN_2048_1024_512_256_model.png)   |
| FCNN_1536_768_384_192_96_model | ![](img/graph_model/FCNN_1536_768_384_192_96_model.png) |

Реализация полносвязной нейронной сети для задачи классификации.

Сравнение конфигураций полносвязных нейронных сетей.

# Описание директорий

## log
Здесь лежат файлы .json со статистикой по разным конфигурациям.
Они содержат параметры сети (число слоёв, число нейронов, параметр обучения, размер пачки, число эпох) и статистику 
обучения (время, функцию потерь на тестовом и тренировочном наборе, точность на тестовом и тренировочном наборе)

## models
Здесь лежат файлы .h5 с конфигурацией сетей Keras для их последующей повторной загрузки.

## src
Скрипты для работы с данными и фреймворком.
### datahandler.py
Содержит методы для чтения, обработки, визуализации данных.
### modelhandler.py
Содержит методы для работы с сетью: запуск обучения, сбор статистики, сохранение и загрузка сетей.
### reporthandler.py
Содержит методы для генерации таблиц для отчёта.
### notebook.py
Является точкой входа. Блокнот для проведения экспериментов. Содержит метод для запуска серийного эксперимента с 
возможностью вариации количества скрытых слоёв и числа нейронов на них.

## Численные эксперименты
### Параметры

x_train = (86989, 3072)

y_train = (86989, 43)

x_test = (12630, 3072) 

y_test = (12630, 43)

hidden_layer_activation = ReLu

output_layer_activation = Softmax

loss = CrossEntropy 

optimizer = Adam

learning_rate = 0.001

batch_size = 128

num_epochs = 15

### Результаты
[comment]: # (result_table_start)

|        Model name        | Test accuracy | Test loss | Train accuracy | Train loss | Time_train (s) |
| :----------------------- | :-----------: | :-------: | :------------: | :--------: | :------------: |
| FCNN_128                 |    0.8389     |  1.2112   |     0.9699     |   0.1048   |    129.3543    |
| FCNN_200                 |    0.8298     |  1.3422   |     0.9631     |   0.131    |    183.3019    |
| FCNN_256                 |    0.8318     |  1.3828   |     0.9768     |   0.0839   |    233.5805    |
| FCNN_300                 |    0.8199     |  1.5928   |     0.9707     |   0.1028   |    272.8827    |
| FCNN_512                 |     0.83      |  1.6081   |     0.9713     |   0.0961   |    440.9299    |
| FCNN_768                 |    0.8304     |  1.8328   |     0.9717     |   0.1066   |    645.9815    |
| FCNN_1024                |    0.8139     |  2.1581   |     0.9548     |   0.2022   |    861.3704    |
| FCNN_1536                |    0.8226     |  2.3028   |     0.9754     |   0.0896   |   1285.1339    |
| FCNN_2048                |    0.8088     |  2.8824   |     0.9552     |   0.2181   |   1749.6416    |
| FCNN_2304                |    0.8306     |   2.309   |     0.9493     |   0.2349   |   1940.3959    |
| FCNN_768_384             |    0.8261     |  2.1872   |     0.9685     |   0.1121   |    720.983     |
| FCNN_1024_512            |    0.7979     |   2.644   |     0.9596     |   0.1457   |   1003.5661    |
| FCNN_1536_768            |    0.8237     |  2.4898   |     0.9703     |   0.1043   |   1582.1086    |
| FCNN_2304_768            |      0.8      |  3.1957   |     0.9434     |   0.2568   |    2378.945    |
| FCNN_768_384_96          |    0.8272     |  1.6932   |     0.9602     |   0.1335   |    726.6468    |
| FCNN_768_384_192         |    0.8144     |  1.8249   |     0.9664     |   0.1153   |    727.8066    |
| FCNN_2048_1024_512_256   |     0.836     |  1.4335   |     0.9761     |   0.0745   |   2476.1017    |
| FCNN_1536_768_384_192_96 |    0.8137     |  1.1759   |     0.9548     |   0.1486   |   1671.4964    |

[comment]: # (result_table_end)

[comment]: # (graph_table_start)

|                              Accuracy                              |                              Loss                              |
| :----------------------------------------------------------------- | :------------------------------------------------------------- |
| ![](img/graph_loss_accuracy/FCNN_128_accuracy.png)                 | ![](img/graph_loss_accuracy/FCNN_128_loss.png)                 |
| ![](img/graph_loss_accuracy/FCNN_200_accuracy.png)                 | ![](img/graph_loss_accuracy/FCNN_200_loss.png)                 |
| ![](img/graph_loss_accuracy/FCNN_256_accuracy.png)                 | ![](img/graph_loss_accuracy/FCNN_256_loss.png)                 |
| ![](img/graph_loss_accuracy/FCNN_300_accuracy.png)                 | ![](img/graph_loss_accuracy/FCNN_300_loss.png)                 |
| ![](img/graph_loss_accuracy/FCNN_512_accuracy.png)                 | ![](img/graph_loss_accuracy/FCNN_512_loss.png)                 |
| ![](img/graph_loss_accuracy/FCNN_768_accuracy.png)                 | ![](img/graph_loss_accuracy/FCNN_768_loss.png)                 |
| ![](img/graph_loss_accuracy/FCNN_1024_accuracy.png)                | ![](img/graph_loss_accuracy/FCNN_1024_loss.png)                |
| ![](img/graph_loss_accuracy/FCNN_1536_accuracy.png)                | ![](img/graph_loss_accuracy/FCNN_1536_loss.png)                |
| ![](img/graph_loss_accuracy/FCNN_2048_accuracy.png)                | ![](img/graph_loss_accuracy/FCNN_2048_loss.png)                |
| ![](img/graph_loss_accuracy/FCNN_2304_accuracy.png)                | ![](img/graph_loss_accuracy/FCNN_2304_loss.png)                |
| ![](img/graph_loss_accuracy/FCNN_768_384_accuracy.png)             | ![](img/graph_loss_accuracy/FCNN_768_384_loss.png)             |
| ![](img/graph_loss_accuracy/FCNN_1024_512_accuracy.png)            | ![](img/graph_loss_accuracy/FCNN_1024_512_loss.png)            |
| ![](img/graph_loss_accuracy/FCNN_1536_768_accuracy.png)            | ![](img/graph_loss_accuracy/FCNN_1536_768_loss.png)            |
| ![](img/graph_loss_accuracy/FCNN_2304_768_accuracy.png)            | ![](img/graph_loss_accuracy/FCNN_2304_768_loss.png)            |
| ![](img/graph_loss_accuracy/FCNN_768_384_96_accuracy.png)          | ![](img/graph_loss_accuracy/FCNN_768_384_96_loss.png)          |
| ![](img/graph_loss_accuracy/FCNN_768_384_192_accuracy.png)         | ![](img/graph_loss_accuracy/FCNN_768_384_192_loss.png)         |
| ![](img/graph_loss_accuracy/FCNN_2048_1024_512_256_accuracy.png)   | ![](img/graph_loss_accuracy/FCNN_2048_1024_512_256_loss.png)   |
| ![](img/graph_loss_accuracy/FCNN_1536_768_384_192_96_accuracy.png) | ![](img/graph_loss_accuracy/FCNN_1536_768_384_192_96_loss.png) |

[comment]: # (graph_table_end)

### Анализ

