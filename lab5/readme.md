# Лабораторная работа №5


# Выбранные модели

## CNN_10_relu

|         Model name          |                     Model graph                      |
| :-------------------------- | :--------------------------------------------------- |
| CNN_10_relu                 | ![](../lab3/img/graph_model/CNN_10_relu_model.png)   |

|                            Accuracy                             |                            Loss                             |
| :-------------------------------------------------------------- | :---------------------------------------------------------- |
| ![](../lab3/img/graph_loss_accuracy/CNN_10_relu_accuracy.png)   | ![](../lab3/img/graph_loss_accuracy/CNN_10_relu_loss.png)   |

|      Model name       | Test accuracy | Test loss | Train accuracy | Train loss | Time_train (s) |
| :-------------------- | :-----------: | :-------: | :------------: | :--------: | :------------: |
| CNN_10_relu           |    0.9481     |  0.3136   |     0.999      |   0.0033   |   2071.7088    |


# Численные эксперименты
## Параметры
x_train = (86989, (32, 32, 3))

y_train = (86989, 43)

x_test = (12630, (32, 32, 3)) 

y_test = (12630, 43)

loss = CrossEntropy 

loss = MeanSquaredError - для моделей с обучением без учителя

optimizer = Adam

learning_rate = 0.001

batch_size = 128

num_epochs = 10

[comment]: # (result_table_start)

|                 Model name                  | Test accuracy | Test loss | Train accuracy | Train loss | Time_train (s) |
| :------------------------------------------ | :-----------: | :-------: | :------------: | :--------: | :------------: |
| del                                         |    0.8823     |  0.7741   |     0.9941     |   0.0216   |    781.0184    |
| NASNetMobile                                |    0.3606     |   9.435   |     0.3812     |   9.0008   |   1682.9806    |
| base_NASNetMobile                           |    0.0001     |  8.9801   |     0.0002     |   9.5025   |    28.2086     |
| NASNetMobile_zoom_data                      |    0.2371     |  11.8239  |     0.266      |  11.4953   |   10204.6502   |
| NASNetMobile_512_256_128_64                 |    0.0272     |  4.3338   |     0.0471     |   4.5555   |    234.9676    |
| base_NASNetMobile_zoom_data                 |      0.0      |  8.1943   |     0.0001     |   8.2727   |    77.7223     |
| NASNetMobile_with_classifier                |    0.4541     |  4.0931   |     0.5236     |   3.4453   |   1709.9733    |
| base_NASNetMobile_with_classifier           |    0.0371     |  4.5505   |     0.0461     |   4.8162   |    234.5792    |
| NASNetMobile_with_classifier_zoom_data      |    0.0574     |  14.7554  |     0.0706     |  14.5958   |   10318.4132   |
| base_NASNetMobile_with_classifier_zoom_data |    0.0071     |  16.0032  |     0.0233     |  15.7433   |    1807.578    |

[comment]: # (result_table_end)

## Анализ
