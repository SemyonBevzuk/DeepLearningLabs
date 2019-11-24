# Лабораторные работы по курсу "Глубокое обучение"
В рамках лабораторных работ будут реализованы разные конфигурации
нейронных сетей для решения задачи классификации.

## Постановка задачи
По входным данным *X* вычислить выход сети *U*. Задача обучения сети заключается
в минимизации функции ошибки *E(U,Y)*, где *Y* - ожидаемый выход сети.

В нашем случае в качестве функции ошибки выступает кросс-энтропия:


![](https://latex.codecogs.com/svg.latex?E(w)=\sum\limits_{j=1}^My_j\ln{u_j})

## Данные
Источник данных: [Traffic Signs Preprocessed](https://www.kaggle.com/valentynsichkar/traffic-signs-preprocessed)

Необходимо распознать на RGB изображении 32 * 32 один из 43 дорожных знаков.

|          |  Число примеров |
|  ------- | --------------- |
| x_train  | 86989           |
| x_test   | 12630           |

В обучающей выборке содержится 86989 примеров.
В тестирующей выборке содержится 12630 примеров.

На каждый класс приходится около 2000 примеров в обучающей выборке.

Пример данных для обучения:

![](img/training_examples_0.png "Исходные данные train")

### Предобработка данных
Предпобработка данных заключается в нормализации и вычитании "среднего изображения".

Пример данных для обучения после предобработки:

![](img/training_examples_2.png "Исходные данные train")

Для использования в сети данные представляются в виде вектор размерностью 32 * 32 * 3 = 3072.


## Математическая модель нейрона
Математическая модель нейрона имеет следующий вид:

![](https://latex.codecogs.com/svg.latex?u_k=b_k+\sum\limits_{j=1}^nw_{k,j}x_j\qquad&space;y_k=\phi(u_k))

Где ![](https://latex.codecogs.com/svg.latex?\phi) - функция активации, ![](https://latex.codecogs.com/svg.latex?b_k) -
смещение, ![](https://latex.codecogs.com/svg.latex?w_{k,j}) - вес, ![](https://latex.codecogs.com/svg.latex?x) - вход.

Для удобства выкладок сделаем некоторое преобразование. Внесем смещение в сумму с новым значением синапса ![](https://latex.codecogs.com/svg.latex?x_0=1).
Тогда модель нейрона можно записать в следующем виде:

![](https://latex.codecogs.com/svg.latex?u_k=\sum\limits_{j=0}^nw_{k,j}x_j\qquad&space;y_k=\phi(u_k))

## Функции активации
### На скрытом слое
На скрытом слое будем использовать ReLU:

![](https://latex.codecogs.com/svg.latex?\phi^{(1)}(u)=max(0,u))

### На выходном слое
На выходе будем использовать функцию Softmax:

![](https://latex.codecogs.com/svg.latex?\phi^{(2)}(u_j)=\frac{e^{u_j}}{\sum\limits_{i=0}^ne^{u_i}})

## Функция ошибки
В качестве функции ошибки рассмотрим кросс-энтропию:

![](https://latex.codecogs.com/svg.latex?E(w)=\sum\limits_{j=1}^My_j\ln{u_j})

Где *y* - выход сети, *u* - ожидаемый выход, *M* - число нейронов на выходном слое.

## Обучение на тренировочной выборке
Сеть обучается заданное число эпох. Эпоха - полный проход по тестовой выборке.
Внутри эпохи набор тестовых данных делится на пакеты.

* Инициализация весов
* Каждую эпоху
    * Перемешиваем выборку
    * Делим выборку на пакеты и для каждого пакета
        * Коррекция весов по алгоритму обратного распространения
        
## Описание реализации
### Внешние зависимости
tensorflow - основа для Keras

keras - фреймворк для работы с сетями

numpy - для работы с векторами и матрицами

pickle - для работы с изображениями

math - для математичский вычислений

matplotlib - для визуализации

pydot, graphviz - для визуализации графа сети

datetime - для замера времени работы

json - для хранения статисткии

re - для регулярных выражений

os - для работы с файловой системой
