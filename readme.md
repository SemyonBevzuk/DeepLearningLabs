# Лабораторные работы по курсу "Глубокое обучение"
В рамках лабораторных работ будут реализованы разные конфигурации
нейронных сетей для решения задачи классификации.

## Постановка задачи
По входным данным *X* вычислить выход сети *U*.

Задача обучения сети заключается 
в минимизации функции ошибки *E(U,Y)*, где *Y* - ожидаемый выход сети.

![](https://latex.codecogs.com/svg.latex?min(E(w,U,Y)))

## Данные
Источник данных: [Traffic Signs Preprocessed](https://www.kaggle.com/valentynsichkar/traffic-signs-preprocessed)

Данные лежат в формате .pickle в виде словаря с ключами: y_train, x_train, y_test, x_test.

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
Предобработка данных заключается в нормализации и вычитании "среднего изображения".

Пример данных для обучения после предобработки:

![](img/training_examples_2.png "Исходные данные train")

Для использования в FCNN данные представляются в виде вектора, размерность которого есть 32 * 32 * 3 = 3072.
Для CNN в виде набора матриц (32, 32, 3)


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
На скрытых слоях будем использовать следующие функции:

| Функции активации                                                                  |
| :--------------------------------------------------------------------------------- |
| ![](https://quicklatex.com/cache3/b1/ql_80c01964e615f46f7cdc36ecc3c2bfb1_l3.png)   |
| ![](https://quicklatex.com/cache3/3c/ql_68ba08bd227169249570bf7a70bf823c_l3.png)   |
| ![](https://quicklatex.com/cache3/66/ql_64d31cb6dec95eaa8d220b31f2b93a66_l3.png)   | 

### На выходном слое
На выходе будем использовать функцию Softmax:

![](https://latex.codecogs.com/svg.latex?Softmax=\frac{e^{u_j}}{\sum\limits_{i=0}^ne^{u_i}})

## Функция ошибки
В качестве функции ошибки рассмотрим кросс-энтропию:

![](https://latex.codecogs.com/svg.latex?E(w,&space;U,&space;Y)&space;=&space;-\frac{1}{L}\sum_{k=1}^{L}\sum_{m=1}^{M}y_{m}^{k}\ln&space;\frac{e^{u_{m}^{k}}}{\sum_{j=1}^{M}e^{u_{j}^{k}}}&space;&plus;&space;(1-y_{m}^{k})\ln&space;\left&space;(1&space;-&space;\frac{e^{u_{m}^{k}}}{\sum_{j=1}^{M}e^{u_{j}^{k}}}&space;\right&space;))

Где *y* - выход сети, *u* - ожидаемый выход, *M* - число нейронов на выходном слое, *L* - число примеров.

## Задача обучения сети

Задача обучения сети состоит в минимизации функции ошибки *E(U,Y)*.

На последнем слое мы используем Softmax, так что её вычисление упростится.

## Метрика качества

Наши данные являются сбалансированными, то есть каждому классу соответствует одинаковое число примеров в обучающей выборке.
Следовательно, мы можем использовать метрику accuracy:

![](https://latex.codecogs.com/svg.latex?accuracy&space;=&space;\frac{1}{N}\sum_{i=1}^{N}[Y_i(X_i)&space;=&space;U_i])

Где *Y* - выход сети,  *X* - вход сети, *U* - ожидаемый выход.

То есть метрика качества - доля правильно классифицированных объектов из выборки.

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

math - для математических вычислений

matplotlib - для визуализации

pydot, graphviz - для визуализации графа сети

datetime - для замера времени работы

json - для хранения статистки

re - для регулярных выражений

os - для работы с файловой системой
