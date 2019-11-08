# Лабораторные работы по курсу "Глубокое обучение"
Реализована двухслойная полносвязная нейронная сеть
для распознавания цифр из базы MNIST.

Входной слой - изображение из базы MNIST

Скрытый слой - функция активации ReLU

Выходной слой - функция активации Softmax

Функция ошибки - кросс-энтропия

## Математическая модель нейрона
Математическая модель нейрона имеет следующий вид:

![](https://latex.codecogs.com/svg.latex?u_k=b_k+\sum\limits_{j=1}^nw_{k,j}x_j\qquad&space;y_k=\phi(u_k))

Где ![](https://latex.codecogs.com/svg.latex?\phi) - функция активации, ![](https://latex.codecogs.com/svg.latex?b_k) -
смещение, ![](https://latex.codecogs.com/svg.latex?w_{k,j}) - вес, ![](https://latex.codecogs.com/svg.latex?x) - вход.

Для удобства выкладок сделаем некоторое преобразование. Внесем смещение в сумму с новым значением синапса ![](https://latex.codecogs.com/svg.latex?x_0=1).
Тогда модель нейрона можно записать в следующем виде:

![](https://latex.codecogs.com/svg.latex?u_k=\sum\limits_{j=0}^nw_{k,j}x_j\qquad&space;y_k=\phi(u_k))

## Предобработка данных
Входные данные нормируются, представляются как матрицы и вектора.

## Начальная инициализация весов
Инициализация весов осуществляется с использованием метода Ксавье.

![](https://latex.codecogs.com/svg.latex?W=\sigma*N(0,1)\qquad&space;\sigma=\frac{2}{\sqrt{size_{input}+size_{output}}})

## Функции активации
### На скрытом слое
На скрытом слое будем использовать ReLU:

![](https://latex.codecogs.com/svg.latex?\phi^{(1)}(u)=max(0,u))

### На выходном слое
На выходе будем использовать функцию Softmax:

![](https://latex.codecogs.com/svg.latex?\phi^{(2)}(u_j)=\frac{e^{u_j}}{\sum\limits_{i=0}^ne^{u_i}})

Её производные:

![](https://latex.codecogs.com/svg.latex?\frac{\partial\phi^{(2)}(u_j)}{\partial{u_j}}=\phi^{(2)}(u_j)(1-\phi^{(2)}(u_j)))

![](https://latex.codecogs.com/svg.latex?\frac{\partial\phi^{(2)}(u_j)}{\partial{u_i}}=-\phi^{(2)}(u_j)\phi^{(2)}(u_i))

## Функция ошибки
В качестве функции ошибки рассмотрим кросс-энтропию:

![](https://latex.codecogs.com/svg.latex?E(w)=\sum\limits_{j=1}^My_j\ln{u_j})

![](https://latex.codecogs.com/svg.latex?u_j&space;=&space;\phi^{(2)}\left&space;(\sum_{s=0}^{K}w_{j,s}^{(2)}v_s&space;\right&space;))

![](https://latex.codecogs.com/svg.latex?v_s&space;=&space;\phi^{(1)}\left&space;(\sum_{i=0}^{N}w_{s,i}^{(1)}x_i&space;\right&space;))

Где *y* - выход сети, *u* - ожидаемый выход, *v* - выход скрытого слоя, *x* - вход сети, *M* - число нейронов на выходном слое,
*K* - число нейронов на скрытом слое, *N* - число нейронов на входе сети, ![](https://latex.codecogs.com/svg.latex?\inline&space;w_{j,s}^{(2)}) -
веса выходного слоя, ![](https://latex.codecogs.com/svg.latex?\inline&space;w_{s,i}^{(1)}) -
веса скрытого слоя.

## Производные функции ошибки
### По выходному слою

![](https://latex.codecogs.com/svg.latex?\frac{\partial&space;E(w)}{\partial&space;w_{j,s}^{(2)}}=\sum\limits_{j=0}^M&space;y_j&space;\frac{\partial&space;\ln&space;u_j}{\partial{w_{j,s}^{(2)}}}&space;=&space;\sum\limits_{j=0}^M&space;y_j&space;\frac{\partial&space;\ln&space;u_j}{\partial{u_j}}&space;\frac{\partial&space;u_j}{\partial&space;w_{j,s}^{(2)}}=...)

![](https://latex.codecogs.com/svg.latex?\frac{\partial&space;\ln&space;u_j}{\partial&space;u_j}&space;=&space;\frac{1}{u_j})

![](https://latex.codecogs.com/svg.latex?\frac{\partial&space;u_j}{\partial&space;w_{j,s}^{(2)}}&space;=&space;\frac{\partial&space;u_j(\sum_{s=0}^{K}w_{j,s}^{(2)}v_s)}{\partial&space;\sum_{s=0}^{K}w_{j,s}^{(2)}v_s}&space;\frac{\partial&space;\sum_{s=0}^{K}w_{j,s}^{(2)}v_s}{\partial&space;w_{j,s}^{(2)}}&space;=&space;\frac{\partial&space;u_j(\sum_{s=0}^{K}w_{j,s}^{(2)}v_s)}{\partial&space;\sum_{s=0}^{K}w_{j,s}^{(2)}v_s}&space;v_s)

Первый множитель - это производная Softmax по аргументу. Она может принимать два значения, это зависит от слагаемого,
по которому мы берем производную. Если он в числителе: ![](https://latex.codecogs.com/svg.latex?\inline&space;\frac{\partial\phi^{(2)}(u_j)}{\partial{u_j}}=\phi^{(2)}(u_j)(1-\phi^{(2)}(u_j))).
Если в знаменателе: ![](https://latex.codecogs.com/svg.latex?\inline&space;\frac{\partial\phi^{(2)}(u_j)}{\partial{u_i}}=-\phi^{(2)}(u_j)\phi^{(2)}(u_i)).

Вынесем из суммы слагаемое, которое соответствует производной по числителю Softmax. Учтем, что ![](https://latex.codecogs.com/svg.latex?\inline&space;\sum_{j=0}^{M}y_j=1).

![](https://latex.codecogs.com/svg.latex?...=\left(y_j\frac{1}{u_j}u_j(1-u_j)+\sum_{j=0}^{M}y_j\frac{1}{u_j}(-u_ju_j)\right)v_s=\left(y_j-y_ju_j-\sum_{j=0}^{M}u_jy_j\right)v_s=(y_j-u_j)v_s=\delta_j^{(2)}v_s)

Отлично, мы нашли производную функции ошибки по выходному слою.

### По скрытому слою

![](https://latex.codecogs.com/svg.latex?\frac{\partial&space;E(w)}{\partial&space;w_{s,i}^{(1)}}=\sum\limits_{j=0}^M&space;y_j&space;\frac{\partial&space;\ln&space;u_j}{\partial&space;w_{s,i}^{(1)}}&space;=&space;\sum\limits_{j=0}^M&space;y_j&space;\frac{\partial&space;\ln&space;u_j}{\partial&space;u_j}&space;\frac{\partial&space;u_j}{\partial&space;w_{s,i}^{(1)}}&space;=&space;\sum\limits_{j=0}^M&space;\frac{y_j&space;}{u_j}&space;\frac{\partial&space;u_j(\sum_{s=0}^{K}w_{j,s}^{(2)}v_s)}{\partial&space;\sum_{s=0}^{K}w_{j,s}^{(2)}v_s}&space;\frac{\partial&space;\sum_{s=0}^{K}w_{j,s}^{(2)}v_s}{\partial&space;w_{s,i}^{(1)}}=...)

![](https://latex.codecogs.com/svg.latex?\inline&space;\frac{\partial&space;u_j(\sum_{s=0}^{K}w_{j,s}^{(2)}v_s)}{\partial&space;\sum_{s=0}^{K}w_{j,s}^{(2)}v_s}) -
производная от Softmax.

![](https://latex.codecogs.com/svg.latex?\frac{\partial\sum_{s=0}^{K}w_{j,s}^{(2)}v_s}{\partial&space;w_{s,i}^{(1)}}&space;=&space;w_{j,s}^{(2)}&space;\frac{\partial&space;\phi^{(1)}\left&space;(&space;\sum_{s=0}^{N}w_{s,i}^{(1)}x_i&space;\right&space;)}{\partial&space;\sum_{s=0}^{N}w_{s,i}^{(1)}x_i}&space;\frac{\partial&space;\phi^{(1)}\left&space;(&space;\sum_{s=0}^{N}w_{s,i}^{(1)}x_i&space;\right&space;)}{\partial&space;w_{s,i}^{(1)}}&space;=&space;w_{j,s}^{(2)}&space;\frac{\partial&space;\phi^{(1)}\left&space;(&space;\sum_{s=0}^{N}w_{s,i}^{(1)}x_i&space;\right&space;)}{\partial&space;\sum_{s=0}^{N}w_{s,i}^{(1)}x_i}&space;x_i)

Второй множитель - производная ReLU.

![](https://latex.codecogs.com/svg.latex?...=(y_j&space;\frac{1}{u_j}&space;u_j&space;(1-u_j)&space;w_{j,s}^{(2)}&space;x_i&space;-&space;\sum_{j=0}^{M}y_j&space;\frac{1}{u_j}&space;u_j&space;u_j&space;w_{j,s}^{(2)}&space;x_i)\dot{\phi^{(1)}}\left&space;(&space;\sum_{i=0}^{N}w_{s,i}^{(1)}x_i&space;\right&space;)=&space;(y_j&space;w_{j,s}^{(2)}&space;x_i&space;-&space;y_j&space;w_{j,s}^{(2)}&space;x_i&space;u_j&space;-&space;\sum_{j=0}^{M}y_j&space;u_j&space;w_{j,s}^{(2)}&space;x_i)&space;\dot{\phi^{(1)}}\left&space;(&space;\sum_{i=0}^{N}w_{s,i}^{(1)}x_i&space;\right&space;)&space;=&space;(y_j&space;w_{j,s}^{(2)}&space;x_i&space;-&space;u_j&space;w_{j,s}^{(2)}&space;x_i)&space;\dot{\phi^{(1)}}\left&space;(&space;\sum_{i=0}^{N}w_{s,i}^{(1)}x_i&space;\right&space;)=&space;(y_j-u_j)w_{j,s}^{(2)}x_i\dot{\phi^{(1)}}\left&space;(&space;\sum_{i=0}^{N}w_{s,i}^{(1)}x_i&space;\right&space;)=\delta_s^{(1)}x_i\dot{\phi^{(1)}}\left&space;(&space;\sum_{i=0}^{N}w_{s,i}^{(1)}x_i&space;\right&space;))

Нашли производную функции ошибки по скрытому слою.

## Прямой проход
На вход принимаем нормированные изображения в форме вектора. Для каждого нейрона скрытого слоя вычисляем взвешенную сумму
по всем входным нейронам и вычисляем функцию активации ReLU.

![](https://latex.codecogs.com/svg.latex?v_s&space;=&space;\phi^{(1)}\left&space;(\sum_{i=0}{N}w_{s,i}^{(1)}x_i&space;\right&space;))

После этого результат работы скрытого слоя умножаем на веса и пропускаем через функцию активации Softmax.

![](https://latex.codecogs.com/svg.latex?u_j&space;=&space;\phi^{(2)}\left&space;(\sum_{s=0}^{K}w_{j,s}^{(2)}v_s&space;\right&space;))

Выходной слой даёт вектор значений достоверности.

## Обратное распространение

1. Прямой проход по сети
* Вычисляем ![](https://latex.codecogs.com/svg.latex?\inline&space;v_s,u_s)
* Сохраняем ![](https://latex.codecogs.com/svg.latex?\inline&space;\sum_{i=0}^{N}w_{s,i}^{(1)}x_i)
* Сохраняем ![](https://latex.codecogs.com/svg.latex?\inline&space;\phi^{(1)}(\sum_{i=0}^{N}w_{s,i}^{(1)}x_i))
* Сохраняем ![](https://latex.codecogs.com/svg.latex?\inline&space;\phi^{(2)}(\sum_{s=0}^{K}w_{j,s}^{(2)}v_s))
2. Вычисляем градиент *E(W)*
3. Обратный проход с коррекцией весов: ![](https://latex.codecogs.com/svg.latex?\inline&space;w(k+1)=w(k)+\eta&space;\nabla&space;E(w))

## Обучение по тестовой выборке
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

keras - фреймворк для сравнения реализаций

numpy - для работы с векторами и матрицами

datetime - для замера времени работы

### read_mnist.py
Функции для чтения данных MNIST из пакета keras и нормировки данных.

**get_MNIST_Keras** - вернет матрично-векторные данных MNIST
**normalized_MNIST** - нормирует данные

### keras_net.py
Содержит реализацию аналогичной сети, но средствами фреймворка.

**fit_and_test_net_on_MNIST(hidden_size, batch_size, num_epochs, lr)** - обучает сеть по заданным параметрам, собирает
метрики

### my_net.py
Содержит собственную реализацию сети.

Класс **NeuralNetwork** - реализация сети.
   * **forward(X)** - прямое распространение со входом Х
   * **predict(X, Y)** - прямое распространение со входом Х, считает функцию потерь по Y
   * **fit(X, Y, batch_size, number_epochs)** - тренировка сети с заданными параметрами

**fit_and_test_net_on_MNIST(hidden_size, batch_size, num_epochs, lr)** - - обучает сеть по заданным параметрам, собирает
метрики

### main.py
Парсит аргументы командной строки, запускает нужную сеть или сравнивает их.

Параметры скрипта:
* **net_type** - выбор реализации сети для запуска: 'my', 'keras'
* **hidden_size** - число узлов на скрытом слое
* **lr** - скорость обучения
* **batch_size** - размер пачки
* **number_epochs** - число эпох
* **compare_nets** - флаг для сравнения двух реализаций на указанных параметрах

Пример командной строки для запуска:
    
    python main.py --net_type my --hidden_size 256 --lr 0.1 --batch_size 128 --number_epochs 20 --compare_nets

## Результаты
Была реализована полносвязная двухслойная нейронная сеть с алгоритмом обратного распространения ошибки.
Реализация сравнивалась с аналогичной сетью из фреймворка Keras.

### Параметры запуска:
* hidden_size = 30
* lr = 0.1
* batch_size = 128
* number_epochs = 20

Командная строка:

    python main.py --net_type my --hidden_size 30 --lr 0.1 --batch_size 128 --number_epochs 20 --compare_nets

|           | Time(s)   | Test accuracy(%)| Test loss | Train accuracy(%)| Train loss | 
|:--------- |:---------:| :--------------:|:---------:|:----------------:|:----------:|
| my_net    | 25.023557 | 0.9648          | 0.122476  | 0.974233         | 0.090380   |
| keras_net | 20.960236 | 0.965699        | 0.114408  | 0.975350         | 0.086029   |

### Параметры запуска:
* hidden_size = 128
* lr = 0.1
* batch_size = 128
* number_epochs = 20

Командная строка:

    python main.py --net_type my --hidden_size 128 --lr 0.1 --batch_size 128 --number_epochs 20 --compare_nets

|           | Time(s)   | Test accuracy(%)| Test loss | Train accuracy(%)| Train loss | 
|:--------- |:---------:| :--------------:|:---------:|:----------------:|:----------:|
| my_net    | 41.901075 | 0.9756          | 0.078803  | 0.988916         | 0.044473   |
| keras_net | 26.540353 | 0.976100        | 0.077987  | 0.988150         | 0.046889   |

### Параметры запуска:
* hidden_size = 256
* lr = 0.1
* batch_size = 128
* number_epochs = 20

Командная строка:

    python main.py --net_type my --hidden_size 256 --lr 0.1 --batch_size 128 --number_epochs 20 --compare_nets

|           | Time(s)   | Test accuracy(%)| Test loss | Train accuracy(%)| Train loss | 
|:--------- |:---------:| :--------------:|:---------:|:----------------:|:----------:|
| my_net    | 64.535192 | 0.978000        | 0.071693  | 0.991483         | 0.036636   |
| keras_net | 33.041822 | 0.976999        | 0.075315  | 0.989810         | 0.041615   |

### Параметры запуска:
* hidden_size = 300
* lr = 0.1
* batch_size = 128
* number_epochs = 20

Командная строка:

    python main.py --net_type my --hidden_size 300 --lr 0.1 --batch_size 128 --number_epochs 20 --compare_nets

|           | Time(s)    | Test accuracy(%)| Test loss | Train accuracy(%)| Train loss | 
|:--------- |:----------:| :--------------:|:---------:|:----------------:|:----------:|
| my_net    | 107.717036 | 0.978200        | 0.070821  | 0.992            | 0.036216   |
| keras_net | 40.644219  | 0.978299        | 0.072413  | 0.990450         | 0.040741   |