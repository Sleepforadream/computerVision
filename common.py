import cv2
import keras.models
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from PIL import Image, ImageOps


class Model:
    model: Sequential = None

    # Переменные, которые будут использоваться для хранения наборов данных MNIST в формате numpy массивов.

    # train_a и train_b будут использоваться для хранения тренировочного набора данных MNIST
    train_a = None
    train_b = None

    # test_a и test_b - для хранения тестового набора данных MNIST
    test_a = None
    test_b = None

    # Создаём и обучаем модель на тренировочном наборе данных MNIST, используем библиотеку Keras
    def build_model(self):
        # Загружаем набор данных MNIST с помощью функции mnist.load_data()
        (self.train_a, self.train_b), (self.test_a, self.test_b) = mnist.load_data()

        # Изменяем значения пикселей из диапазона 0-255 в диапазон 0-1
        self.train_a = self.train_a.astype('float32') / 255.
        self.test_a = self.test_a.astype('float32') / 255.

        # Добавляем новую ось для соответствия формата входных данных
        self.train_a = np.expand_dims(self.train_a, axis=-1)
        self.test_a = np.expand_dims(self.test_a, axis=-1)

        # Преобразуем метки классов в формат one-hot encoding
        self.train_b = to_categorical(self.train_b, 10)
        self.test_b = to_categorical(self.test_b, 10)

        # Создаём модель нейронной сети
        model = Sequential([
            # Создаём сверточный слой с 32 фильтрами, размером ядра(3, 3)
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            # Создаём сверточный слой с 64 фильтрами, размером ядра(3, 3)
            Conv2D(64, (3, 3), activation='relu'),
            # Создаём слой с размером фильтра (2,2)
            MaxPooling2D(pool_size=(2, 2)),
            # Слой Dropout для регуляризации
            Dropout(0.25),
            # Преобразуем выходные данный сверточных слоев в одномерный массив
            Flatten(),
            # Создаём полносвязный слой с 128 нейронами
            Dense(128, activation='relu'),
            # Создаём слой Dropout для регуляризации
            Dropout(0.5),
            # Создаём выходной слой с 10 нейронами и функцией активации Softmax
            Dense(10, activation='softmax')
        ])

        # Компилируем модель. Для оценки качества модели присутствует выбор функции потерь, метрик и оптимизатора
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # На тренировочном наборе данных производим обучение модели
        model.fit(self.train_a, self.train_b, batch_size=128, epochs=1, verbose=1, validation_split=0.2)
        scores = model.evaluate(self.test_a, self.test_b, verbose=1)
        print("Доля верных ответов на тестовых данных, в процентах:", round(scores[1] * 100, 4))
        self.model = model

    # Сохраняем модель в файл
    def save_model(self, destination):
        self.model.save(destination)

    # Загружаем модель из файла
    def load_model(self, source):
        self.model = keras.models.load_model(source, compile=None)

    # Распознаём созданную нами цифру на изображении и возвращаем цифру, которую распознали
    def get_recognized_number(self, source):
        # Загружаем изображение
        img = Image.open(source)
        # Изменяем изображение к numpy массиву
        img_array = np.array(img)
        img_array = np.invert(img_array)

        img_array_expanded = np.expand_dims(img_array, axis=0)
        plt.imshow(img_array, cmap=plt.cm.binary)
        plt.show()
        result = self.model.predict(img_array_expanded)
        result_digit = np.argmax(result)
        return result_digit

    # Показываем изображение из тестового набора данных MNIST с номером n и возвращаем цифру для этого изображения
    def test(self, n):
        (self.train_a, self.train_b), (self.test_a, self.test_b) = mnist.load_data()
        img = self.test_a[n]
        img_array_expanded = np.expand_dims(img, axis=0)
        plt.imshow(img, cmap=plt.cm.binary)
        plt.show()
        result = self.model.predict(img_array_expanded)
        result_digit = np.argmax(result)
        print(self.test_b[n])
        return result_digit
