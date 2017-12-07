import os
import csv
import random

import numpy as np

from glob import glob
from keras.utils import to_categorical
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array


TARGET_TRAIN = './train.csv'
TARGET_TEST = './test.csv'


class Dataset(object):
    def __init__(self, batch_size, image_shape):
        self.batch_size = batch_size
        self.image_shape = image_shape

        self.train = self.get_data('train')
        self.test = self.get_data('test')

        self.classes = self.get_classes(self.train)


    def get_data(self, train_or_test):
        data_file = TARGET_TRAIN if train_or_test == 'train' else TARGET_TEST

        with open(data_file, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)

            return np.array(data)


    def get_classes(self, data):
        classes = []

        for item in data:
            if item[0] not in classes:
                classes.append(item[0])

        return sorted(classes)


    def get_class_one_hot(self, class_str):
        label = self.classes.index(class_str)
        label = to_categorical(label, len(self.classes))[0]

        return np.array(label)


    def image_generator(self, train_or_test):
        data = self.train if train_or_test == 'train' else self.test

        print('Creating {} generator with {} samples.'.format(train_or_test, len(data)))

        while True:
            X, y = [], []

            for _ in range(self.batch_size):
                sample = random.choice(data)

                image = self.preprocess_image(sample[1])

                X.append(image)
                y.append(self.get_class_one_hot(sample[0]))

            yield np.array(X), np.array(y)


    def preprocess_image(self, image):
        x = load_img(image, target_size=self.image_shape[:2])
        x = img_to_array(x)
        x = preprocess_input(x)

        return x


    def load_sample(self, sample):
        image = self.preprocess_image(sample[1])
        image = np.expand_dims(image, axis=0)

        return image
