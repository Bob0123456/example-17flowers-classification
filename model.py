from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.applications.vgg16 import VGG16


def CNN2D(n_classes, image_shape):
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=image_shape)

    model = Sequential()

    for layer in vgg16.layers:
        layer.trainable = False
        model.add(layer)

    model.add(Flatten(name='block6_flatten'))
    model.add(Dense(4096, activation='relu', name='block7_dense1'))
    model.add(Dense(4096, activation='relu', name='block7_dense2'))
    model.add(Dense(n_classes, activation='softmax', name='block7_dense3'))

    return model
