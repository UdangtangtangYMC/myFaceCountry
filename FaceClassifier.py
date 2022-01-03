from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, BatchNormalization, MaxPool2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
import data


class FaceModel:

    def __init__(self, image_row, image_col, batch=128, epoch=100):
        self.batch = batch
        self.epoch = epoch
        self.image_row = image_row
        self.image_col = image_col
        self.optimizer = Adam()

        self.model = self.build_model()
        self.model.compile(optimizer=self.optimizer, loss=CategoricalCrossentropy())

    def build_model(self):
        model = Sequential()
        model.add(Convolution2D(filters=64, kernel_size=3, input_shape=(self.image_row, self.image_col)))
        model.add(MaxPool2D())

        model.add(Convolution2D(filters=256, kernel_size=3))
        model.add(MaxPool2D())

        model.add(Convolution2D(filters=256, kernel_size=3))
        model.add(MaxPool2D())

        model.add(Convolution2D(filters=128, kernel_size=3))
        model.add(MaxPool2D())
        model.add(Flatten())

        model.add(Dense(units=128))
        model.add(Dense(units=7, activation='softmax'))

        return model

    def train(self, x, y):
        self.model.summary()
        self.model.fit(x, y, batch_size=self.batch, epochs=self.epoch)

        self.model.save()

    def test(self, x, y):
        loss = self.model.evaluate(x, y, batch_size=self.batch)
        return loss
