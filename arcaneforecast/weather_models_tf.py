# pyright: reportUnknownMemberType=false
# pyright: reportMissingTypeStubs=false

import tensorflow as tf
import keras

import arcaneforecast.data.data_models as data_models


class WeatherModelV1(keras.Model):
    def __init__(self, embedding_dimensions: int, lstm_units: int):
        # Embedding dimensions is the number of values in a WeatherTimePoint. Since a WeatherTimePoint is being predicted, it is also the number of output dimensions.
        super().__init__()
        self.lstm: keras.layers.LSTM = keras.layers.LSTM(lstm_units)
        self.dense: keras.layers.Dense = keras.layers.Dense(embedding_dimensions)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        lstm_output: tf.Tensor = self.lstm(inputs)
