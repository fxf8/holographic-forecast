# pyright: reportUnknownMemberType=false
# pyright: reportMissingTypeStubs=false

from typing import cast

import tensorflow as tf
import tensorflow.python.framework.ops as tf_ops
import keras


@keras.saving.register_keras_serializable()
class WeatherModelV1(keras.Model):
    """
    Encoder: Uses WeatherEncoderV1

    Input Shape: (
            n timesteps (ragged),
            query info (shape (2 + n identifier chars (ragged)) at [0][0]) + n points (ragged),
            point info (shape (3,) at [0]) + n entries (ragged),
            value (shape () at [0]) + identifier chars (ragged)
            )

    Output shape: (
            value (shape ())
            )
    """

    weather_char_identifier_embedding: keras.layers.Embedding

    def __init__(self):
        super().__init__()

        self.weather_char_identifier_embedding = keras.layers.Embedding(
            input_dim=1, output_dim=16
        )

    def call(self, query: tf.Tensor) -> tf.Tensor:
        # Shape: (timesteps, points, entries, identifier chars)
        entry_identifier_chars: tf.Tensor = cast(tf.Tensor, query)[:, 1:, :, 1:]
