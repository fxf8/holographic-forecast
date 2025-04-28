# pyright: reportUnknownMemberType=false
# pyright: reportMissingTypeStubs=false

from typing import override, cast

import tensorflow as tf
import keras


@keras.saving.register_keras_serializable()
class WeatherModelV1(keras.Model):

