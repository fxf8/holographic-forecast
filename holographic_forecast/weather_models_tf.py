# pyright: reportUnknownMemberType=false
# pyright: reportMissingTypeStubs=false

from typing import override, cast

import numpy as np
import tensorflow as tf
import keras


@keras.saving.register_keras_serializable()
class WeatherModelV1(keras.Model):
	"""
	WeatherModelV1: Weather Model Version 1

	Model Sequence (Step, Result):
	    1) Input: (batch_size, timesteps, n_points, n_features)
	    2) Linear + Softmax reduction over points: (batch_size, timesteps, n_features_1)
	    3) keras GRU over timesteps: (batch_size, n_features_2 = GRU units)
	    4) keras Dense layer (batch_size, n_features_3)
	    5) Linear layer to final prediction: (batch_size, n_features)
	"""

	def __init__(
		self,
		n_features: int,
		n_reduced_features_points: int | None = None,
		n_reduced_features_timesteps: int | None = None,
	):
		super().__init__()
		self.n_features: int = n_features

		# Shape: (n_features, n_features_1)
		self.point_reduction_kernel: keras.Variable = self.add_weight(
			shape=(
				n_features,
				n_reduced_features_points
				if n_reduced_features_points is not None
				else n_features,
			),
			trainable=True,
		)

		self.timesteps_reduction_gru: keras.layers.GRU = keras.layers.GRU(
			n_features
			if n_reduced_features_timesteps is None
			else n_reduced_features_timesteps
		)

		self.dense_layer: keras.layers.Dense = keras.layers.Dense(n_features, "softmax")
		self.final_layer: keras.layers.Dense = keras.layers.Dense(n_features)

	@override
	def call(self, inputs: tf.Tensor) -> tf.Tensor:
		# Input shape: (batch_size (a), timesteps (b), n_points (c), n_features (d))

		# Shape: (batch_size (a), timesteps (b), n_features_1 (e))
		points_reduced: tf.Tensor = cast(
			tf.Tensor, tf.einsum("abcd,de->abe", inputs, self.point_reduction_kernel)
		)

		# Shape: (batch_size (a), n_features_2 (f))
		timesteps_reduced: tf.Tensor = cast(
			tf.Tensor, self.timesteps_reduction_gru(points_reduced)
		)

		# Shape: (batch_size (a), n_features_3 (g))
		dense_output: tf.Tensor = cast(tf.Tensor, self.dense_layer(timesteps_reduced))

		# Shape: (batch_size (a), n_features (d))
		final_output: tf.Tensor = cast(tf.Tensor, self.final_layer(dense_output))

		return final_output
