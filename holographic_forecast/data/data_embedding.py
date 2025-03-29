# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Callable, SupportsFloat, cast

import numpy as np
import numpy.typing as npt
import tensorflow as tf

import holographic_forecast.data.data_models as data_models
import holographic_forecast.data.openmeteo_data_collection as openmeteo


@dataclass
class WeatherTimePointEmbedder:
    weather_time_point: data_models.WeatherTimePoint

    def embed_weather_data(
        self,
        expected_quantities: Sequence[data_models.WeatherQuantity] | None = None,
        parameter_to_float: Mapping[
            data_models.WeatherQuantity, Callable[[float | str], SupportsFloat]
        ]
        | None = None,
        additional_parameter_for_missing_values: bool = True, # If true, this will add an additional parameter for whether or not a value is present
    ) -> tuple[npt.NDArray[np.float32], int]:  # shape (n_features,), output_dimension
        if parameter_to_float is None:
            parameter_to_float = {}

        # Only use parameter_to_float when its weather quantity exists.
        # If its a string, use `float`

        if expected_quantities is None:
            expected_quantities = (
                openmeteo.OPEN_METEO_HOURLY_PARAMETERS
                + openmeteo.OPEN_METEO_DAILY_PARAMETERS
            )

        output_dimension: int = (
            len(expected_quantities) * 2
            if additional_parameter_for_missing_values
            else len(expected_quantities)
        )
        embedded_data: npt.NDArray[np.float32] = np.zeros(
            output_dimension, dtype=np.float32
        )

        existing_data: Sequence[tuple[data_models.WeatherQuantity, float | str]] = [
            *self.weather_time_point.data
        ]

        existing_quantities_and_index: Mapping[data_models.WeatherQuantity, int] = {
            quantity[0]: index
            for index, quantity in enumerate(self.weather_time_point.data)
            if quantity in existing_data
        }

        if additional_parameter_for_missing_values:
            for index, quantity in enumerate(expected_quantities):
                if quantity not in existing_quantities_and_index:
                    embedded_data[index * 2] = parameter_to_float.get(quantity, float)(
                        0.0
                    )

                    embedded_data[index * 2 + 1] = parameter_to_float.get(
                        quantity, float
                    )(0.0 if quantity in existing_quantities_and_index else 1.0)

        return (embedded_data, output_dimension)

    def embed_geographic_data(self) -> npt.NDArray[np.float32]:
        return np.array(
            [
                self.weather_time_point.cordinate.latitude_deg,
                self.weather_time_point.cordinate.longitude_deg,
                self.weather_time_point.elevation_meters,
            ]
        )

    def embed(
        self,
        predicted_cordinate: data_models.GeographicCordinate,
        ordering: Sequence[data_models.WeatherQuantity] | None = None,
        parameter_to_float: Mapping[
            data_models.WeatherQuantity, Callable[[float | str], np.float32]
        ]
        | None = None,
        additional_parameter_for_missing_values: bool = True,
    ) -> npt.NDArray[np.float32]:  # shape (n_features,)
        return np.concatenate(
            [
                self.embed_weather_data(
                    expected_quantities=ordering,
                    parameter_to_float=parameter_to_float,
                    additional_parameter_for_missing_values=additional_parameter_for_missing_values,
                )[0],
                self.embed_geographic_data(),
                np.array(
                    [
                        predicted_cordinate.latitude_deg,
                        predicted_cordinate.longitude_deg,
                    ]
                ),
            ],
            dtype=np.float32,
        )


@dataclass
class WeatherTimeAreaEmbedder:
    weather_time_area: data_models.WeatherTimeArea

    def embed_to_numpy_2D_array(
        self,
        predicted_cordinate: data_models.GeographicCordinate,
        ordering: Sequence[data_models.WeatherQuantity] | None = None,
        parameter_to_float: Mapping[
            data_models.WeatherQuantity, Callable[[float | str], np.float32]
        ]
        | None = None,
        additional_parameter_for_missing_values: bool = True,
    ) -> npt.NDArray[np.float32]:
        return np.array(  # index represents point over time
            [
                WeatherTimePointEmbedder(weather_time_point).embed(
                    predicted_cordinate=predicted_cordinate,
                    ordering=ordering,
                    parameter_to_float=parameter_to_float,
                    additional_parameter_for_missing_values=additional_parameter_for_missing_values,
                )
                for weather_time_point in self.weather_time_area
            ]
        )

    def embed_to_model_input(
        self,
        predicted_cordinate: data_models.GeographicCordinate,
        ordering: Sequence[data_models.WeatherQuantity] | None = None,
        parameter_to_float: Mapping[
            data_models.WeatherQuantity, Callable[[float | str], np.float32]
        ]
        | None = None,
        additional_parameter_for_missing_values: bool = True,
    ) -> tf.Tensor:  # shape (n_points, n_features)
        """
        The embedded tensor (2D) is meant to be **combined into a singular 1D tensor (through RNN)** later used in the prediction model.
        """
        return tf.convert_to_tensor(
            np.expand_dims(
                self.embed_to_numpy_2D_array(
                    predicted_cordinate=predicted_cordinate,
                    ordering=ordering,
                    parameter_to_float=parameter_to_float,
                    additional_parameter_for_missing_values=additional_parameter_for_missing_values,
                ),
                axis=0,
            )
        )


@dataclass
class WeatherSpanAreaEmbedder:
    weather_span_area: data_models.WeatherSpanArea

    def embed_model_input(
        self,
        predicted_cordinates: data_models.GeographicCordinate
        | Collection[data_models.GeographicCordinate],
        ordering: Sequence[data_models.WeatherQuantity] | None = None,
        parameter_to_float: dict[
            data_models.WeatherQuantity, Callable[[float | str], np.float32]
        ]
        | None = None,
    ) -> (
        tf.RaggedTensor
    ):  # Tensor shape (batch_size, timesteps (ragged), n_points (ragged), n_features)
        """
        Embeds the tensor input for the weather prediction model.

        Args:
            predicted_cordinate (data_models.GeographicCordinate | Colection[data_models.GeographicCordinate] | None): The cordinate to predict the weather for. If None, all points in the area will be placed in parallel in the batch. If a collection, each cordinate will be placed in parallel in the batch.
            ordering (Sequence[data_models.WeatherQuantity]): The ordering of the weather quantities.
            parameter_to_float (dict[data_models.WeatherQuantity, Callable[[float | str], np.float32]]): The function to convert the weather quantity to float.

        Returns:
            tf.Tensor: The embedded tensor. Axis: (batch_size, timesteps, n_points_per_area, n_features) Each batch is a different cordinate within the cordinates within the area whose data is provided
        """

        if len(self.weather_span_area.data) == 0:
            raise ValueError("The weather span area is empty.")

        if isinstance(predicted_cordinates, data_models.GeographicCordinate):
            predicted_cordinates = [predicted_cordinates]

        return cast(
            tf.RaggedTensor,
            tf.ragged.constant(
                [
                    [
                        WeatherTimeAreaEmbedder(
                            weather_time_area
                        ).embed_to_numpy_2D_array(
                            predicted_cordinate=predicted_cordinate,
                            ordering=ordering,
                            parameter_to_float=parameter_to_float,
                            additional_parameter_for_missing_values=True,
                        )
                        for weather_time_area in self.weather_span_area
                    ]
                    for predicted_cordinate in predicted_cordinates
                ]
            ),
        )
