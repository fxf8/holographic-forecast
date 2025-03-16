# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false

from collections.abc import Collection, Sequence, Generator, Mapping
from typing import Callable, cast, SupportsFloat
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import tensorflow as tf

import holographic_forecast.data.data_models as data_models


@dataclass
class WeatherTimePointEmbedder:
    weather_time_point: data_models.WeatherTimePoint

    def embed_weather_data(
        self,
        ordering: Sequence[data_models.WeatherQuantity] | None = None,
        parameter_to_float: Mapping[
            data_models.WeatherQuantity, Callable[[float | str], SupportsFloat]
        ]
        | None = None,
    ) -> npt.NDArray[np.float32]:
        if parameter_to_float is None:
            parameter_to_float = {}

        # Only use parameter_to_float when its weather quantity exists.
        # If its a string, use `float`
        if ordering is None:
            return np.array(
                [
                    parameter_to_float.get(item[0], float)(item[1])
                    for item in self.weather_time_point.data
                    if not isinstance(item[1], str)
                ]
            )

        embedded_data: Sequence[SupportsFloat] = []

        existing_data: Sequence[tuple[data_models.WeatherQuantity, float | str]] = [
            *self.weather_time_point.data
        ]

        existing_quantities_and_index: Mapping[data_models.WeatherQuantity, int] = {
            quantity[0]: index
            for index, quantity in enumerate(self.weather_time_point.data)
            if quantity in existing_data
        }

        for expected_weather_quantity in ordering:
            if expected_weather_quantity not in self.weather_time_point.data:
                raise ValueError(
                    f"Expected to find quantity {expected_weather_quantity} in data, but did not."
                )

            value: str | float = existing_data[
                existing_quantities_and_index[expected_weather_quantity]
            ][1]

            if (
                isinstance(value, str)
                and expected_weather_quantity not in parameter_to_float
            ):
                raise ValueError(
                    f"Expected to find parameter_to_float for quantity {expected_weather_quantity}, but did not."
                )

            embedded_data.append(
                parameter_to_float.get(expected_weather_quantity, float)(value)
            )

        return np.array(embedded_data)

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
    ) -> npt.NDArray[np.float32]:  # shape (n_features,)
        return np.concatenate(
            [
                self.embed_weather_data(
                    ordering=ordering, parameter_to_float=parameter_to_float
                ),
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
    ) -> npt.NDArray[np.float32]:
        return np.array(  # index represents point over time
            [
                WeatherTimePointEmbedder(weather_time_point).embed(
                    predicted_cordinate=predicted_cordinate,
                    ordering=ordering,
                    parameter_to_float=parameter_to_float,
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
        | Collection[data_models.GeographicCordinate]
        | None = None,
        ordering: Sequence[data_models.WeatherQuantity] | None = None,
        parameter_to_float: dict[
            data_models.WeatherQuantity, Callable[[float | str], np.float32]
        ]
        | None = None,
    ) -> Collection[tf.Tensor]:  # Tensor shape (timesteps, n_points, n_features)
        """
        Embeds the tensor input for the weather prediction model.

        Args:
            predicted_cordinate (data_models.GeographicCordinate | Colection[data_models.GeographicCordinate] | None): The cordinate to predict the weather for. If None, all points in the area will be placed in parallel in the batch. If a collection, each cordinate will be placed in parallel in the batch.
            ordering (Sequence[data_models.WeatherQuantity]): The ordering of the weather quantities.
            parameter_to_float (dict[data_models.WeatherQuantity, Callable[[float | str], np.float32]]): The function to convert the weather quantity to float.
        """

        if len(self.weather_span_area.data) == 0:
            raise ValueError("The weather span area is empty.")

        if isinstance(predicted_cordinates, data_models.GeographicCordinate):
            predicted_cordinates = [predicted_cordinates]

        elif predicted_cordinates is None:
            predicted_cordinates = [
                *cast(
                    Generator[data_models.GeographicCordinate],
                    self.weather_span_area.geographic_points(),
                )
            ]

        return [
            tf.convert_to_tensor(
                np.array(
                    [
                        WeatherTimeAreaEmbedder(
                            weather_time_area
                        ).embed_to_numpy_2D_array(
                            predicted_cordinate=predicted_cordinate,
                            ordering=ordering,
                            parameter_to_float=parameter_to_float,
                        )
                        for weather_time_area in self.weather_span_area
                    ]
                )
            )
            for predicted_cordinate in predicted_cordinates
        ]
