# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false

from typing import cast

import tensorflow as tf

import holographic_forecast.data.data_models as data_models


class WeatherEncoderV1:
    """
    WeatherModelV1Encoder
        - Output Shape: (n timesteps (ragged), 1 + n points (ragged), 1 + n entries (ragged) | 1, 1 + n identifier chars (ragged) | 2 + n predicted identifier chars (ragged))
    """

    @staticmethod
    def encode_weather_quantity(
        weather_quantity: data_models.WeatherQuantity,
    ) -> list[float | int]:
        """
        Output Shape: (n identifier chars (ragged))
        """

        return [ord(letter) for letter in weather_quantity.identifier]

    @staticmethod
    def encode_weather_entry(
        weather_entry: data_models.WeatherEntry,
    ) -> list[float | int]:
        """
        Output Shape: (
                value (shape () at [0]) + identifier chars (ragged)
                )
        """

        return [weather_entry.value] + WeatherEncoderV1.encode_weather_quantity(
            weather_entry.quantity
        )

    @staticmethod
    def encode_weather_time_point(
        weather_time_point: data_models.WeatherTimePoint,
    ) -> list[list[float | int]]:
        """
        Output Shape: (
                point info (shape (3,) at [0]) + n entries (ragged)
                value (shape () at [0]) + identifier chars (ragged)
                )
        """

        return [
            [
                weather_time_point.cordinate.latitude_deg,
                weather_time_point.cordinate.longitude_deg,
                weather_time_point.time.timestamp(),
            ],
            *(
                WeatherEncoderV1.encode_weather_entry(entry)
                for entry in weather_time_point.data
            ),
        ]

    @staticmethod
    def encode_weather_time_area(
        weather_time_area: data_models.WeatherTimeArea,
        predicted_cordinate: data_models.GeographicCordinate,
        predicted_quantity: data_models.WeatherQuantity,
    ) -> list[list[list[float | int]]]:
        """
        Output Shape: (
                query info (shape (2 + n identifier chars (ragged)) at [0][0]) + n points (ragged),
                point info (shape (3,) at [0]) + n entries (ragged)
                value (shape () at [0]) + identifier chars (ragged)
                )
        """

        encoded_query: list[float | int] = [  # Ragged
            predicted_cordinate.latitude_deg,
            predicted_cordinate.longitude_deg,
            *WeatherEncoderV1.encode_weather_quantity(predicted_quantity),
        ]

        return [[encoded_query]] + [
            WeatherEncoderV1.encode_weather_time_point(weather_time_point)
            for weather_time_point in weather_time_area.data
        ]

    @staticmethod
    def encode_weather_timespan_area(
        weather_timespan_area: data_models.WeatherTimespanArea,
        predicted_cordinate: data_models.GeographicCordinate,
        predicted_quantity: data_models.WeatherQuantity,
    ) -> list[list[list[list[float | int]]]]:
        """
        Output Shape: (
                n timesteps (ragged),
                query info (shape (2 + n identifier chars (ragged)) at [0][0]) + n points (ragged),
                point info (shape (3,) at [0]) + n entries (ragged)
                value (shape () at [0]) + identifier chars (ragged)
                )
        """

        return [
            WeatherEncoderV1.encode_weather_time_area(
                weather_time_area, predicted_cordinate, predicted_quantity
            )
            for weather_time_area in weather_timespan_area.data.values()
        ]

    @staticmethod
    def encode_weather_timespan_area_to_tensor(
        weather_timespan_area: data_models.WeatherTimespanArea,
        predicted_cordinate: data_models.GeographicCordinate,
        predicted_quantity: data_models.WeatherQuantity,
    ) -> tf.Tensor:
        """
        Output Shape: (
                n timesteps (ragged),
                query info (shape (2 + n identifier chars (ragged)) at [0][0]) + n points (ragged),
                point info (shape (3,) at [0]) + n entries (ragged)
                value (shape () at [0]) + identifier chars (ragged)
                )
        """
        return cast(
            tf.Tensor,
            cast(
                tf.RaggedTensor,
                tf.ragged.constant(
                    WeatherEncoderV1.encode_weather_timespan_area(
                        weather_timespan_area, predicted_cordinate, predicted_quantity
                    )
                ),
            ).to_tensor(),
        )
