from typing import ClassVar
from dataclasses import dataclass

import torch

import holographic_forecast.data.data_models as data_models


@dataclass
class WeatherQuantityEncodingV1:
    def __init__(self, quantity: data_models.WeatherQuantity):
        self.data = torch.tensor(
            [
                ord(letter) - WeatherQuantityEncodingV1.CHAR_MIN_VALUE
                for letter in quantity.identifier
            ],
            dtype=WeatherQuantityEncodingV1.dtype,
        )

    data: torch.Tensor  # shape (n features of int)

    intermediate: torch.Tensor | None = (
        None  # intermediate value for model computations
    )

    dtype: ClassVar[torch.dtype] = torch.int

    CHAR_MIN_VALUE: ClassVar[int] = ord(" ") - ord(" ")  # 0
    CHAR_MAX_VALUE: ClassVar[int] = ord("~") - ord(" ")  # 94


@dataclass
class GeographicCordinateEncodingV1:
    def __init__(
        self,
        cordinate: data_models.GeographicCordinate,
    ):
        self.longitude_latitude = torch.tensor(
            [cordinate.longitude_deg, cordinate.latitude_deg],
            dtype=GeographicCordinateEncodingV1.dtype,
        )

    longitude_latitude: torch.Tensor

    dtype: ClassVar[torch.dtype] = torch.float32


@dataclass
class WeatherEntryEncodingV1:
    def __init__(self, entry: data_models.WeatherEntry):
        self.data = torch.tensor([entry.value], dtype=WeatherEntryEncodingV1.dtype)
        self.weather_quantity = WeatherQuantityEncodingV1(entry.quantity)

    weather_quantity: WeatherQuantityEncodingV1
    data: torch.Tensor  # shape ()

    dtype: ClassVar[torch.dtype] = torch.float32


@dataclass
class WeatherTimePointEncodingV1:
    def __init__(self, weather_time_point: data_models.WeatherTimePoint):
        self.timestamp = torch.tensor(
            weather_time_point.time.timestamp(),
            dtype=WeatherTimePointEncodingV1.dtype,
        )
        self.cordinate = GeographicCordinateEncodingV1(weather_time_point.cordinate)
        self.weather_entries = [
            WeatherEntryEncodingV1(entry) for entry in weather_time_point.data
        ]

    timestamp: torch.Tensor  # shape ()
    cordinate: GeographicCordinateEncodingV1
    weather_entries: list[WeatherEntryEncodingV1]

    intermediate: torch.Tensor | None = (
        None  # intermediate value for model computations
    )

    dtype: ClassVar[torch.dtype] = torch.float32


@dataclass
class WeatherTimeAreaEncodingV1:
    def __init__(self, weather_time_area: data_models.WeatherTimeArea):
        self.weather_time_points = [
            WeatherTimePointEncodingV1(weather_time_point)
            for weather_time_point in weather_time_area.data
        ]

    weather_time_points: list[WeatherTimePointEncodingV1]

    intermediate: torch.Tensor | None = (
        None  # intermediate value for model computations
    )

    dtype: ClassVar[torch.dtype] = torch.float32


@dataclass
class WeatherTimespanAreaEncodingV1:
    def __init__(self, weather_timespan_area: data_models.WeatherTimespanArea):
        self.weather_time_areas = [
            WeatherTimeAreaEncodingV1(weather_time_area)
            for weather_time_area in weather_timespan_area.data.values()
        ]

    weather_time_areas: list[WeatherTimeAreaEncodingV1]

    dtype: ClassVar[torch.dtype] = torch.float32


@dataclass
class QueryInfoEncodingV1:
    weather_quantity: WeatherQuantityEncodingV1
    cordinate: GeographicCordinateEncodingV1
    timestamp: torch.Tensor  # shape ()

    def __init__(
        self,
        weather_quantity: data_models.WeatherQuantity,
        cordinate: data_models.GeographicCordinate,
        timestamp: float,
    ):
        self.weather_quantity = WeatherQuantityEncodingV1(weather_quantity)
        self.cordinate = GeographicCordinateEncodingV1(cordinate)
        self.timestamp = torch.tensor(timestamp, dtype=QueryInfoEncodingV1.dtype)

    dtype: ClassVar[torch.dtype] = torch.float32


@dataclass
class QueryEncodingV1:
    weather_timespan_area: WeatherTimespanAreaEncodingV1
    query_info: QueryInfoEncodingV1
