# pyright: reportAny=false

import pathlib
import pickle
from dataclasses import dataclass

import torch

import holographic_forecast.data.data_models as data_models
import holographic_forecast.data.data_encoding_torch as data_encoding
import holographic_forecast.weather_models_torch as torch_models


@dataclass
class WeatherForecasterTorch:
    weather_timespan_area: data_models.WeatherTimespanArea
    weather_model: torch_models.WeatherModelV1

    progress_path: pathlib.Path | None = None
    loaded: bool = False

    def load(self, path: pathlib.Path | None = None):
        if path is None:
            path = self.progress_path

        if path is None:
            raise ValueError(
                "No path provided in either self.save_path or argument path"
            )

        with path.open("rb") as file:
            self.loaded = True
            self.progress_path = path
            self.weather_timespan_area = pickle.load(file)
            self.weather_model = torch_models.WeatherModelV1()
            _ = self.weather_model.load_state_dict(torch.load(path.with_suffix(".pt")))

    def save(self, path: pathlib.Path | None = None):
        if path is None:
            path = self.progress_path

        if path is None:
            raise ValueError(
                "No path provided in either self.save_path or argument path"
            )

        with path.open("wb") as file:
            pickle.dump(self.weather_timespan_area, file)
            torch.save(self.weather_model.state_dict(), path.with_suffix(".pt"))

    def forecast(
        self,
        cordinate: data_models.GeographicCordinate,
        timestamp: float,
        quantity: data_models.WeatherQuantity,
    ) -> torch.Tensor:
        query_info: data_encoding.QueryInfoEncodingV1 = (
            data_encoding.QueryInfoEncodingV1(
                weather_quantity=quantity, cordinate=cordinate, timestamp=timestamp
            )
        )

        query: data_encoding.QueryEncodingV1 = data_encoding.QueryEncodingV1(
            query_info=query_info,
            weather_timespan_area=data_encoding.WeatherTimespanAreaEncodingV1(
                self.weather_timespan_area
            ),
        )

        return self.weather_model(query)
