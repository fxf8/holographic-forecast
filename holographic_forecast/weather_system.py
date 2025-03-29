# pyright: reportMissingTypeStubs=false

from collections.abc import Collection, Sequence
import datetime

import keras

import holographic_forecast.data.data_database as weather_db
import holographic_forecast.data.data_embedding as weather_embedding
import holographic_forecast.data.data_models as data_models
import holographic_forecast.weather_models_tf as weather_models


class WeatherSystem:
    database: weather_db.WeatherDatabaseOpenMeteo
    model: weather_models.WeatherModelV1  # Input shape: (batch_size (a), timesteps (ragged) (b), n_points (ragged) (c), n_features (d))

    def __init__(self):
        self.database = weather_db.WeatherDatabaseOpenMeteo()
        self.model = weather_models.WeatherModelV1(100)

    def save(self, filename: str):
        self.database.save(f"{filename}.pkl")
        self.model.save(f"{filename}.keras")

    def load(self, filename: str):
        self.database = weather_db.WeatherDatabaseOpenMeteo.load(filename)
        self.model = keras.models.load_model(filename)

    # Takes cordinates and time range
    def pull_data(
        self,
        cordinates: Collection[data_models.GeographicCordinate],
        start_date: datetime.date,
        end_date: datetime.date,
        hourly_parameters: Sequence[data_models.WeatherQuantity] | None = None,
        daily_parameters: Sequence[data_models.WeatherQuantity] | None = None,
    ):
        self.database.pull_data(
            cordinates,
            start_date,
            end_date,
            hourly_parameters,
            daily_parameters,
        )

    # Pull random data within a radius_miles, timespan, and density_per_square_mile_percentage
    def pull_data_in_circle(
        self,
        radius_miles: float,
        center: data_models.GeographicCordinate,
        start_date: datetime.date,
        end_date: datetime.date,
        distance_between_points_miles: float,
    ):
        queried_cordinates: list[data_models.GeographicCordinate] = [
            *center.points_within_radius_grid(
                radius_miles, distance_between_points_miles
            )
        ]

        self.pull_data(
            queried_cordinates,
            start_date,
            end_date,
        )
