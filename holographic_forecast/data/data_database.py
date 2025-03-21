from dataclasses import dataclass
import pickle

import holographic_forecast.data.data_models as data_models

# All the different kinds of data objects
# WeatherTimePoint
# WeatherTimeArea
# WeatherSpanArea


# Database for weather data from OpenMeteo
@dataclass
class WeatherDatabaseOpenMeteo:
    ...

    # Register methods which can add data to the database

    # TODO
    def register_weather_time_point(
        self, weather_time_point: data_models.WeatherTimePoint
    ): ...

    # TODO
    def register_weather_time_area(
        self, weather_time_area: data_models.WeatherTimeArea
    ): ...

    # TODO
    def register_weather_span_area(
        self, weather_span_area: data_models.WeatherSpanArea
    ): ...
