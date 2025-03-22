from typing import Self, cast
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
	# Register methods which can add data to the database

	# Entriy keys are dependent on the latitude, longitude, and degrees per index. Entries within the same 'grid' decided by the degrees per index will be elemnts of a list
	entries: dict[
		int,
		dict[int, data_models.WeatherTimePoint | list[data_models.WeatherTimePoint]],
	]

	# Each index in the entries dictionary is a range of degrees of latitude and longitude. Default is 5 miles
	degrees_per_index: float = (
		data_models.GeographicCordinate.LATITUDE_DEGREES_PER_MILE * 5
	)

	def save(self, filename: str):
		with open(filename, "wb") as f:
			pickle.dump(self, f)

	@staticmethod
	def load(filename: str) -> "WeatherDatabaseOpenMeteo":
		with open(filename, "rb") as f:
			return cast(WeatherDatabaseOpenMeteo, pickle.load(f))

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

	# TODO
	def combine(self, other_database: Self): ...
