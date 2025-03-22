# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false

from typing import Self, cast, ClassVar
from dataclasses import dataclass
import pickle
import bisect

import intervaltree

import holographic_forecast.data.data_models as data_models

# All the different kinds of data objects
# WeatherTimePoint
# WeatherTimeArea
# WeatherSpanArea


# Database for weather data from OpenMeteo
@dataclass
class WeatherDatabaseOpenMeteo:
	# Used for intervals to avoid floating point inaccuracy errors
	EPSILON: ClassVar[float] = 1e-2

	# Entriy keys are dependent on the latitude, longitude, and degrees per index.
	# Entries within the same 'grid' decided by the degrees per index will be elemnts of a list.
	# The 'interval' element of the tuple represents the time interval. The time is represented
	# using hours since epoch
	entry_grid_squares: dict[
		int,
		dict[
			int,
			tuple[list[data_models.WeatherTimePoint], intervaltree.IntervalTree],
		],
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

	# Register methods which can add data to the database

	def register_weather_time_point(
		self, weather_time_point: data_models.WeatherTimePoint, overwrite: bool = True
	):
		index_latitude: int = int(
			weather_time_point.cordinate.latitude_deg / self.degrees_per_index
		)

		index_longitude: int = int(
			weather_time_point.cordinate.longitude_deg / self.degrees_per_index
		)

		unix_datetime_seconds: float = weather_time_point.time.timestamp()
		unix_datetime_hours: float = unix_datetime_seconds // 3600

		if not isinstance(self.entry_grid_squares.get(index_latitude), dict):
			self.entry_grid_squares[index_latitude] = {}
			self.entry_grid_squares[index_latitude][index_longitude] = (
				[weather_time_point],
				intervaltree.IntervalTree(
					intervaltree.Interval(
						unix_datetime_hours - self.EPSILON,
						unix_datetime_hours + self.EPSILON,
					)
				),
			)

			return

		if not isinstance(
			self.entry_grid_squares[index_latitude].get(index_longitude), tuple
		):
			self.entry_grid_squares[index_latitude][index_longitude] = (
				[weather_time_point],
				intervaltree.IntervalTree(
					intervaltree.Interval(
						unix_datetime_hours - self.EPSILON,
						unix_datetime_hours + self.EPSILON,
					)
				),
			)

			return

		# Make sure to insert at the chronologically correct time in the entries list

		entry_list, interval_tree = self.entry_grid_squares[index_latitude][
			index_longitude
		]

		# pyright ignore partially unknown
		entry_exists: bool = unix_datetime_hours in interval_tree

		if entry_exists and not overwrite:
			return

		if not entry_exists:
			interval_tree.addi(
				unix_datetime_hours - self.EPSILON,
				unix_datetime_hours + self.EPSILON,
			)

		bisect.insort(
			entry_list,
			weather_time_point,
			key=lambda weather_time_point_: weather_time_point_.time,
		)

	def register_weather_time_area(
		self, weather_time_area: data_models.WeatherTimeArea
	):
		for weather_time_point in weather_time_area:
			self.register_weather_time_point(weather_time_point)

	def register_weather_span_area(
		self, weather_span_area: data_models.WeatherSpanArea
	):
		for weather_time_area in weather_span_area:
			self.register_weather_time_area(weather_time_area)

	def register(
		self,
		data: data_models.WeatherTimePoint
		| data_models.WeatherTimeArea
		| data_models.WeatherSpanArea,
	):
		match data:
			case data_models.WeatherTimePoint():
				self.register_weather_time_point(data)
			case data_models.WeatherTimeArea():
				self.register_weather_time_area(data)
			case data_models.WeatherSpanArea():
				self.register_weather_span_area(data)

	def combine(self, other_database: Self):
		for _, latitude_entries in other_database.entry_grid_squares.items():
			for _, longitude_entries in latitude_entries.items():
				for weather_time_points in longitude_entries[0]:
					self.register_weather_time_point(weather_time_points)
