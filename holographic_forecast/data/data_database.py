# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false

from typing import Self, cast, ClassVar
from collections.abc import Generator, Sequence
from dataclasses import dataclass
import pickle
import bisect

import intervaltree

import holographic_forecast.data.data_models as data_models

# All the different kinds of data objects
# WeatherTimePoint
# WeatherTimeArea
# WeatherSpanArea


def first_element_[T](sequence: Sequence[T]) -> T:
	return sequence[0]


# Database for weather data from OpenMeteo
@dataclass
class WeatherDatabaseOpenMeteo:
	# Used for intervals to avoid floating point inaccuracy errors.
	# Epsilon is subtracted from the lower bounds and Epsilon / 2 is subtracted from the upper bounds
	EPSILON: ClassVar[float] = 1e-2

	# Entriy keys are dependent on the latitude, longitude, and degrees per index.
	# Entries within the same 'grid' decided by the degrees per index will be elemnts of a list.
	# The 'interval' element of the tuple represents the time interval. The time is represented
	# using hours since epoch

	entry_grid_squares: list[
		tuple[
			int,
			list[
				tuple[
					int,
					tuple[
						list[data_models.WeatherTimePoint], intervaltree.IntervalTree
					],
				]
			],
		]
	]

	def __post_init__(self):
		self.entry_grid_squares = []

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

		# Ensure that the entry grid square exists
		if not (
			entry_grid_latitude_index := bisect.bisect_left(
				self.entry_grid_squares, index_latitude, key=first_element_
			)
		):
			self.entry_grid_squares.insert(
				entry_grid_latitude_index, (index_latitude, [])
			)

		if not (
			entry_grid_longitude_index := bisect.bisect_left(
				self.entry_grid_squares[index_latitude][1],
				index_longitude,
				key=first_element_,
			)
		):
			self.entry_grid_squares[index_latitude][1].insert(
				entry_grid_longitude_index,
				(index_longitude, ([], intervaltree.IntervalTree())),
			)

		entries_list, interval_tree = self.entry_grid_squares[index_latitude][1][
			entry_grid_longitude_index
		][1]

		if not overwrite and unix_datetime_hours in interval_tree:
			return

		interval_tree.addi(
			unix_datetime_hours - self.EPSILON,
			unix_datetime_hours + self.EPSILON,
			weather_time_point,
		)
		bisect.insort(
			entries_list,
			weather_time_point,
			key=lambda weather_time_point: weather_time_point.time,
		)

		# Make sure to insert at the chronologically correct time in the correct entries list

		# The 'interval' element of the tuple represents the time interval which exists in the database.
		# The time is represented using hours since epoch

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

	def __iter__(self) -> Generator[data_models.WeatherTimePoint]:
		for entry_grid_square in self.entry_grid_squares:
			for entry_grid_longitude_index in entry_grid_square[1]:
				for weather_time_point in entry_grid_longitude_index[1][0]:
					yield weather_time_point

	def combine(self, other_database: Self):
		for weather_time_point in other_database:
			self.register_weather_time_point(weather_time_point)
