# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false

from typing import Self, cast, ClassVar
from collections.abc import Generator, Sequence
from dataclasses import dataclass
import pickle
import bisect
import datetime

import intervaltree

import holographic_forecast.data.data_models as data_models
import holographic_forecast.data.openmeteo_data_collection as openmeteo

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

	"""
	Entriy keys are dependent on the latitude, longitude, and degrees per index.
	Entries within the same 'grid' decided by the degrees per index will be elemnts of a list.
	The 'interval' element of the tuple represents the time interval. The time is represented
	using hours since epoch
	"""

	"""
	IMPORTANT NOTES:
		- 'latitude_index' and 'longitude_index' are the indexes of the latitude and longitude representing latitude and longitude segments
		- 'entry_array_latitude_index' and 'entry_array_longitude_index' are the indexes for the literal array
	"""


	entry_grid_squares: list[
		tuple[
			int,  # latitude index
			list[  # latitude entries
				tuple[
					int,  # longitude index
					tuple[  # longitude entries
						list[data_models.WeatherTimePoint], intervaltree.IntervalTree
					],
				]
			],
		]
	]

	def __post_init__(self):
		self.entry_grid_squares = []

	def __iter__(self) -> Generator[data_models.WeatherTimePoint]:
		for entry_grid_square in self.entry_grid_squares:
			for entry_grid_longitude_index in entry_grid_square[1]:
				for weather_time_point in entry_grid_longitude_index[1][0]:
					yield weather_time_point

	def latitude_index_to_array_index(self, latitude_index: int) -> int | None:
		index = bisect.bisect_left(
			self.entry_grid_squares, latitude_index, key=first_element_
		)

		if (
			index != len(self.entry_grid_squares)
			and self.entry_grid_squares[index][0] == latitude_index
		):
			return index

		return None

	def cordinate_indices_to_array_index(
		self, latitude_index: int, longitude_index: int
	) -> int | None:
		latitude_array_index: int | None = self.latitude_index_to_array_index(
			latitude_index
		)
		if latitude_array_index is None:
			return None

		index = bisect.bisect_left(
			self.entry_grid_squares[latitude_array_index][1],
			longitude_index,
			key=lambda entry: entry[0],
		)

		if (
			index != len(self.entry_grid_squares[latitude_array_index][1])
			and self.entry_grid_squares[latitude_array_index][1][index][0]
			== longitude_index
		):
			return index

		return None

	def insert_

	@staticmethod
	def datetime_to_unix_hours(datetime: datetime.datetime) -> float:
		return datetime.timestamp() // 3600

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

		unix_datetime_hours: float = WeatherDatabaseOpenMeteo.datetime_to_unix_hours(
			weather_time_point.time
		)

		# Ensure that the entry grid square exists
		if not (
			entry_array_latitude_index := self.cordinate_indices_to_array_index(
				index_latitude, index_longitude
			)
		):
			self.entry_grid_squares.append((index_latitude, []))

		if not (
			entry_array_longitude_index := bisect.bisect_left(
				self.entry_grid_squares[index_latitude][1],
				index_longitude,
				key=first_element_,
			)
		):
			self.entry_grid_squares[index_latitude][1].insert(
				entry_array_longitude_index,
				(index_longitude, ([], intervaltree.IntervalTree())),
			)

		entries_list, interval_tree = self.entry_grid_squares[
			entry_array_latitude_index
		][1][entry_array_longitude_index][1]

		if not overwrite and unix_datetime_hours in interval_tree:
			return

		interval_tree.addi(
			unix_datetime_hours - self.EPSILON,
			unix_datetime_hours + self.EPSILON,
			weather_time_point,
		)

		interval_tree.merge_overlaps()

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

	def combine(self, other_database: Self):
		for weather_time_point in other_database:
			self.register_weather_time_point(weather_time_point)

	# Sometimes, points will have missing data over a priod of time. The function 'pull_data' will pull that data
	# from openmeteo and add it to the database so that each point within the entry grid square exists for a continuous period of time.
	# This can also be used to collect training data
	def pull_data(
		self,
		entry_grid_latitude_index: int,
		entry_grid_longitude_index: int,
		start_datetime: datetime.datetime | None = None,
		end_datetime: datetime.datetime | None = None,
		overwrite: bool = False,
	) -> bool:
		collector: openmeteo.OpenMeteoAreaDataCollector = (
			openmeteo.OpenMeteoAreaDataCollector()
		)
