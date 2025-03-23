# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false

from typing import Self, cast, ClassVar
from collections.abc import Generator, Sequence, Collection
from dataclasses import dataclass
import pickle
import bisect
import datetime

import portion as P

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
						# Interval represents datetimes which exist
						list[data_models.WeatherTimePoint], P.Interval
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

	def _latitude_index_to_array_index(self, latitude_index: int) -> int | None:
		index = bisect.bisect_left(
			self.entry_grid_squares, latitude_index, key=first_element_
		)

		if (
			index != len(self.entry_grid_squares)
			and self.entry_grid_squares[index][0] == latitude_index
		):
			return index

		return None

	def _cordinate_indices_to_array_indices(
		self, latitude_index: int, longitude_index: int
	) -> int | None:
		latitude_array_index: int | None = self._latitude_index_to_array_index(
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

		# Ensure that the entry grid square exists
		if not (
			entry_array_latitude_index := self._cordinate_indices_to_array_indices(
				index_latitude, index_longitude
			)
		):
			self.entry_grid_squares.append((index_latitude, []))
			entry_array_latitude_index = 0

		if not (
			entry_array_longitude_index := bisect.bisect_left(
				self.entry_grid_squares[index_latitude][1],
				index_longitude,
				key=first_element_,
			)
		):
			self.entry_grid_squares[index_latitude][1].insert(
				entry_array_longitude_index,
				(index_longitude, ([], P.empty())),
			)

		entries_list, existing_interval = self.entry_grid_squares[
			entry_array_latitude_index
		][1][entry_array_longitude_index][1]

		if not overwrite and weather_time_point.time in existing_interval:
			return

		existing_interval |= P.closedopen(
			weather_time_point.time,
			weather_time_point.time + datetime.timedelta(hours=1),
		)

		bisect.insort(
			entries_list,
			weather_time_point,
			key=lambda weather_time_point: weather_time_point.time,
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
		for weather_time_point in other_database:
			self.register_weather_time_point(weather_time_point)

	# The function 'pull_data' will pull that data from openmeteo and add
	# it to the database so that each point within the entry grid square exists for a
	# continuous period of time.
	def pull_data(
		self,
		cordinates: data_models.GeographicCordinate
		| Collection[data_models.GeographicCordinate],
		start_datetime: datetime.datetime,
		end_datetime: datetime.datetime,
		hourly_parameters: Sequence[data_models.WeatherQuantity] | None = None,
		daily_parameters: Sequence[data_models.WeatherQuantity] | None = None,
	):
		if hourly_parameters is None:
			hourly_parameters = openmeteo.OPEN_METEO_HOURLY_PARAMETERS

		if daily_parameters is None:
			daily_parameters = openmeteo.OPEN_METEO_DAILY_PARAMETERS

		collector: openmeteo.OpenMeteoAreaSpanDataCollector = (
			openmeteo.OpenMeteoAreaSpanDataCollector.from_points(
				list_of_points=cordinates,
				start_date=start_datetime,
				end_date=end_datetime,
				hourly_parameters=hourly_parameters,
				daily_parameters=daily_parameters,
			)
		)

		self.register_weather_span_area(collector.get())
