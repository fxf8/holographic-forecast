# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false

from typing import Self, cast
from collections.abc import (
    Callable,
    Generator,
    Sequence,
    Collection,
    ValuesView,
)
from dataclasses import dataclass
import pickle
import datetime
import math

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

    # Dimensionality: (latitude: float, longitude: float, time: datetime.datetime)
    entry_interval_dict: P.IntervalDict[
        P.IntervalDict[P.IntervalDict[data_models.WeatherTimePoint]]
    ]

    def __iter__(self) -> Generator[data_models.WeatherTimePoint]:
        for latitude_line in self.entry_interval_dict.values():
            for weather_point_span in latitude_line.values():
                yield from weather_point_span.values()

    # Each index in the entries dictionary is a range of degrees of latitude and longitude. Default is 5 miles

    def save(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str) -> "WeatherDatabaseOpenMeteo":
        with open(filename, "rb") as f:
            return cast(WeatherDatabaseOpenMeteo, pickle.load(f))

    # Register methods which can add data to the database

    def register_weather_time_point(
        self, weather_time_point: data_models.WeatherTimePoint
    ):
        latitude_point: float = weather_time_point.cordinate.latitude_deg
        longitude_point: float = weather_time_point.cordinate.longitude_deg

        self.entry_interval_dict[latitude_point][longitude_point][
            weather_time_point.time
        ] = weather_time_point

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

    @dataclass
    class PointSelector:
        # Latitude bounds
        latitude_interval_bounds_deg: P.Interval

        # Takes a latitude line and returns a longitude interval
        longitude_interval_bounds_deg: Callable[[float], P.Interval]

        @classmethod
        def circle(
            cls, center: data_models.GeographicCordinate, radius_miles: float
        ) -> Self:
            # Algebra:
            # (center.latitude - latitude)^2 + (center.longitude - longitude)^2 = radius
            # (center.longitude - longitude)^2 = radius - (center.latitude - latitude)^2
            # center.longitude - longitude = +-sqrt(radius - (center.latitude - latitude)^2)
            # longitude = center.longitude +-sqrt(radius - (center.latitude - latitude)^2)

            # Radius in measure of longitude/latitude degrees
            radius_deg: float = (
                radius_miles * data_models.GeographicCordinate.LATITUDE_DEGREES_PER_MILE
            )

            return cls(
                latitude_interval_bounds_deg=P.Interval(
                    center.in_direction_miles(
                        data_models.GeographicCordinate.Direction.North, radius_miles
                    ).latitude_deg,
                    center.in_direction_miles(
                        data_models.GeographicCordinate.Direction.South, radius_miles
                    ).latitude_deg,
                ),
                longitude_interval_bounds_deg=lambda latitude_deg: P.Interval(
                    center.longitude_deg
                    - math.sqrt(radius_deg - (center.latitude_deg - latitude_deg) ** 2),
                    center.longitude_deg
                    + math.sqrt(radius_deg - (center.latitude_deg - latitude_deg) ** 2),
                ),
            )

    # Gets data from the database rather than pulling from openmeteo
    def database_get_weather_span_area(
        self,
        start_datetime: datetime.datetime,
        end_datetime: datetime.datetime,
        point_selector: PointSelector,
    ) -> data_models.WeatherSpanArea:
        weather_collection: data_models.WeatherCollection = (
            data_models.WeatherCollection(data=[])
        )

        time_interval: P.Interval = P.Interval(start_datetime, end_datetime)

        latitude_interval_dict_selection: P.IntervalDict[
            P.IntervalDict[P.IntervalDict[data_models.WeatherTimePoint]]
        ] = self.entry_interval_dict[point_selector.latitude_interval_bounds_deg]

        for latitude_interval_dict in latitude_interval_dict_selection.values():
            if latitude_interval_dict.domain().lower is None:
                continue

            longitude_interval_dict_selection: P.IntervalDict[
                P.IntervalDict[data_models.WeatherTimePoint]
            ] = latitude_interval_dict[
                point_selector.longitude_interval_bounds_deg(
                    cast(float, latitude_interval_dict.domain().lower)
                )
            ]

            for time_interval_dict in longitude_interval_dict_selection.values():
                for weather_time_point in time_interval_dict[time_interval].values():
                    weather_collection.add(weather_time_point)

        return data_models.WeatherSpanArea.from_weather_collection(weather_collection)
