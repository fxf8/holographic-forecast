import datetime
import enum
import math
import pickle

from collections.abc import (
    Collection,
    Generator,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from pathlib import Path
from typing import ClassVar, Self, cast, override
from dataclasses import dataclass

import noaa_cdo_api

noaa_responses = noaa_cdo_api.json_responses

KM_PER_MILE: float = 1.60934


def drange(start: float, stop: float, step: float = 1.0) -> Generator[float]:
    while start < stop:
        yield float(start)
        start += step


@dataclass
class GeographicCordinate:
    MILES_PER_LATITUDE_DEGREE: ClassVar[float] = 69.0
    LATITUDE_DEGREES_PER_MILE: ClassVar[float] = 1 / MILES_PER_LATITUDE_DEGREE

    latitude_deg: float  # y axis vertical (relative)
    longitude_deg: float  # y axis horizontal (relative)

    class Direction(enum.Enum):
        North = 1
        South = 2
        East = 3
        West = 4

    def in_direction_degrees(
        self, direction: Direction, degrees: float
    ) -> "GeographicCordinate":
        if direction == GeographicCordinate.Direction.North:
            return GeographicCordinate(
                latitude_deg=self.latitude_deg + degrees,
                longitude_deg=self.longitude_deg,
            )

        if direction == GeographicCordinate.Direction.South:
            return GeographicCordinate(
                latitude_deg=self.latitude_deg - degrees,
                longitude_deg=self.longitude_deg,
            )

        if direction == GeographicCordinate.Direction.East:
            return GeographicCordinate(
                latitude_deg=self.latitude_deg,
                longitude_deg=self.longitude_deg + degrees,
            )

        if direction == GeographicCordinate.Direction.West:
            return GeographicCordinate(
                latitude_deg=self.latitude_deg,
                longitude_deg=self.longitude_deg - degrees,
            )

    def in_direction_miles(
        self, direction: Direction, miles: float
    ) -> "GeographicCordinate":
        return self.in_direction_degrees(
            direction, miles * GeographicCordinate.LATITUDE_DEGREES_PER_MILE
        )

    def points_within_radius_grid(
        self, radius_miles: float, distance_between_points_miles: float
    ) -> Generator[Self]:
        radius_degrees: float = (
            radius_miles / GeographicCordinate.MILES_PER_LATITUDE_DEGREE
        )
        distance_between_points_degrees = (
            distance_between_points_miles
            / GeographicCordinate.MILES_PER_LATITUDE_DEGREE
        )

        for y_axis_degrees_delta in drange(
            -radius_degrees, radius_degrees, distance_between_points_degrees
        ):
            width: float = math.sqrt(radius_degrees**2 - y_axis_degrees_delta**2)

            for x_axis_degrees_delta in drange(
                -width, width, distance_between_points_degrees
            ):
                yield type(self)(
                    self.latitude_deg + x_axis_degrees_delta,
                    self.longitude_deg + y_axis_degrees_delta,
                )

    def points_within_radius_radial_lines(
        self,
        radial_lines_count: int,
        radius_miles: float,
        distance_between_first_point_miles: float,
    ) -> Generator[Self]:
        """
        Calculates points on radial lines. Points on each line progressively become farther still within the radius
        """

        tau: float = math.pi * 2
        angle_step = tau / radial_lines_count

        # Calculating stop
        # distance_between_first_point * (point_number**2) < radius_miles
        # point_number < math.sqrt(radius / distance_between_first_point)

        distances_from_center_miles: list[float] = [
            distance_between_first_point_miles * (point_number**2)
            for point_number in range(
                1,  # start
                int(math.sqrt(radius_miles / distance_between_first_point_miles))
                + 1,  # stop
                1,  # step
            )
        ]

        for distance_from_center_miles in distances_from_center_miles:
            for angle in drange(0, tau - angle_step / 2, angle_step):
                yield type(self)(
                    math.cos(angle)
                    * distance_from_center_miles
                    * GeographicCordinate.LATITUDE_DEGREES_PER_MILE,
                    math.sin(angle)
                    * distance_from_center_miles
                    * GeographicCordinate.LATITUDE_DEGREES_PER_MILE,
                )


# 'hourly_units' key. Maps unit name to unit symbol
OpenMeteoResponseQuantities = Mapping[str, str]
OpenMeteoResponseValues = Mapping[str, Sequence[str | float]]
OpenMeteoResponseJSON = Mapping[
    str, float | int | str | OpenMeteoResponseQuantities | OpenMeteoResponseValues
]


@dataclass(frozen=True)
class WeatherQuantity:
    identifier: str

    # Name does not have as much influence, therefore it is optional
    name: str | None = None

    @override
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, WeatherQuantity) and self.identifier == other.identifier
        )

    @override
    def __hash__(self) -> int:
        return hash(self.identifier)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, WeatherQuantity):
            return NotImplemented

        return self.identifier < other.identifier

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, WeatherQuantity):
            return NotImplemented

        return self.identifier > other.identifier


@dataclass(frozen=True)
class WeatherTimePoint:
    """
    Weather data at a certain time and certain location
    """

    time: datetime.datetime
    cordinate: GeographicCordinate
    data: MutableSequence[tuple[WeatherQuantity, float | str]]


@dataclass
class NOAAWeatherCollection:
    """
    Unstructured and flexible collection of weather time points
    """

    data: MutableSequence[WeatherTimePoint]
    noaa_stations: MutableMapping[str, noaa_cdo_api.json_schemas.StationIDJSON]

    def __iter__(self) -> Iterator[WeatherTimePoint]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def save_file(self, path: Path):
        with path.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load_file(cls, path: Path):
        with path.open("rb") as file:
            return cast(NOAAWeatherCollection, pickle.load(file))

    def import_noaa_json(
        self,
        json_response: noaa_responses.DataJSON
        | list[noaa_cdo_api.json_schemas.DatapointJSON],
        stations_info: Collection[noaa_cdo_api.json_schemas.StationIDJSON],
    ):
        TIME_EPSILON: datetime.timedelta = datetime.timedelta(seconds=10)
        for station in stations_info:
            self.noaa_stations[station["id"]] = station

        results = (
            json_response["results"]
            if isinstance(json_response, dict)
            else json_response
        )

        station_id_to_data: MutableMapping[str, MutableSequence[WeatherTimePoint]] = {}

        for result in results:
            station_of_result = result["station"]

            if station_of_result not in self.noaa_stations:
                raise ValueError(
                    f"{station_of_result} not in stations_info or stored self.noaa_stations"
                )

            index_of_time: int | None = None

            # See if the result time is already in the station_id_to_data

            if station_of_result not in station_id_to_data:
                station_id_to_data[station_of_result] = []

            for index, weather_time_point in enumerate(
                station_id_to_data[station_of_result]
            ):
                lower_bound: datetime.datetime = weather_time_point.time - TIME_EPSILON
                upper_bound: datetime.datetime = weather_time_point.time + TIME_EPSILON

                result_time: datetime.datetime = datetime.datetime.fromisoformat(
                    result["date"]
                )

                if lower_bound <= result_time <= upper_bound:
                    index_of_time = index
                    break

            if index_of_time is None:
                index_of_time = len(station_id_to_data[station_of_result])
                station_id_to_data[station_of_result].append(
                    WeatherTimePoint(
                        time=datetime.datetime.fromisoformat(result["date"]),
                        cordinate=GeographicCordinate(
                            longitude_deg=self.noaa_stations[station_of_result][
                                "longitude"
                            ],
                            latitude_deg=self.noaa_stations[station_of_result][
                                "latitude"
                            ],
                        ),
                        data=[],
                    )
                )

            # each noaa result only has one quantity
            station_id_to_data[station_of_result][index_of_time].data.append(
                (
                    WeatherQuantity(result["datatype"]),
                    float(result["value"]),
                )
            )

            if not any(
                quantity_and_value[0].identifier == "ELEVATION"
                for quantity_and_value in station_id_to_data[station_of_result][
                    index_of_time
                ].data
            ):
                station_id_to_data[station_of_result][index_of_time].data.append(
                    (
                        WeatherQuantity("ELEVATION"),
                        self.noaa_stations[station_of_result]["elevation"],
                    )
                )

        for station in station_id_to_data:
            for weather_time_point in station_id_to_data[station]:
                self.data.append(weather_time_point)

    def combine(self, other: "NOAAWeatherCollection"):
        for weather_time_point in other:
            self.data.append(weather_time_point)

    def add(self, weather_time_point: WeatherTimePoint):
        self.data.append(weather_time_point)


@dataclass
class WeatherTimeArea:
    """
    Weather at a certain time in multiple points (an area)
    """

    data: MutableSequence[WeatherTimePoint]

    def __iter__(self) -> Iterator[WeatherTimePoint]:
        return iter(self.data)


# Create an exception for missing quanitity


@dataclass
class WeatherTimespanArea:
    """
    Weather over a span of time in multiple points (an area). Each index is one hour apart
    """

    data: MutableSequence[WeatherTimeArea]
    start_datetime: datetime.datetime
    end_datetime: datetime.datetime

    def __iter__(self) -> Iterator[WeatherTimeArea]:
        return iter(self.data)

    @classmethod
    def from_weather_collection(cls, weather_collection: NOAAWeatherCollection) -> Self:
        if len(weather_collection) == 0:
            raise ValueError(
                "weather_collection is empty `len(weather_collection) == 0`"
            )

        sample_point: WeatherTimePoint = next(iter(weather_collection))
        start_datetime: datetime.datetime = sample_point.time
        end_datetime: datetime.datetime = sample_point.time

        for weather_time_point in weather_collection:
            if weather_time_point.time < start_datetime:
                start_datetime = weather_time_point.time

            if end_datetime < weather_time_point.time:
                end_datetime = weather_time_point.time

        number_of_timesteps: int = (
            int((end_datetime - start_datetime).total_seconds()) // 3600 + 1
        )

        new_weather_span_area: WeatherTimespanArea = cls(
            data=[WeatherTimeArea(data=[]) for _ in range(number_of_timesteps)],
            start_datetime=start_datetime,
            end_datetime=end_datetime,
        )

        for weather_time_point in weather_collection:
            insertion_index: int = (
                int((weather_time_point.time - start_datetime).total_seconds()) // 3600
            )

            new_weather_span_area.data[insertion_index].data.append(weather_time_point)

        return new_weather_span_area
