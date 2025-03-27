import bisect
import datetime
import enum
import math

from collections.abc import (
    Collection,
    Generator,
    Iterator,
    Mapping,
    MutableSequence,
    Sequence,
)
from typing import ClassVar, Self, cast
from dataclasses import dataclass

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
    name: str


@dataclass(frozen=True)
class WeatherTimePoint:
    """
    Weather data at a certain time and certain location
    """

    time: datetime.datetime
    cordinate: GeographicCordinate
    elevation_meters: float
    data: Collection[tuple[WeatherQuantity, float | str]]


@dataclass
class WeatherCollection:
    """
    Unstructured and flexible collection of weather time points
    """

    data: MutableSequence[WeatherTimePoint]

    def __iter__(self) -> Iterator[WeatherTimePoint]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def import_meteo_json(self, json_response: OpenMeteoResponseJSON):
        elevation_meters: float = cast(float, json_response["elevation"])
        latitude_deg: float = cast(float, json_response["latitude"])
        longitude_deg: float = cast(float, json_response["longitude"])

        hourly_data: OpenMeteoResponseValues = cast(
            OpenMeteoResponseValues, json_response["hourly"]
        )
        daily_data: OpenMeteoResponseValues = cast(
            OpenMeteoResponseValues, json_response["daily"]
        )

        timezone = datetime.timezone(
            name=cast(str, json_response["timezone"]),
            offset=datetime.timedelta(
                seconds=cast(int, json_response["utc_offset_seconds"])
            ),
        )

        hourly_times_iso: Sequence[str] = cast(Sequence[str], hourly_data["time"])
        hourly_quantities: Collection[str] = cast(Collection[str], hourly_data.keys())
        daily_quantities: Collection[str] = cast(Collection[str], daily_data.keys())

        for index in range(len(hourly_times_iso)):
            time: datetime.datetime = datetime.datetime.fromisoformat(
                hourly_times_iso[index]
            ).astimezone(timezone)

            self.data.append(
                WeatherTimePoint(
                    time=time,
                    cordinate=GeographicCordinate(
                        latitude_deg=latitude_deg,
                        longitude_deg=longitude_deg,
                    ),
                    elevation_meters=elevation_meters,
                    data=[
                        *(
                            (
                                WeatherQuantity(hourly_quantity),
                                hourly_data[hourly_quantity][index],
                            )
                            for hourly_quantity in hourly_quantities
                        ),
                        *(
                            (
                                WeatherQuantity(daily_quantity),
                                daily_data[daily_quantity][index // 24],
                            )
                            for daily_quantity in daily_quantities
                        ),
                    ],
                )
            )

    def combine(self, other: "WeatherCollection"):
        for weather_time_point in other:
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
class WeatherSpanArea:
    """
    Weather over a span of time in multiple points (an area). Each index is one hour apart
    """

    data: MutableSequence[WeatherTimeArea]
    start_datetime: datetime.datetime
    end_datetime: datetime.datetime

    def __iter__(self) -> Iterator[WeatherTimeArea]:
        return iter(self.data)

    @classmethod
    def from_weather_collection(cls, weather_collection: WeatherCollection) -> Self:
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
            int((end_datetime - start_datetime).total_seconds()) // 3600
        )

        new_weather_span_area: WeatherSpanArea = cls(
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

    @classmethod
    def from_openmeteo_json(
        cls, json_responses: Sequence[OpenMeteoResponseJSON]
    ) -> Self:
        weather_collection: WeatherCollection = WeatherCollection(data=[])

        for json_response in json_responses:
            weather_collection.import_meteo_json(json_response)

        return cls.from_weather_collection(weather_collection)
