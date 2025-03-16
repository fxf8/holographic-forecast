import datetime
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
    latitude_deg: float  # x axis (relative)
    longitude_deg: float  # y axis (relative)

    def points_within_radius(
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


OpenMeteoResponseQuantities = Mapping[str, str]
OpenMeteoResponseValues = Mapping[str, Sequence[str | float]]
OpenMeteoResponseJSON = Mapping[
    str, float | int | str | OpenMeteoResponseQuantities | OpenMeteoResponseValues
]


@dataclass(frozen=True)
class WeatherQuantity:
    name: str


@dataclass
class WeatherTimePoint:
    """
    Weather data at a certain time and certain location
    """

    time: datetime.datetime
    cordinate: GeographicCordinate
    elevation_meters: float
    data: Collection[tuple[WeatherQuantity, float | str]]


@dataclass
class WeatherTimeArea:
    """
    Weather at a certain time in multiple points (an area)
    """

    data: MutableSequence[WeatherTimePoint]

    def __iter__(self) -> Iterator[WeatherTimePoint]:
        return iter(self.data)

    def geographic_points(self) -> Generator[GeographicCordinate]:
        return (weather_time_point.cordinate for weather_time_point in self.data)


# Create an exception for missing quanitity


class MissingQuantityException(Exception):
    def __init__(self, message: str) -> None:
        self.message: str = message
        super().__init__(message)


@dataclass
class WeatherSpanArea:
    """
    Weather over a span of time in multiple points (an area)
    """

    data: Sequence[WeatherTimeArea]

    def __iter__(self) -> Iterator[WeatherTimeArea]:
        return iter(self.data)

    @classmethod
    def from_openmeteo_json(
        cls,
        json_responses: Collection[OpenMeteoResponseJSON],
        expected_weather_quantities: Sequence[WeatherQuantity] | None = None,
    ) -> Self:
        """
        Creates a WeatherSpanArea from a JSON response from OpenMeteo

        Args:
            json_responses (Collection[Any]): A Collection of JSON responses from OpenMeteo, each from a different location
            expected_weather_quantities (Sequence[WeatherQuantity] | None): Set of quantities to *EXPECT* and extract from the json response. Can be used to remove unintended quantities, specify order for quantities, and error for expected quanitities not provided. Defaults to None which returns all quantities in the response in lexicographical order.

        Returns:
            WeatherSpanArea: WeatherSpanArea
        """

        if len(json_responses) == 0:
            return cls([])

        example_response: OpenMeteoResponseJSON = next(iter(json_responses))

        # Quantities only referes to the *NAME* of the quantity, not the value
        hourly_quantities_in_response: Collection[WeatherQuantity] = [
            WeatherQuantity(quantity_name)
            for quantity_name in cast(
                OpenMeteoResponseQuantities, example_response["hourly_units"]
            ).keys()
        ]

        daily_quantities_in_response: Collection[WeatherQuantity] = [
            WeatherQuantity(quantity_name)
            for quantity_name in cast(
                OpenMeteoResponseQuantities, example_response["daily_units"]
            ).keys()
        ]

        quantities_in_response: Collection[WeatherQuantity] = (
            hourly_quantities_in_response + daily_quantities_in_response
        )

        if expected_weather_quantities is None:
            expected_weather_quantities = [
                *sorted(quantities_in_response, key=lambda quantity: quantity.name)
            ]

        elif not set(expected_weather_quantities).issubset(set(quantities_in_response)):
            raise MissingQuantityException(
                f"Missing quantities in response: {set(expected_weather_quantities) - set(quantities_in_response)},"
                + f"Expected: {set(expected_weather_quantities)}, Received: {set(quantities_in_response)}"
            )

        hourly_times: Sequence[str] = cast(
            Sequence[str],
            cast(OpenMeteoResponseValues, example_response["daily"])["time"],
        )

        # Currently empty data. Filled later
        weather_span_area: WeatherSpanArea = cls(
            data=[WeatherTimeArea([]) for _ in range(len(hourly_times))]
        )

        for json_response in json_responses:
            latitude: float = cast(float, json_response["latitude"])
            longitude: float = cast(float, json_response["longitude"])

            timezone = datetime.timezone(
                name=cast(str, json_response["timezone"]),
                offset=datetime.timedelta(
                    seconds=cast(int, json_response["utc_offset_seconds"])
                ),
            )

            elevation: float = cast(float, json_response["elevation"])

            hourly_data: OpenMeteoResponseValues = cast(
                OpenMeteoResponseValues, json_response["hourly"]
            )

            daily_data: OpenMeteoResponseValues = cast(
                OpenMeteoResponseValues, json_response["daily"]
            )

            for time_index, hourly_time in enumerate(hourly_times):
                weather_span_area.data[time_index].data.append(
                    WeatherTimePoint(
                        time=datetime.datetime.fromisoformat(hourly_time).astimezone(
                            timezone
                        ),
                        cordinate=GeographicCordinate(
                            latitude_deg=latitude, longitude_deg=longitude
                        ),
                        data=[
                            (
                                expected_weather_quantity,
                                hourly_data[expected_weather_quantity.name][time_index]
                                if expected_weather_quantity.name in hourly_data
                                else daily_data[expected_weather_quantity.name][
                                    time_index // 24
                                ],
                            )
                            for expected_weather_quantity in expected_weather_quantities
                        ],
                        elevation_meters=elevation,
                    )
                )

        return weather_span_area

    def geographic_points(self) -> Generator[GeographicCordinate] | None:
        """
        Returns None of the WeatherSpanArea is empty, otherwise returns a generator of all the geographic points in the WeatherSpanArea
        """

        if len(self.data) == 0:
            return None

        return self.data[0].geographic_points()
