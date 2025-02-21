from collections.abc import Generator
from typing import ClassVar, Self
from dataclasses import dataclass
import math

KM_PER_MILE: float = 1.60934


def drange(start: float, stop: float, step: float = 1.0):
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
            radius_degrees, -radius_degrees, distance_between_points_degrees
        ):
            width: float = math.sqrt(radius_degrees**2 - y_axis_degrees_delta**2)

            for x_axis_degrees_delta in drange(
                -width, width, distance_between_points_degrees
            ):
                yield type(self)(
                    self.latitude_deg + x_axis_degrees_delta,
                    self.longitude_deg + y_axis_degrees_delta,
                )


class A:
    def __init__(self, x: float, y: float):
        self.x: float = x
        self.y: float = y

    def up_one(self) -> Self:
        return type(self)(self.x, self.y + 1)
