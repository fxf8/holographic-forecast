import datetime
from dataclasses import dataclass
from collections.abc import Collection, Mapping

import requests

import holographic_forecast.data.data_models as data_models

OPEN_METEO_HISTORICAL_ENDPOINT: str = "https://archive-api.open-meteo.com/v1/archive"


@dataclass
class OpenMeteoPointDataCollector:
    """
    Requester for OpenMeteo historical data at a single point. Meant to be used by OpenMeteoAreaDataCollector which uses a requests.Session
    """

    position: data_models.GeographicCordinate

    # Openmeteo requests iso8601 date format
    start_date: datetime.date
    end_date: datetime.date

    timezone: datetime.timezone
    hourly_parameters: list[data_models.WeatherQuantity]
    daily_parameters: list[data_models.WeatherQuantity]

    def prepare_request(self) -> requests.PreparedRequest:
        parameters: Mapping[str, str] = {
            "latitude": f"{self.position.latitude_deg:.10f}",
            "longitude": f"{self.position.latitude_deg:.10f}",
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "timezone": self.timezone.tzname(None),
            "hourly": ",".join(
                map(lambda parameter: parameter.name, self.hourly_parameters)
            ),
            "daily": ",".join(
                map(lambda parameter: parameter.name, self.daily_parameters)
            ),
        }

        return requests.Request(
            "GET", OPEN_METEO_HISTORICAL_ENDPOINT, params=parameters
        ).prepare()

    def get(self, requests_session: requests.Session) -> requests.Response:
        return requests_session.send(self.prepare_request())


@dataclass
class OpenMeteoAreaDataCollector:
    points: list[OpenMeteoPointDataCollector]

    def get(self) -> list[requests.Response]:
        with requests.Session() as session:
            return [point.get(session) for point in self.points]

    @classmethod
    def from_points(
        cls,
        list_of_points: list[data_models.GeographicCordinate],
        start_date: datetime.date,
        end_date: datetime.date,
        hourly_parameters: list[data_models.WeatherQuantity],
        daily_parameters: list[data_models.WeatherQuantity],
        timezone: datetime.timezone = datetime.timezone.utc,
    ) -> "OpenMeteoAreaDataCollector":
        return cls(
            [
                OpenMeteoPointDataCollector(
                    point,
                    start_date,
                    end_date,
                    timezone,
                    hourly_parameters,
                    daily_parameters,
                )
                for point in list_of_points
            ]
        )


OPEN_METEO_HOURLY_PARAMETERS: list[data_models.WeatherQuantity] = [
    data_models.WeatherQuantity(name)
    for name in [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "apparent_temperature",
        "precipitation",
        "rain",
        "snowfall",
        "snow_depth",
        "weather_code",
        "pressure_msl",
        "surface_pressure",
        "cloud_cover",
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        "et0_fao_evapotranspiration",
        "vapour_pressure_deficit",
        "wind_speed_10m",
        "wind_speed_100m",
        "wind_direction_10m",
        "wind_direction_100m",
        "wind_gusts_10m",
        "soil_temperature_0_to_7cm",
        "soil_temperature_7_to_28cm",
        "soil_temperature_28_to_100cm",
        "soil_temperature_100_to_255cm",
        "soil_moisture_0_to_7cm",
        "soil_moisture_7_to_28cm",
        "soil_moisture_28_to_100cm",
        "soil_moisture_100_to_255cm",
        "boundary_layer_height",
        "wet_bulb_temperature_2m",
        "total_column_integrated_water_vapour",
        "is_day",
        "sunshine_duration",
        "albedo",
        "snow_depth_water_equivalent",
        "shortwave_radiation_instant",
        "direct_radiation_instant",
        "diffuse_radiation_instant",
        "direct_normal_irradiance_instant",
        "global_tilted_irradiance_instant",
        "terrestrial_radiation_instant",
    ]
]

OPEN_METEO_DAILY_PARAMETERS: list[data_models.WeatherQuantity] = [
    data_models.WeatherQuantity(name)
    for name in [
        "weather_code",
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "apparent_temperature_max",
        "apparent_temperature_min",
        "apparent_temperature_mean",
        "sunrise",
        "sunset",
        "daylight_duration",
        "sunshine_duration",
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        "precipitation_hours",
        "wind_speed_10m_max",
        "wind_gusts_10m_max",
        "wind_direction_10m_dominant",
        "shortwave_radiation_sum",
        "et0_fao_evapotranspiration",
    ]
]
