import datetime
from dataclasses import dataclass
from typing import Any
import requests

import arcaneforecast.data_collection.data_models as data_models

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
        parameters: dict[str, str] = {
            "latitude": f"{self.position.latitude_deg:.10f}",
            "longitude": f"{self.position.latitude_deg:.10f}",
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "timezone": self.timezone.tzname(None),
            "hourly": ",".join(
                map(lambda parameter: parameter.name, self.hourly_parameters)
            ),
            "daily": ",".join(
                map(lambda parameter: parameter.name, self.hourly_parameters)
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
