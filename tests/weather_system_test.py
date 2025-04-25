import os
import datetime

import holographic_forecast.weather_system as weather_system
import holographic_forecast.data.data_models as data_models

import tests.log_setup as log_setup

logger = log_setup.get_logger(__name__, "logs/embed-test.log")
os.makedirs("systems", exist_ok=True)


def test_weather_system():
    system = weather_system.WeatherSystem()

    # las vegas cordinate
    las_vegas: data_models.GeographicCordinate = data_models.GeographicCordinate(
        latitude_deg=36.1716, longitude_deg=-115.1391
    )

    system.pull_data_in_circle(
        radius_miles=100,
        center=las_vegas,
        start_date=datetime.date(2000, 1, 1),
        end_date=datetime.date(2010, 1, 1),
        distance_between_points_miles=75,
    )

    system.save("systems/test_system_1")

    logger.info("Done")


if __name__ == "__main__":
    test_weather_system()
