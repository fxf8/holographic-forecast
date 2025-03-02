import sys
import logging
import datetime
import os
import json

from typing import cast

import arcaneforecast.data_collection.data_models as data_models
import arcaneforecast.data_collection.openmeteo_data_collection as odc


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("parse-test.log")],
)


def test_data_model_json_parsing():
    data_cache_filepath: str = "example/open-meteo/parse-test-data-cache.json"
    logger.info(f"Using data cache filepath: {data_cache_filepath}")

    # check if there is a file at the filepath

    if not os.path.exists(data_cache_filepath):
        logger.info(f"{data_cache_filepath} does not exist. Pulling data...")
        center: data_models.GeographicCordinate = data_models.GeographicCordinate(
            latitude_deg=36.1716, longitude_deg=115.1391
        )

        points: list[data_models.GeographicCordinate] = [
            *center.points_within_radius(
                radius_miles=100, distance_between_points_miles=75
            )
        ]

        start_date = datetime.date(2021, 1, 1)
        end_date = datetime.date(2021, 1, 10)

        logger.info(
            f"{center = }\n"
            + "radius_miles=100, distance_between_points_miles=75\n"
            + f"{len(points) = }\n"
            + f"{start_date = } {end_date = }"
        )

        data_collector = odc.OpenMeteoAreaDataCollector.from_points(
            list_of_points=points,
            start_date=start_date,
            end_date=end_date,
            hourly_parameters=odc.OPEN_METEO_HOURLY_PARAMETERS[0:2],
            daily_parameters=odc.OPEN_METEO_DAILY_PARAMETERS[0:2],
        )

        responses = data_collector.get()

        file_contents = [response.json() for response in responses]

        logger.info(f"Saving data to {data_cache_filepath}")

        with open(data_cache_filepath, "w") as file:
            json.dump(file_contents, file)

    logger.info("Parsing data...")

    with open(data_cache_filepath, "r") as file:
        json_responses = cast(list[data_models.OpenMeteoResponseJSON], json.load(file))

        parsed_data = data_models.WeatherSpanArea.from_openmeteo_json(
            json_responses=json_responses,
        )


        logger.info("Done parsing data. Data info:\n" + f"{len(parsed_data.data) = }\n")

        print("hi")


if __name__ == "__main__":
    test_data_model_json_parsing()
