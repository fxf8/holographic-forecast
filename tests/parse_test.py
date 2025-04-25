from collections.abc import Sequence
import os
import json

from typing import cast

import holographic_forecast.data.data_models as data_models
import holographic_forecast.data.data_embedding as data_embedding

import tests.collection_test as collection_test

import tests.log_setup as log_setup

logger = log_setup.get_logger(__name__, "logs/parse-test.log")


def sample_json_responses() -> Sequence[data_models.OpenMeteoResponseJSON]:
    # Reads from cache. If nonexistent, pulls from open-meteo

    data_cache_filepath: str = "example/open-meteo/parse-test-data-cache.json"
    logger.info(f"Using data cache filepath: {data_cache_filepath}")

    # check if there is a file at the filepath

    if not os.path.exists(data_cache_filepath):
        logger.info(f"{data_cache_filepath} does not exist. Pulling data...")
        data_collector = collection_test.collect_data_sample(
            data_models.GeographicCordinate(
                latitude_deg=36.1716, longitude_deg=115.1391
            )
        )

        file_contents = [*data_collector.request()]
        logger.info(f"Saving data to {data_cache_filepath}")

        with open(data_cache_filepath, "w") as file:
            json.dump(file_contents, file)

    logger.info("Parsing data...")

    with open(data_cache_filepath, "r") as file:
        return cast(list[data_models.OpenMeteoResponseJSON], json.load(file))


def test_data_model_json_parsing():
    logger.info("Parsing data...")

    parsed_data = data_models.WeatherTimespanArea.from_openmeteo_json(
        json_responses=sample_json_responses()
    )

    logger.info(f"Done parsing data. Data info:\n{len(parsed_data.data) = }\n")

    predicted_cordinates: list[data_models.GeographicCordinate] = [
        weather_time_point.cordinate for weather_time_point in parsed_data.data[-1]
    ]

    embedded_data = data_embedding.WeatherSpanAreaEmbedder(
        parsed_data
    ).embed_model_input(predicted_cordinates=predicted_cordinates)

    logger.info(f"Done embedding data. Data info:\n{embedded_data = }")


if __name__ == "__main__":
    test_data_model_json_parsing()
