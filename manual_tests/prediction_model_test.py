# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false

import datetime
import pathlib
import json
import os
import logging
from typing import cast

import tensorflow as tf
import keras

import holographic_forecast.data.data_models as data_models
import holographic_forecast.data.openmeteo_data_collection as odc
import holographic_forecast.data.data_embedding as data_embedding
import holographic_forecast.weather_models_tf as weather_models

import tests.log_setup as log_setup

logger = log_setup.get_logger(__name__, "logs/prediction-model-test.log")


def generate_data_sample() -> data_models.WeatherSpanArea:
    os.makedirs("example/open-meteo", exist_ok=True)

    json_cache_path: pathlib.Path = pathlib.Path(
        "example/open-meteo/model-test-data-cache.json"
    )

    if not json_cache_path.exists():
        logger.info(f"{json_cache_path} does not exist. Pulling data...")

        # las vegas cordinates: 36.1716° N, 115.1391° W

        las_vegas: data_models.GeographicCordinate = data_models.GeographicCordinate(
            latitude_deg=36.1716, longitude_deg=-115.1391
        )

        # log location

        logger.info(f"Generating data sample for: {las_vegas}")

        points: list[data_models.GeographicCordinate] = [
            *las_vegas.points_within_radius_radial_lines(
                radial_lines_count=5,
                radius_miles=100,
                distance_between_first_point_miles=50,
            )
        ]

        logger.info(f"{len(points) = }")

        # data over 10 days

        start_date = datetime.date(2021, 1, 1)
        end_date = datetime.date(2021, 1, 10)

        logger.info(f"{start_date = }")
        logger.info(f"{end_date = }")

        data_collector = odc.OpenMeteoAreaSpanDataCollector.from_points(
            list_of_points=points,
            start_date=start_date,
            end_date=end_date,
            hourly_parameters=odc.OPEN_METEO_HOURLY_PARAMETERS,
            daily_parameters=odc.OPEN_METEO_DAILY_PARAMETERS,
        )

        response_data: list[data_models.OpenMeteoResponseJSON] = (
            data_collector.request()
        )

        with open(json_cache_path, "w") as f:
            json.dump(response_data, f)

    with open(json_cache_path, "r") as f:
        return data_models.WeatherSpanArea.from_openmeteo_json(
            cast(list[data_models.OpenMeteoResponseJSON], json.load(f))
        )


def embedded_data_sample() -> tf.Tensor:
    weather_time_span = generate_data_sample()
    logging.info(f"{weather_time_span = }")

    predicted_cordinates: list[data_models.GeographicCordinate] = [
        weather_time_point.cordinate
        for weather_time_point in weather_time_span.data[-1]
    ]

    # Shape (batch_size, timesteps, n_points, n_features)
    return cast(
        tf.Tensor,
        data_embedding.WeatherSpanAreaEmbedder(weather_time_span)
        .embed_model_input(predicted_cordinates=predicted_cordinates)
        .to_tensor(),
    )


def test_prediction_model(model_type: type):
    """
    Simply tests whether or not the model is functional. Does not include training or accuracy
    """

    embedded_data: tf.Tensor = embedded_data_sample()

    n_features: int = cast(int, embedded_data.shape[-1])

    prediction_model_v1: keras.Model = model_type(n_features=n_features)

    predictions: tf.Tensor = cast(tf.Tensor, prediction_model_v1(embedded_data))

    logger.info(f"{predictions.shape = }")


if __name__ == "__main__":
    test_prediction_model(weather_models.WeatherModelV1)
