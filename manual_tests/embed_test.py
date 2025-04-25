import holographic_forecast.data.data_models as data_models
import holographic_forecast.data.data_embedding as data_embedding

from tests.parse_test import sample_json_responses

import tests.log_setup as log_setup

logger = log_setup.get_logger(__name__, "logs/embed-test.log")


def test_embedding():
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
    test_embedding()
