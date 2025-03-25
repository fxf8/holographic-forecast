import sys
import logging

import holographic_forecast.data.data_models as data_models
import holographic_forecast.data.data_embedding as data_embedding

from holographic_forecast.tests.parse_test import sample_json_responses

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/embedding-test.log"),
    ],
)


def test_embedding():
    parsed_data = data_models.WeatherSpanArea.from_openmeteo_json(
        json_responses=sample_json_responses()
    )

    logger.info(f"Done parsing data. Data info:\n{len(parsed_data.data) = }\n")

    embedded_data = data_embedding.WeatherSpanAreaEmbedder(
        parsed_data
    ).embed_model_input()

    logger.info(f"Done embedding data. Data info:\n{embedded_data = }")


if __name__ == "__main__":
    test_embedding()
