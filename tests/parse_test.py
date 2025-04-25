import pathlib

import holographic_forecast.data.data_models as data_models
import holographic_forecast.data.data_embedding as data_embedding

import tests.log_setup as log_setup

logger = log_setup.get_logger(__name__, "logs/parse-test.log")

SAMPLE_DATA_PATH: pathlib.Path = pathlib.Path("tests/sample/collection_test_cache.pkl")


def test_data_model_json_parsing():
    logger.info("Loading weather collection pkl from {SAMPLE_DATA_PATH}")
    weather_collection = data_models.NOAAWeatherCollection.load_file(SAMPLE_DATA_PATH)

    logger.info("Parsing data...")
    print(weather_collection)

    return

    parsed_data = data_models.WeatherTimespanArea.from_noaa_weather_collection(
        weather_collection=weather_collection
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
