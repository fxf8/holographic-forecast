import pathlib

import holographic_forecast.data.data_models as data_models
import holographic_forecast.data.data_encoding as data_encoding

import tests.log_setup as log_setup

logger = log_setup.get_logger(__name__, "logs/encode-test.log")

SAMPLE_DATA_PATH: pathlib.Path = (
    pathlib.Path(__file__).parent / "sample" / "sample_noaa_weather_collection.pkl"
)


def test_data_model_json_parsing():
    logger.info(f"Loading weather collection pkl from {SAMPLE_DATA_PATH}")
    weather_collection = data_models.NOAAWeatherCollection.load_file(SAMPLE_DATA_PATH)

    logger.info("Parsing data...")
    print(weather_collection)

    parsed_data = data_models.WeatherTimespanArea.from_noaa_weather_collection(
        weather_collection=weather_collection
    )

    logger.info(f"Done parsing data. Data info:\n{len(parsed_data.data) = }\n")

    encoded_tensor = (
        data_encoding.WeatherModelV1Encoder.encode_weather_timespan_area_to_tensor(
            weather_timespan_area=parsed_data,
            predicted_cordinate=data_models.GeographicCordinate(
                latitude_deg=36.1716, longitude_deg=115.1391
            ),
            predicted_quantity=data_models.WeatherQuantity("Temperature"),
        )
    )

    logger.info(f"Done encoding data. Data info:\n{encoded_tensor = }")


if __name__ == "__main__":
    test_data_model_json_parsing()
