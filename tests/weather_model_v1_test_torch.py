import pathlib

import tests.log_setup as log_setup

import holographic_forecast.weather_models_torch as torch_models
import holographic_forecast.data.data_models as data_models
import holographic_forecast.data.data_encoding_torch as data_encoding

logger = log_setup.get_logger(__name__, "logs/weather-model-v1-test-torch.log")

SAMPLE_DATA_PATH: pathlib.Path = (
    pathlib.Path(__file__).parent / "sample" / "sample_noaa_weather_collection.pkl"
)


def test_model_v1():
    logger.info(f"Loading weather collection pkl from {SAMPLE_DATA_PATH}")
    weather_collection = data_models.NOAAWeatherCollection.load_file(SAMPLE_DATA_PATH)

    logger.info("Parsing data...")
    print(weather_collection)

    parsed_data = data_models.WeatherTimespanArea.from_noaa_weather_collection(
        weather_collection=weather_collection
    )

    logger.info(f"Done parsing data. Data info:\n{len(parsed_data.data) = }\n")

    encoded_data = data_encoding.WeatherTimespanAreaEncodingV1(parsed_data)
    query_info = data_encoding.QueryInfoEncodingV1(
        cordinate=data_models.GeographicCordinate(
            latitude_deg=36.1716, longitude_deg=115.1391
        ),
        weather_quantity=data_models.WeatherQuantity("TMAX"),
        timestamp=0.0,
    )

    query = data_encoding.QueryEncodingV1(
        query_info=query_info,
        weather_timespan_area=encoded_data,
    )

    model = torch_models.WeatherModelV1()

    output = model(query)

    logger.info(f"Done encoding data. Data info:\n{output = }\n")
