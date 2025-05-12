import pathlib

import tests.log_setup as log_setup

import holographic_forecast.weather_models_torch as torch_models
import holographic_forecast.data.data_models as data_models
import holographic_forecast.weather_forecaster_torch as wf_torch

logger = log_setup.get_logger(__name__, "logs/weather-forcaster-torch.log")

SAMPLE_DATA_PATH: pathlib.Path = (
    pathlib.Path(__file__).parent / "sample" / "sample_noaa_weather_collection.pkl"
)

progress_path = pathlib.Path(__file__).parent / "sample" / "progress.pkl"


def forecaster_test():
    logger.info(f"Loading weather collection pkl from {SAMPLE_DATA_PATH}")
    weather_collection = data_models.NOAAWeatherCollection.load_file(SAMPLE_DATA_PATH)

    logger.info("Parsing data...")
    print(weather_collection)

    weather_timespan_area = (
        data_models.WeatherTimespanArea.from_noaa_weather_collection(
            weather_collection=weather_collection
        )
    )

    logger.info(
        f"Done parsing data. Data info:\n{len(weather_timespan_area.data) = }\n"
    )

    weather_model = torch_models.WeatherModelV1()

    weather_forecaster = wf_torch.WeatherForecasterTorch(
        weather_model=weather_model,
        weather_timespan_area=weather_timespan_area,
        progress_path=progress_path,
    )

    res = weather_forecaster.forecast(
        cordinate=data_models.GeographicCordinate(
            latitude_deg=36.1716, longitude_deg=115.1391
        ),
        timestamp=0.0,
        quantity=data_models.WeatherQuantity("TMAX"),
    )

    logger.info(f"Result: {res}")


if __name__ == "__main__":
    forecaster_test()
