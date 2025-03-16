import datetime
import json

import holographic_forecast.data_collection.data_models as data_models
import holographic_forecast.data_collection.openmeteo_data_collection as odc


def test_collection_all_parameters():
    center: data_models.GeographicCordinate = data_models.GeographicCordinate(
        latitude_deg=36.1716, longitude_deg=115.1391
    )

    # points: list[data_models.GeographicCordinate] = [
    #    *center.points_within_radius(radius_miles=100, distance_between_points_miles=75)
    # ]

    points: list[data_models.GeographicCordinate] = [center]

    start_date = datetime.date(2021, 1, 1)
    end_date = datetime.date(2021, 1, 2)

    print(
        f"{center = }\n"
        + "radius_miles=100, distance_between_points_miles=10\n"
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

    print(f"{len(responses) = }")

    print(f"{responses[0].json() = }")

    with open("example/open-meteo/open-meteo-4-param-response-2-day.json", "w") as f:
        json.dump(responses[0].json(), f)

if __name__ == "__main__":
    test_collection_all_parameters()
