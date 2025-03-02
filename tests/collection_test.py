from datetime import datetime

import arcaneforecast.data_collection.data_models as data_models
import arcaneforecast.data_collection.openmeteo_data_collection as odc


def test_collection_all_parameters():
    center: data_models.GeographicCordinate = data_models.GeographicCordinate(
        latitude_deg=36.1716, longitude_deg=115.1391
    )

    points: list[data_models.GeographicCordinate] = [
        *center.points_within_radius(radius_miles=100, distance_between_points_miles=50)
    ]

    start_date = datetime(2021, 1, 1)
    end_date = datetime(2021, 2, 1)

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
        hourly_parameters=odc.OPEN_METEO_HOURLY_PARAMETERS,
        daily_parameters=odc.OPEN_METEO_DAILY_PARAMETERS,
    )

    responses = data_collector.get()


if __name__ == "__main__":
    test_collection_all_parameters()
