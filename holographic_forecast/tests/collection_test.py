import datetime
import json
import logging
import sys

import holographic_forecast.data.data_models as data_models
import holographic_forecast.data.openmeteo_data_collection as odc

# NOTES:

# JSON file size:
#

logger = logging.getLogger(__name__)
logging.basicConfig(
	level=logging.INFO,
	handlers=[
		logging.StreamHandler(sys.stdout),
		logging.FileHandler("logs/collection-test.log"),
	],
)


def generate_points_radial():
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


def collect_data_sample(center: data_models.GeographicCordinate):
	points: list[data_models.GeographicCordinate] = [
		*center.points_within_radius_grid(
			radius_miles=100, distance_between_points_miles=75
		)
	]

	logger.info(f"{len(points) = }")

	start_date = datetime.date(2021, 1, 1)
	end_date = datetime.date(2021, 1, 2)

	logger.info(f"{start_date = }")
	logger.info(f"{end_date = }")

	data_collector = odc.OpenMeteoAreaDataCollector.from_points(
		list_of_points=points,
		start_date=start_date,
		end_date=end_date,
		hourly_parameters=odc.OPEN_METEO_HOURLY_PARAMETERS,
		daily_parameters=odc.OPEN_METEO_DAILY_PARAMETERS,
	)

	return data_collector


def test_collection_all_parameters():
	data_collector = collect_data_sample(
		data_models.GeographicCordinate(latitude_deg=36.1716, longitude_deg=115.1391)
	)

	logger.info("Getting data...")
	responses = data_collector.get()

	logger.info(f"Done getting data. Data info:\n{len(responses) = }\n")

	with open("example/open-meteo/open-meteo-4-param-response-2-day.json", "w") as f:
		json.dump(responses[0].json(), f)


if __name__ == "__main__":
	generate_points_radial()
