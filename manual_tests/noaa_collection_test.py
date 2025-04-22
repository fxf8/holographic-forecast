import asyncio
from pathlib import Path
import dotenv

import holographic_forecast.data.noaa_data_collection as noaa_weather_collection
import noaa_cdo_api

cache_path = Path("noaa_cache/collection_test_cache.pkl")

# Ensure the directory exists, then the file

Path(cache_path).parent.mkdir(parents=True, exist_ok=True)


async def test_noaa_collection():
    maybe_token = dotenv.dotenv_values().get("token", None)

    if maybe_token is None:
        raise Exception("Missing NOAA token")

    collector = noaa_weather_collection.NOAADataCollector(maybe_token)

    las_vegas_extent = noaa_cdo_api.Extent(
        latitude_min=34.05,
        longitude_min=-115.15,
        latitude_max=36.05,
        longitude_max=-113.15,
    )

    result = await collector.data_in_extent_and_timespan(
        dataset_id="GHCND",
        extent=las_vegas_extent,
        timespan_start="2022-01-01",
        timespan_end="2022-01-31",
    )

    # Print numbers from 0 to 9 using a loop

    print(result)

    result.save_file(cache_path)


if __name__ == "__main__":
    asyncio.run(test_noaa_collection())
