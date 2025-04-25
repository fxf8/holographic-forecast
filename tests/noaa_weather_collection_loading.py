import asyncio

import holographic_forecast.data.data_models as data_models

import manual_tests.noaa_collection_test as nct


async def test_loading():
    if not nct.cache_path.exists():
        await nct.test_noaa_collection()

    result = data_models.NOAAWeatherCollection.load_file(nct.cache_path)

    print(result)

if __name__ == "__main__":
    asyncio.run(test_loading())
