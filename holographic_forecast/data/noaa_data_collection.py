import datetime
import noaa_cdo_api
import holographic_forecast.data.data_models as data_models


class NOAADataCollector:
    noaa_client: noaa_cdo_api.NOAAClient

    def __init__(self, token: str):
        self.noaa_client = noaa_cdo_api.NOAAClient(token)

    async def stations_in_extent_and_timespan(
        self,
        dataset_id: str,
        extent: noaa_cdo_api.Extent,
        timespan_start: datetime.datetime | datetime.date | str,
        timespan_end: datetime.datetime | datetime.date | str,
        max_results: int = 1000,
    ) -> list[noaa_cdo_api.json_responses.StationIDJSON]:
        start_date_iso: str = (
            (
                timespan_start.date().isoformat()
                if isinstance(timespan_start, datetime.datetime)
                else timespan_start.isoformat()
            )
            if isinstance(timespan_start, (datetime.datetime, datetime.date))
            else timespan_start
        )

        end_date_iso: str = (
            (
                timespan_end.date().isoformat()
                if isinstance(timespan_end, datetime.datetime)
                else timespan_end.isoformat()
            )
            if isinstance(timespan_end, (datetime.datetime, datetime.date))
            else timespan_end
        )

        first_station_result = await self.noaa_client.get_stations(
            datasetid=dataset_id,
            extent=extent,
            startdate=start_date_iso,
            enddate=end_date_iso,
            limit=1,
        )

        if (
            "results" not in first_station_result
            and "metadata" not in first_station_result
        ):  # in this scenario, it is a ratelimit json value
            raise ValueError("noaa returned ratelimit json value")

        total_available_number_of_results = first_station_result["metadata"][
            "resultset"
        ]["count"]

        if max_results <= total_available_number_of_results:
            raise ValueError(
                f"max_results ({max_results}) must be greater than or equal to total_available_number_of_results ({total_available_number_of_results})"
            )

        max_results_per_request: int = 1000
        collected_results: list[noaa_cdo_api.json_responses.StationIDJSON] = []
        collected_results.append(first_station_result["results"][0])

        for result_number in range(
            1, total_available_number_of_results, max_results_per_request
        ):
            results = await self.noaa_client.get_stations(
                datasetid=dataset_id,
                extent=extent,
                startdate=start_date_iso,
                enddate=end_date_iso,
                limit=max_results_per_request,
                offset=result_number,
            )

            if (
                "results" not in results and "metadata" not in results
            ):  # in this scenario, it is a ratelimit json value
                raise ValueError("noaa returned ratelimit json value")

            collected_results.extend(results["results"])

        return collected_results

    async def data_in_extent_and_timespan(
        self,
        dataset_id: str,
        extent: noaa_cdo_api.Extent,
        timespan_start: datetime.datetime | datetime.date | str,
        timespan_end: datetime.datetime | datetime.date | str,
        max_results: int = 1000,
    ) -> data_models.NOAAWeatherCollection:
        start_date_iso: str = (
            (
                timespan_start.date().isoformat()
                if isinstance(timespan_start, datetime.datetime)
                else timespan_start.isoformat()
            )
            if isinstance(timespan_start, (datetime.datetime, datetime.date))
            else timespan_start
        )

        end_date_iso: str = (
            (
                timespan_end.date().isoformat()
                if isinstance(timespan_end, datetime.datetime)
                else timespan_end.isoformat()
            )
            if isinstance(timespan_end, (datetime.datetime, datetime.date))
            else timespan_end
        )

        stations = await self.stations_in_extent_and_timespan(
            dataset_id=dataset_id,
            extent=extent,
            timespan_start=start_date_iso,
            timespan_end=end_date_iso,
        )

        first_result = await self.noaa_client.get_data(
            datasetid=dataset_id,
            stationid=[station_info["id"] for station_info in stations],
            startdate=start_date_iso,
            enddate=end_date_iso,
            limit=1,
        )

        if (
            "results" not in first_result and "metadata" not in first_result
        ):  # in this scenario, it is a ratelimit json value
            raise ValueError("noaa returned ratelimit json value")

        total_available_number_of_results = first_result["metadata"]["resultset"][
            "count"
        ]

        MAX_RESULTS_PER_REQUEST: int = 1000
        collected_results: list[noaa_cdo_api.json_schemas.DatapointJSON] = []

        for offset in range(
            1,
            min(total_available_number_of_results, max_results) - 1,
            MAX_RESULTS_PER_REQUEST,
        ):
            results = await self.noaa_client.get_data(
                datasetid=dataset_id,
                stationid=[station_info["id"] for station_info in stations],
                startdate=start_date_iso,
                enddate=end_date_iso,
                limit=MAX_RESULTS_PER_REQUEST,
                offset=offset,
            )

            if (
                "results" not in results and "metadata" not in results
            ):  # in this scenario, it is a ratelimit json value
                raise ValueError("noaa returned ratelimit json value")

            collected_results.extend(results["results"])

        collected_results.append(first_result["results"][0])

        weather_collection: data_models.NOAAWeatherCollection = (
            data_models.NOAAWeatherCollection(data=[], noaa_stations={})
        )

        weather_collection.import_noaa_json(
            json_response=collected_results,
            stations_info=[station_info for station_info in stations],
        )

        return weather_collection
