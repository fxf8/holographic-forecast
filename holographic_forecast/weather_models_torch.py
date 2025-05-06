# pyright: reportUnknownMemberType=false

from typing import override, cast, ClassVar
import torch
import torch.nn as nn

import holographic_forecast.data.data_encoding_torch as encoding


class QuantityInterpreterV1(nn.Module):
    char_embedding: nn.Embedding
    char_embedding_dim: int
    quantity_processor: nn.Linear
    quantity_meaning_dim: int

    def __init__(
        self, quantity_meaning_dim: int, char_embedding_dim: int | None = None
    ):
        super(QuantityInterpreterV1, self).__init__()

        if char_embedding_dim is None:
            char_embedding_dim = quantity_meaning_dim

        self.char_embedding_dim = char_embedding_dim
        self.quantity_meaning_dim = quantity_meaning_dim

        self.char_embedding = nn.Embedding(
            num_embeddings=encoding.WeatherQuantityEncodingV1.CHAR_MAX_VALUE + 1,
            embedding_dim=self.char_embedding_dim,
        )

        self.quantity_processor = nn.Linear(
            in_features=self.char_embedding_dim,
            out_features=self.quantity_meaning_dim,
        )

    @override
    def forward(self, query: encoding.WeatherQuantityEncodingV1) -> torch.Tensor:
        """
        Computes the meaning of a quantity

        Returns shape: (quantity_meaning_dim,)
        """

        embedded_query = cast(torch.Tensor, self.char_embedding(query.data))

        return cast(
            torch.Tensor, self.quantity_processor(torch.einsum("rc->c", embedded_query))
        )


class QueryInfoInterpreterV1(nn.Module):
    quantity_interpreter: QuantityInterpreterV1
    query_meaning_dim: int
    quantity_processor: nn.Linear

    def __init__(self, quantity_interpreter: QuantityInterpreterV1):
        super(QueryInfoInterpreterV1, self).__init__()

        self.quantity_interpreter = quantity_interpreter
        self.query_meaning_dim = 3 + quantity_interpreter.quantity_meaning_dim

        self.quantity_processor = nn.Linear(
            in_features=self.query_meaning_dim,
            out_features=self.query_meaning_dim,
        )

    @override
    def forward(self, query_info: encoding.QueryInfoEncodingV1) -> torch.Tensor:
        """
        Computes the meaning of a query info

        Returns shape: (query_meaning_dim = 3 + quantity_meaning_dim)
        """

        return cast(
            torch.Tensor,
            self.quantity_processor(
                torch.cat(
                    (
                        query_info.cordinate.longitude_latitude,
                        query_info.timestamp,
                        self.quantity_interpreter(query_info.weather_quantity),
                    )
                )
            ),
        )


class EntryInterpreterV1(nn.Module):
    quantity_interpreter: QuantityInterpreterV1
    entry_meaning_dim: int
    entry_processor: nn.Linear

    def __init__(self, quantity_interpreter: QuantityInterpreterV1):
        super(EntryInterpreterV1, self).__init__()

        self.quantity_interpreter = quantity_interpreter
        self.entry_meaning_dim = 1 + quantity_interpreter.quantity_meaning_dim

        self.entry_processor = nn.Linear(
            in_features=self.entry_meaning_dim,
            out_features=self.entry_meaning_dim,
        )

    @override
    def forward(self, entry: encoding.WeatherEntryEncodingV1) -> torch.Tensor:
        """
        Computes the meaning of an entry

        Returns shape: (entry_meaning_dim = 1 + quantity_meaning_dim)
        """

        return cast(
            torch.Tensor,
            self.entry_processor(
                torch.cat(
                    (
                        entry.data,
                        self.quantity_interpreter(entry.weather_quantity),
                    )
                )
            ),
        )


class WeatherTimePointInterpreterV1(nn.Module):
    entry_interpreter: EntryInterpreterV1
    query_interpreter: QueryInfoInterpreterV1

    query_meaning_dim: int
    quantity_meaning_dim: int
    weather_time_point_meaning_dim: int

    def __init__(
        self,
        query_interpreter: QueryInfoInterpreterV1,
        entry_interpreter: EntryInterpreterV1,
        weather_time_point_meaning_dim: int,
    ):
        super(WeatherTimePointInterpreterV1, self).__init__()

        self.entry_interpreter = entry_interpreter
        self.query_interpreter = query_interpreter

        self.query_meaning_dim = query_interpreter.query_meaning_dim
        self.quantity_meaning_dim = (
            entry_interpreter.quantity_interpreter.quantity_meaning_dim
        )
        self.weather_time_point_meaning_dim = weather_time_point_meaning_dim

    @override
    def forward(
        self,
        weather_time_point: encoding.WeatherTimePointEncodingV1,
        query_meaning: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns shape: (weather_time_point_meaning_dim,)
        """

        weather_time_point_meta: torch.Tensor = torch.cat(
            (
                weather_time_point.cordinate.longitude_latitude,
                weather_time_point.timestamp,
            )
        )

        stacked_weather_entries = torch.stack(
            [
                self.entry_interpreter(entry)
                for entry in weather_time_point.weather_entries
            ]
        )

        num_weather_entries = stacked_weather_entries.size(0)

        expanded_query_meaning = query_meaning.expand(num_weather_entries, -1)

        expanded_weather_time_point_meta = weather_time_point_meta.expand(
            num_weather_entries, -1
        )

        return cast(
            torch.Tensor,
            self.entry_interpreter(
                torch.einsum(
                    "rc->c",
                    (
                        torch.cat(
                            (
                                stacked_weather_entries,
                                expanded_query_meaning,
                                expanded_weather_time_point_meta,
                            ),
                            dim=1,
                        )
                    ),
                )
            ),
        )


class WeatherTimeAreaInterpreterV1(nn.Module):
    weather_time_point_interpreter: WeatherTimePointInterpreterV1
    weather_time_area_meaning_dim: int
    weather_time_area_processor: nn.Linear

    def __init__(
        self,
        weather_time_point_interpreter: WeatherTimePointInterpreterV1,
        weather_time_area_meaning_dim: int,
    ):
        super(WeatherTimeAreaInterpreterV1, self).__init__()

        self.weather_time_point_interpreter = weather_time_point_interpreter
        self.weather_time_area_meaning_dim = weather_time_area_meaning_dim

        self.weather_time_area_processor = nn.Linear(
            in_features=self.weather_time_area_meaning_dim,
            out_features=self.weather_time_area_meaning_dim,
        )

    @override
    def forward(
        self,
        weather_time_area: encoding.WeatherTimeAreaEncodingV1,
        query_meaning: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns shape: (weather_time_area_meaning_dim,)
        """

        return cast(
            torch.Tensor,
            self.weather_time_area_processor(
                torch.stack(
                    [
                        self.weather_time_point_interpreter(
                            weather_time_point, query_meaning
                        )
                        for weather_time_point in weather_time_area.weather_time_points
                    ]
                )
            ),
        )


class WeatherTimespanInterpreterV1(nn.Module):
    weather_time_area_interpreter: WeatherTimeAreaInterpreterV1
    weather_timespan_meaning_dim: int
    weather_timespan_processor: nn.Linear

    def __init__(
        self,
        weather_time_area_interpreter: WeatherTimeAreaInterpreterV1,
        weather_timespan_meaning_dim: int,
    ):
        super(WeatherTimespanInterpreterV1, self).__init__()

        self.weather_time_area_interpreter = weather_time_area_interpreter
        self.weather_timespan_meaning_dim = weather_timespan_meaning_dim

        self.weather_timespan_processor = nn.Linear(
            in_features=self.weather_timespan_meaning_dim,
            out_features=self.weather_timespan_meaning_dim,
        )

    @override
    def forward(
        self,
        weather_timespan_area: encoding.WeatherTimespanAreaEncodingV1,
        query_meaning: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns shape: (weather_timespan_meaning_dim,)
        """

        return cast(
            torch.Tensor,
            self.weather_timespan_processor(
                torch.stack(
                    [
                        self.weather_time_area_interpreter(
                            weather_time_area, query_meaning
                        )
                        for weather_time_area in weather_timespan_area.weather_time_areas
                    ]
                )
            ),
        )


class WeatherModelV1(nn.Module):
    quantity_interpreter: QuantityInterpreterV1
    entry_interpreter: EntryInterpreterV1
    query_interpreter: QueryInfoInterpreterV1
    weather_time_point_interpreter: WeatherTimePointInterpreterV1
    weather_time_area_interpreter: WeatherTimeAreaInterpreterV1
    weather_timespan_interpreter: WeatherTimespanInterpreterV1

    post_processor: nn.Sequential

    def __init__(self):
        super(WeatherModelV1, self).__init__()

        self.quantity_interpreter = QuantityInterpreterV1(quantity_meaning_dim=16)
        self.entry_interpreter = EntryInterpreterV1(self.quantity_interpreter)
        self.query_interpreter = QueryInfoInterpreterV1(self.quantity_interpreter)
        self.weather_time_point_interpreter = WeatherTimePointInterpreterV1(
            query_interpreter=self.query_interpreter,
            entry_interpreter=self.entry_interpreter,
            weather_time_point_meaning_dim=32,
        )
        self.weather_time_area_interpreter = WeatherTimeAreaInterpreterV1(
            weather_time_point_interpreter=self.weather_time_point_interpreter,
            weather_time_area_meaning_dim=64,
        )
        self.weather_timespan_interpreter = WeatherTimespanInterpreterV1(
            weather_time_area_interpreter=self.weather_time_area_interpreter,
            weather_timespan_meaning_dim=128,
        )

        self.post_processor = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=1),
        )

    @override
    def forward(self, query: encoding.QueryEncodingV1) -> torch.Tensor:
        query_meaning: torch.Tensor = self.query_interpreter(query)

        weather_timespan_area_meaning: torch.Tensor = self.weather_timespan_interpreter(
            query.weather_timespan_area, query_meaning
        )

        return cast(torch.Tensor, self.post_processor(weather_timespan_area_meaning))
