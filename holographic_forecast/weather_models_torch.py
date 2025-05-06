# pyright: reportUnknownMemberType=false

from typing import override, cast, ClassVar
import torch
import torch.nn as nn

import holographic_forecast.data.data_encoding_torch as encoding


class QuantityInterpreterV1(nn.Module):
    char_embedding: nn.Embedding
    linear_combiner: nn.Parameter
    embedding_dim: int
    final_dim: int

    def __init__(self, embedding_dim: int, final_dim: int):
        super(QuantityInterpreterV1, self).__init__()

        self.embedding_dim = embedding_dim
        self.final_dim = final_dim

        self.char_embedding = nn.Embedding(
            num_embeddings=encoding.WeatherQuantityEncodingV1.CHAR_MAX_VALUE + 1,
            embedding_dim=self.embedding_dim,
        )

        self.linear_combiner = nn.Parameter(
            torch.rand((self.embedding_dim, self.final_dim))
        )

    @override
    def forward(self, query: encoding.WeatherQuantityEncodingV1) -> torch.Tensor:
        embedded_query = cast(torch.Tensor, self.char_embedding(query.data))

        return torch.einsum("ce,ke->k", embedded_query, self.linear_combiner)


class WeatherModelV1(nn.Module):
    quantity_meaning: QuantityInterpreterV1
    final_dim: ClassVar[int] = 16

    entries_meta_reducer: nn.Linear

    weather_time_point_processor: nn.Linear
    weather_time_area_reduced_dim: ClassVar[int] = 32

    weather_time_area_processor: nn.Linear
    weather_timespan_area_reduced_dim: ClassVar[int] = 64

    post_processor: nn.Sequential

    def __init__(self):
        super(WeatherModelV1, self).__init__()

        self.quantity_meaning = QuantityInterpreterV1(
            embedding_dim=16, final_dim=self.final_dim
        )
        self.entries_meta_reducer = nn.Linear(
            (WeatherModelV1.final_dim + 1) + (3 + WeatherModelV1.final_dim) + 3, 1
        )

        self.weather_time_point_processor = nn.Linear(
            (WeatherModelV1.final_dim + 1) + (3 + WeatherModelV1.final_dim) + 3,
            WeatherModelV1.weather_time_area_reduced_dim,
        )

        self.weather_time_area_processor = nn.Linear(
            WeatherModelV1.weather_time_area_reduced_dim,
            WeatherModelV1.weather_timespan_area_reduced_dim,
        )

        self.post_processor = nn.Sequential(
            nn.Linear(WeatherModelV1.weather_timespan_area_reduced_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    @override
    def forward(self, query: encoding.QueryEncodingV1) -> torch.Tensor:
        query_quantity_meaning: torch.Tensor = self.quantity_meaning(
            query.query_info.weather_quantity
        )

        query_meaning: torch.Tensor = torch.cat(  # shape (3 + final_dim)
            (
                query.query_info.cordinate.longitude_latitude,
                query.query_info.timestamp,
                query_quantity_meaning,
            )
        )

        for weather_time_area in query.weather_timespan_area.weather_time_areas:
            for weather_time_point in weather_time_area.weather_time_points:
                for weather_entry in weather_time_point.weather_entries:
                    weather_entry.weather_quantity.intermediate = torch.cat(
                        (
                            self.quantity_meaning(
                                weather_entry.weather_quantity
                            ),  # shape (final_dim)
                            weather_entry.data,  # (1)
                        )
                    )

                weather_time_point_meta: torch.Tensor = torch.cat(
                    (
                        weather_time_point.cordinate.longitude_latitude,
                        weather_time_point.timestamp,
                    )
                )

                # Combines weather entries (ragged, final_dim + 1), query meaning (3 + final_dim), and weather_time_point_meta (3)

                stacked_weather_entries: torch.Tensor = torch.stack(
                    [
                        cast(torch.Tensor, entry.weather_quantity.intermediate)
                        for entry in weather_time_point.weather_entries
                    ]
                )

                number_of_stacked_weather_entries: int = stacked_weather_entries.size(0)

                expanded_query_meaning: torch.Tensor = query_meaning.expand(
                    number_of_stacked_weather_entries, -1
                )

                expanded_weather_time_point_meta: torch.Tensor = (
                    weather_time_point_meta.expand(
                        number_of_stacked_weather_entries, -1
                    )
                )

                weather_time_point.intermediate = torch.einsum(
                    "rc->c",
                    self.entries_meta_reducer(
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

            weather_time_area.intermediate = torch.einsum(
                "rc->c",
                self.weather_time_point_processor(
                    torch.stack(
                        cast(
                            list[torch.Tensor],
                            [
                                weather_time_point.intermediate
                                for weather_time_point in weather_time_area.weather_time_points
                            ],
                        )
                    )
                ),
            )

        reduced_weather_timespan_area: torch.Tensor = torch.einsum(
            "rc->c",
            self.weather_time_area_processor(
                torch.stack(
                    cast(
                        list[torch.Tensor],
                        [
                            weather_time_area.intermediate
                            for weather_time_area in query.weather_timespan_area.weather_time_areas
                        ],
                    )
                )
            ),
        )

        return cast(torch.Tensor, self.post_processor(reduced_weather_timespan_area))
