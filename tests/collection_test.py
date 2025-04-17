from typing import cast
from noaa_cdo_api import json_responses, json_schemas

resp: json_responses.DataJSON = cast(json_responses.DataJSON, object())

t = resp["results"][0]

json_schemas.DatapointJSON
