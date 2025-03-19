import sys
import logging

import holographic_forecast.data.data_models as data_models
import holographic_forecast.data.data_embedding as data_embedding

from holographic_forecast.tests.parse_test import sample_json_responses

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/prediction-model-test.log"),
    ],
)
