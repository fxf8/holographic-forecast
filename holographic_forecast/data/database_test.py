import sys
import logging

import holographic_forecast.data.data_database as weather_db

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/embedding-test.log"),
    ],
)

