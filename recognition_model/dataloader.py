#!/usr/bin/env python3
# -*- coding: latin-1 -*-

from pathlib import Path
from recognition_model.utils import Logging

"""Data loader class to handle the data. Functionality includes splitting the data into train and test sets
and creating image pairs and labels for the siamese network.
"""

class DataLoader(object):
    def __init__(self, logger_level: str = "INFO") -> None:
        self._logger = Logging().create_logger(logger_name="Data Loader", logger_level=logger_level)
        self._logger.info("Initialising Data Loader")