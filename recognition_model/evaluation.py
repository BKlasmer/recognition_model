#!/usr/bin/env python3
# -*- coding: latin-1 -*-

import random
from recognition_model.utils import Logging

"""Class for evaluating the model
"""

class Evaluation(object):
    def __init__(self, random_seed: int = 42, logger_level: str = "INFO") -> None:
        self._logger = Logging().create_logger(logger_name="Evaluation", logger_level=logger_level)
        self._logger.info("Initialising Evaluation")
        self._logger.info(f"Setting random seed to: {random_seed}")