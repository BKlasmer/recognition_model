#!/usr/bin/env python3
# -*- coding: latin-1 -*-

import random
import torch
from recognition_model.utils import Logging

"""Class for training the model
"""


class Trainer(object):
    def __init__(self, random_seed: int = 42, logger_level: str = "INFO") -> None:
        self._logger = Logging().create_logger(logger_name="Trainer", logger_level=logger_level)
        self._logger.info("Initialising Trainer")
        self._logger.info(f"Setting random seed to: {random_seed}")
