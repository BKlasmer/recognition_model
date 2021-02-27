#!/usr/bin/env python3
# -*- coding: latin-1 -*-

import random
import torch.nn as nn
from typing import Tuple
from recognition_model.utils import Logging

"""Class for Siamese Network in PyTorch
"""


class SiameseNetwork(nn.Module):
    def __init__(self, random_seed: int = 42, logger_level: str = "INFO") -> None:
        super(SiameseNetwork, self).__init__()
        self._logger = Logging().create_logger(logger_name="Siamese Network", logger_level=logger_level)
        self._logger.info("Initialising Siamese Network")
        self._logger.info(f"Setting random seed to: {random_seed}")

        random.seed(random_seed)