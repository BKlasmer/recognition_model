#!/usr/bin/env python3
# -*- coding: latin-1 -*-

import random
import torch
import torch.nn as nn
from typing import Tuple
from recognition_model.utils import Logging

"""Class for Siamese Network in PyTorch
"""


class SiameseNetwork(nn.Module):
    def __init__(self, logger_level: str = "INFO") -> None:
        super(SiameseNetwork, self).__init__()
        self._logger = Logging().create_logger(logger_name="Siamese Network", logger_level=logger_level)
        self._logger.info("Initialising Siamese Network")

        self.convnet = nn.Sequential(

            nn.Conv2d(3, 8, kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1), nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1), nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1), nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1), nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
        
        )

        self.fully_connected = nn.Sequential(

            nn.Linear(2 * 32 * 10 * 10, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 2)

        )

    def forward(self, input1, input2):
        # Forward Pass
        vector1 = self.convnet(input1)
        vector2 = self.convnet(input2)
        vector1 = vector1.view(vector1.size()[0], -1)
        vector2 = vector2.view(vector2.size()[0], -1)
        combined_vector = torch.cat((vector1, vector2), 0)
        logits = self.fully_connected(combined_vector)
        return logits
