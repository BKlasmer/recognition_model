#!/usr/bin/env python3
# -*- coding: latin-1 -*-

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchLoader
from recognition_model.utils import Logging

"""Class for training the model
"""


class Trainer(object):
    def __init__(self, random_seed: int = 42, logger_level: str = "INFO") -> None:
        self._logger = Logging().create_logger(logger_name="Trainer", logger_level=logger_level)
        self._logger.info("Initialising Trainer")
        self._logger.info(f"Setting random seed to: {random_seed}")

    def cross_entropy_loss(self):
        return nn.BCELoss()

    def adam_optimizer(self, model_weights, learning_rate: float = 3e-4):
        return optim.Adam(model_weights, lr=learning_rate)

    def train(self, dataset, model, optimizer, loss_function, n_batches: int = 1000, batch_size: int = 32):

        loader = TorchLoader(dataset, batch_size=batch_size)

        loss_history = []

        for batch_number, batch in enumerate(loader):
            img1, img2, label = batch
            optimizer.zero_grad()
            logits = model(img1, img2)
            label = label.unsqueeze(1)
            label = label.type_as(logits)
            loss = loss_function(logits, label)
            loss.backward()
            optimizer.step()
            if batch_number % 5 == 0:
                print(f"Batch number: {batch_number}. Loss: {loss.item():.4f}")
                loss_history.append(loss.item())

            if batch_number == n_batches:
                break

        return model

    