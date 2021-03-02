#!/usr/bin/env python3
# -*- coding: latin-1 -*-

import random
import matplotlib.pyplot as plt
from recognition_model.utils import Logging
from torch.utils.data import DataLoader as TorchLoader
from sklearn.metrics import roc_curve, auc, fbeta_score


"""Class for evaluating the model
"""

class Evaluation(object):
    def __init__(self, logger_level: str = "INFO") -> None:
        self._logger = Logging().create_logger(logger_name="Evaluation", logger_level=logger_level)
        self._logger.info("Initialising Evaluation")

    def calculate_auc(self, model, test_loader):

        loader = TorchLoader(test_loader, batch_size=250)

        for batch in loader:
            img1, img2, labels = batch
            break

        predicted_scores = model(img1, img2)
        predicted_scores = predicted_scores.detach().numpy()
        labels = labels.numpy()

        fpr, tpr, thresholds = roc_curve(labels, predicted_scores)
        roc_auc = auc(fpr, tpr)

        return roc_auc, fpr, tpr, thresholds
