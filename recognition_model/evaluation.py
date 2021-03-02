#!/usr/bin/env python3
# -*- coding: latin-1 -*-

import random
import numpy as np
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

    def calculate_logits(self, model, test_loader, number_test_points=250):
        loader = TorchLoader(test_loader, batch_size=number_test_points)

        for batch in loader:
            img1, img2, labels = batch
            break

        predicted_scores = model(img1, img2)
        predicted_scores = predicted_scores.detach().numpy()
        labels = labels.numpy()

        return predicted_scores, labels

    def plot_evaluation(self, model, test_loader, number_test_points):
        predicted_scores, labels = self.calculate_logits(model, test_loader, number_test_points)
        roc_auc, fpr, tpr, thresholds = self.calculate_auc(predicted_scores, labels)

        plt.figure(figsize=[15, 6])
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color="darkorange", label=f"ROC Curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
        plt.xlim([-0.02, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")

        plt.subplot(1, 2, 2)
        
        x_range = np.linspace(0, 1, 100)
        for beta in [0.1, 0.5, 1, 2, 10]:
            f_beta, x_range = self.calculate_fbeta(predicted_scores, labels, beta, x_range)
            plt.plot(x_range, f_beta, label=f"Beta = {beta}")

        plt.ylim([0.0, 1.05])
        plt.xlabel("Thresholds")
        plt.ylabel("F-Beta Score")
        plt.title("F-Beta")
        plt.legend(loc="lower right")


    @staticmethod
    def calculate_auc(predicted_scores, labels):

        fpr, tpr, thresholds = roc_curve(labels, predicted_scores)
        roc_auc = auc(fpr, tpr)

        return roc_auc, fpr, tpr, thresholds

    @staticmethod
    def calculate_fbeta(predicted_scores, labels, beta, x_range):

        f_beta = []
        for threshold in x_range:
            binary_predictions = [1 if x >= threshold else 0 for x in predicted_scores]
            f_beta.append(fbeta_score(labels, binary_predictions, beta=beta))

        return f_beta, x_range
