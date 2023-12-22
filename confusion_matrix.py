from typing import List, Optional

import torch
from tabulate import tabulate
from torch import Tensor


class ConfusionMatrix:
    """
    Ref:
    https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1
    """

    def __init__(self, class_count: int) -> None:
        self.class_count: int = class_count
        self.confusion_matrix: Tensor = torch.zeros(class_count, class_count, dtype=torch.int32)
        self.class_recall_list: List[float] = []
        self.class_precision_list: List[float] = []
        self.class_f1_score_list: List[float] = []
        self.accuracy: Optional[float] = None
        self.recall: Optional[float] = None
        self.precision: Optional[float] = None
        self.f1_score: Optional[float] = None

    def __str__(self) -> str:
        confusion_matrix_list = []
        for class_index in range(self.class_count):
            class_sum = int(torch.sum(self.confusion_matrix[class_index]))
            class_list = []
            for prediction_class_index in range(self.class_count):
                prediction_percentage = self.confusion_matrix[class_index][prediction_class_index] / class_sum * 100
                class_list.append(
                    f"{self.confusion_matrix[class_index][prediction_class_index]:,}\n{prediction_percentage:6.2f}%"
                )
            confusion_matrix_list.append(class_list)
        return tabulate(confusion_matrix_list, tablefmt="grid")

    def return_metrics_table(self) -> str:
        """Return table with metrics inside"""
        metric_list = [
            ["Accuracy", "Recall", "Precision", "F1 score"],
            [f"{self.accuracy:6.2f}%", f"{self.recall:6.2f}%", f"{self.precision:6.2f}%", f"{self.f1_score:6.2f}%"],
        ]
        return tabulate(metric_list, tablefmt="grid")

    def return_class_metrics_table(self) -> str:
        """Return table with class metrics inside"""
        class_metrics_list = [["Recall", "Precision", "F1 score"]]
        for class_index in range(self.class_count):
            class_metrics_list.append(
                [
                    f"{self.class_recall_list[class_index]:6.2f}%",
                    f"{self.class_precision_list[class_index]:6.2f}%",
                    f"{self.class_f1_score_list[class_index]:6.2f}%",
                ]
            )
        return tabulate(class_metrics_list, tablefmt="grid")

    def populate(self, label_class_index: int, output_class_index: int) -> None:
        """Populate confusion matrix with indexes"""
        self.confusion_matrix[label_class_index][output_class_index] += 1

    def calculate_metrics(self) -> None:
        """Calculate recall, precision, f1 score metrics from populated confusion matrix"""
        # Reset variables
        self.class_recall_list = []
        self.class_precision_list = []
        self.class_f1_score_list = []
        # Sum everything
        total_count = self.confusion_matrix.sum().item()
        # Sum true label counts
        label_count_tensor = self.confusion_matrix.sum(1)
        # Sum predicted counts
        prediction_count_tensor = self.confusion_matrix.sum(0)
        # Check if matrix is empty
        if total_count == 0:
            raise ValueError("Failed to calculate accuracy on an empty confusion matrix")
        # Traverse class values
        accurate_prediction_count = 0
        for class_index in range(self.class_count):
            class_correct_prediction = self.confusion_matrix[class_index][class_index].item()
            accurate_prediction_count += class_correct_prediction
            # Calculate recall
            class_label_count = label_count_tensor[class_index].item()
            class_recall = 0.0 if class_label_count == 0 else class_correct_prediction / class_label_count * 100
            self.class_recall_list.append(class_recall)
            # Calculate precision
            class_prediction_count = prediction_count_tensor[class_index].item()
            class_precision = 0.0 if class_prediction_count == 0 else class_correct_prediction / class_prediction_count * 100
            self.class_precision_list.append(class_precision)
            # Calculate f1 score
            if class_recall == 0.0 or class_precision == 0:
                f1_score = 0.0
            else:
                f1_score = 2 * (class_recall * class_precision) / (class_recall + class_precision)
            self.class_f1_score_list.append(f1_score)
        # Calculate accuracy
        self.accuracy = accurate_prediction_count / total_count * 100
        # Calculate macro recall
        self.recall = sum(self.class_recall_list) / self.class_count
        # Calculate macro precision
        self.precision = sum(self.class_precision_list) / self.class_count
        # Calculate macro f1 score
        self.f1_score = sum(self.class_f1_score_list) / self.class_count
