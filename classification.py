import os
import random
from datetime import datetime
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from confusion_matrix import ConfusionMatrix
from core.config import config
from core.log import log
from core.util import Util
from data import Data
from dataset.customdataset import CustomDataset
from model import Model


class Classification:
    PRINT_INFO_THRESHOLD = 25

    def __init__(self) -> None:
        # Model
        self.model = Model()
        self.class_count = 43
        # Dataloader
        self.training_dataloader: Optional[DataLoader] = None
        self.validation_dataloader: Optional[DataLoader] = None
        self.dataloader: Optional[DataLoader] = None
        self.dataloader_num_workers = 0
        self.prefetch_factor = 1
        # Chart path
        self.loss_chart_name = f"model-{self.model.return_model_version()}-loss.png"
        self.loss_chart_path = os.path.join(config.model_path, self.loss_chart_name)
        self.accuracy_chart_name = f"model-{self.model.return_model_version()}-accuracy.png"
        self.accuracy_chart_path = os.path.join(config.model_path, self.accuracy_chart_name)
        # Training variables
        self.epoch_training_loss_list: List[float] = []
        self.epoch_validation_loss_list: List[float] = []
        self.training_accuracy_list: List[float] = []
        self.validation_accuracy_list: List[float] = []

    def load_training_input(self, batch_size) -> None:
        # Prepare input list
        input_list = Data.parse_gtsrb_training()
        # Filter the input list by image size
        input_list = Data.filter_input_by_img_size(input_list, 2500)
        # Shuffle the input list -> Better training-validation splitting
        random.Random(4).shuffle(input_list)
        # Prepare training and validation sets
        training_list, validation_list = Data.split_sets(input_list, config.validation_split_percentage)
        # Initialize training dataloader
        self.training_dataloader = DataLoader(
            CustomDataset(training_list),
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.dataloader_num_workers,
            prefetch_factor=self.prefetch_factor if self.dataloader_num_workers != 0 else None,
            collate_fn=CustomDataset.collate_fn,
        )
        # Initialize validation dataloader
        self.validation_dataloader = DataLoader(
            CustomDataset(validation_list),
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.dataloader_num_workers,
            prefetch_factor=self.prefetch_factor if self.dataloader_num_workers != 0 else None,
            collate_fn=CustomDataset.collate_fn,
        )
        # Initialize training variables
        self.epoch_training_loss_list.clear()
        self.epoch_validation_loss_list.clear()
        self.training_accuracy_list.clear()
        self.validation_accuracy_list.clear()

    def save_loss_plot(self) -> None:
        """Save loss plot for training and validation"""
        plt.figure()
        plt.title("Loss")
        plt.plot(self.epoch_training_loss_list, "b", label="training loss")
        plt.plot(self.epoch_validation_loss_list, "r", label="validation loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(loc="upper right")
        plt.savefig(self.loss_chart_path)
        plt.close()

    def save_accuracy_plot(self) -> None:
        """Save accuracy plot for training and validation"""
        plt.figure()
        plt.title("Accuracy")
        plt.plot(self.training_accuracy_list, "b", label="training")
        plt.plot(self.validation_accuracy_list, "r", label="validation")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend(loc="lower right")
        axes = plt.gca()
        axes.set_ylim([0, 100])  # type: ignore  # noqa: PGH003
        plt.savefig(self.accuracy_chart_path)
        plt.close()

    def reset_training(self) -> None:
        """Reset training variables"""
        # Model
        self.model = Model()
        # Training variables
        self.epoch_training_loss_list.clear()
        self.epoch_validation_loss_list.clear()
        self.training_accuracy_list.clear()
        self.validation_accuracy_list.clear()
        # Empty cuda cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def training(self, **kwargs) -> float:
        """Training function, return latest epoch validation loss"""
        # Hyperparameter config
        if len(kwargs) == 0:
            # Tuning is disabled -> Parse from config
            batch_size = config.batch_size
            learning_rate = config.learning_rate
            beta1 = config.beta1
            beta2 = config.beta2
        else:
            # Tuning is activated
            log.log("This is a tuning run.", write=True)
            # Parse from optimizer
            batch_size = config.batch_size
            learning_rate = kwargs["learning_rate"]
            beta1 = config.beta1
            beta2 = config.beta2
            # Reset training variables
            self.reset_training()

        # Print model summary
        log.log(f"Model: {self.model.return_model_version()}", write=True)
        total_params = self.model.summary()
        log.log(f"Parameter count: {total_params:,}", write=True)
        log.log(f"Learning rate: {learning_rate}", write=True)
        input("?")
        # Load datasets and initialize training variables
        self.load_training_input(batch_size)
        input("?")
        # Loss function
        criterion = nn.CrossEntropyLoss()
        # Optimizer
        optimizer = torch.optim.Adam(self.model.model.parameters(), lr=learning_rate, betas=(beta1, beta2))
        # TODO: Initial model efficiency calculation

        # Initiate training values
        training_start_time = datetime.utcnow()
        lowest_validation_loss = 999999.9
        hightest_validation_accuracy = 0.0
        for epoch_index in range(1, config.epoch + 1):
            epoch_start_time = datetime.utcnow()

            # Training
            self.model.train_mode()
            training_loss_list = []
            for input_index, (input_tensor, label_tensor) in enumerate(self.training_dataloader):  # type: ignore  # noqa: PGH003
                # Squeeze label tensor for cross entropy loss
                label_tensor.squeeze_(dim=1)
                # Clear accumulated gradient
                self.model.zero_grad()
                # Forward pass to get output
                output_tensor = self.model.return_output_tensor(input_tensor)
                loss = criterion(output_tensor, label_tensor)
                training_loss_list.append(loss.item())
                # Getting gradients w.r.t. parameters
                loss.backward()
                # Updating parameters
                optimizer.step()
                # Print training info
                if input_index % self.PRINT_INFO_THRESHOLD == 0:
                    log.log(
                        f"Training | "
                        f"Epoch: {epoch_index}/{config.epoch} | "
                        f"Step: {input_index:,}/{len(self.training_dataloader):,} | "  # type: ignore  # noqa: PGH003
                        f"loss: {np.mean(training_loss_list):.5f}"
                    )

            # Process training loss
            epoch_mean_training_loss = np.mean(training_loss_list)
            self.epoch_training_loss_list.append(epoch_mean_training_loss)  # type: ignore  # noqa: PGH003
            log.log(f"Training | Epoch: {epoch_index}/{config.epoch} | loss: {epoch_mean_training_loss:.5f}", write=True)

            # Validation
            self.model.eval_mode()
            validation_loss_list = []
            with torch.no_grad():
                for input_index, (input_tensor, label_tensor) in enumerate(self.validation_dataloader):  # type: ignore  # noqa: PGH003
                    # Squeeze label tensor for cross entropy loss
                    label_tensor.squeeze_(dim=1)
                    # Forward pass to get output
                    output_tensor = self.model.return_output_tensor(input_tensor)
                    # Calculate the loss
                    loss = criterion(output_tensor, label_tensor)
                    validation_loss_list.append(loss.item())
                    # Print validation info
                    if input_index % self.PRINT_INFO_THRESHOLD == 0:
                        log.log(
                            f"Validation | "
                            f"Epoch: {epoch_index}/{config.epoch} | "
                            f"Step: {input_index:,}/{len(self.validation_dataloader):,} | "  # type: ignore  # noqa: PGH003
                            f"loss: {np.mean(validation_loss_list):.5f}"
                        )

            # Process validation loss
            epoch_mean_validation_loss = np.mean(validation_loss_list)
            self.epoch_validation_loss_list.append(epoch_mean_validation_loss)  # type: ignore  # noqa: PGH003
            log.log(f"Validation | Epoch: {epoch_index}/{config.epoch} | loss: {epoch_mean_validation_loss:.5f}", write=True)
            # Model validation
            training_confusion_matrix, validation_confusion_matrix = self.check_model_efficiency()
            # Epoch validation loss
            if epoch_mean_validation_loss < lowest_validation_loss:
                # Save model
                self.model.save_model(epoch_index, optimizer.state_dict())
                log.log(">> Lowest validation loss", write=True)
                lowest_validation_loss = epoch_mean_validation_loss
            # Epoch validation accuracy
            if validation_confusion_matrix.accuracy > hightest_validation_accuracy:  # type: ignore  # noqa: PGH003
                log.log(">> Highest validation accuracy", write=True)
                hightest_validation_accuracy = validation_confusion_matrix.accuracy  # type: ignore  # noqa: PGH003
            # Epoch is finished
            log.log(f"Epoch training time: {Util.return_readable_time(epoch_start_time)}", write=True)
            log.log(f"Overall training time: {Util.return_readable_time(training_start_time)}")
            # Save plots
            log.log("Saving plots...")
            self.save_loss_plot()
            self.save_accuracy_plot()
            log.log("Plots are saved.")
            # TODO: early stop?

        log.log(f"Training is finished in {Util.return_readable_time(training_start_time)}", write=True)
        # Negative value is because of Bayesian Optimization (maximizing value)
        return lowest_validation_loss  # type: ignore  # noqa: PGH003

    def check_model_efficiency(self) -> Tuple[ConfusionMatrix, ConfusionMatrix]:
        """Calculate model efficiency of training and validation dataloaders and populate confusion matrices lists for plots"""
        self.model.eval_mode()
        with torch.no_grad():
            # Calculate training accuracy
            start_time = datetime.utcnow()
            log.log("Calculating training dataset accuracy...")
            training_confusion_matrix = self.calculate_dataloader_efficiency(self.training_dataloader)  # type: ignore  # noqa: PGH003
            self.training_accuracy_list.append(training_confusion_matrix.accuracy)  # type: ignore  # noqa: PGH003
            log.log(f"Finished in {Util.return_readable_time(start_time)}")
            log.log(f"Training metrics:\n{training_confusion_matrix.return_metrics_table()}", write=True)
            log.log(f"Training class metrics:\n{training_confusion_matrix.return_class_metrics_table()}", write=True)
            log.log(f"Confusion matrix:\n{training_confusion_matrix!s}", write=True)
            # Calculate validation accuracy
            start_time = datetime.utcnow()
            log.log("Calculating validation dataset accuracy...")
            validation_confusion_matrix = self.calculate_dataloader_efficiency(self.validation_dataloader)  # type: ignore  # noqa: PGH003
            self.validation_accuracy_list.append(validation_confusion_matrix.accuracy)  # type: ignore  # noqa: PGH003
            log.log(f"Finished in {Util.return_readable_time(start_time)}")
            log.log(f"Validation metrics:\n{validation_confusion_matrix.return_metrics_table()}", write=True)
            log.log(f"Validation class metrics:\n{validation_confusion_matrix.return_class_metrics_table()}", write=True)
            log.log(f"Confusion matrix:\n{validation_confusion_matrix!s}", write=True)
            return training_confusion_matrix, validation_confusion_matrix

    def calculate_dataloader_efficiency(self, dataloader: DataLoader) -> ConfusionMatrix:
        """Calculate dataloader efficiency"""
        confusion_matrix = ConfusionMatrix(self.class_count)
        for index, (input_tensor, label_tensor) in enumerate(dataloader, start=1):
            # Squeeze label tensor for cross entropy loss
            label_tensor.squeeze_(dim=1)
            # Forward pass to get output
            class_index_tensor, _ = self.model.return_output_class_tensor(input_tensor)
            # Traverse output tensor and label tensor
            for tensor_index, class_tensor in enumerate(class_index_tensor):
                # Populate confusion matrix with respect to output index and label index
                confusion_matrix.populate(label_tensor[tensor_index].item(), class_tensor.item())  # type: ignore  # noqa: PGH003
            if index % self.PRINT_INFO_THRESHOLD == 0:
                confusion_matrix.calculate_metrics()
                log.log(f"Step: {index:,}/{len(dataloader):,} Accuracy: {confusion_matrix.accuracy:6.2f}%")
        # Calculate final accuracy
        confusion_matrix.calculate_metrics()
        return confusion_matrix


def main():
    classification = Classification()
    classification.training()


if __name__ == "__main__":
    main()
