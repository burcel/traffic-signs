import os
from typing import Any, Iterator, Optional, Tuple

import torch
import torch.nn.functional as F  # noqa: N812
from tabulate import tabulate
from torch import Tensor
from torch.nn.parameter import Parameter

from core.config import config
from core.log import log
from core.util import Util
from nn.resnet18 import CVModel


class Model:
    def __init__(self) -> None:
        # Decide device acceleration
        self.device = Util.return_torch_device()
        # Create model directory if necessary
        Util.create_directory(config.model_path)
        self.model_path = os.path.join(config.model_path, f"model-{CVModel.VERSION}.pth")
        self.model = CVModel()
        self.model.to(self.device)
        self.optimizer_state_dict = None

    def return_model_version(self) -> str:
        """Return model version"""
        return self.model.VERSION

    def train_mode(self) -> None:
        """Change model into training mode"""
        self.model.train()

    def eval_mode(self) -> None:
        """Change model into eval mode"""
        self.model.eval()

    def return_model_parameters(self) -> Iterator[Parameter]:
        """Returns an iterator over model parameters"""
        return self.model.parameters()

    def return_model_layers(self) -> Any:
        """Returns model layers"""
        return self.model.layers

    def zero_grad(self) -> None:
        """Sets the gradients to zero"""
        self.model.zero_grad()

    def summary(self) -> int:
        """Print model summary"""
        log.log("-- Model Summary --")
        log.log(f"Groups of parameters: {len(list(self.model.parameters()))}")
        table_list = []
        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            total_params += params
            table_list.append([name, f"{params:,}", parameter.shape])
        print(tabulate(table_list, headers=["Name", "Parameter", "Shape"], tablefmt="presto"))
        print(f"Total Trainable Params: {total_params:,}")
        return total_params

    def load_model(self) -> None:
        """Load model in eval mode from file"""
        # Load model from file
        model_state = torch.load(self.model_path, map_location=self.device)
        # Load model parameters
        self.model.load_state_dict(model_state["model"], strict=False)
        # Send model to device
        self.model.to(self.device)
        # Load epoch index
        self.epoch_index = model_state["epoch"]
        # Load optimizer state dict -> To continue training
        self.optimizer_state_dict = model_state["optimizer_state_dict"]

    def save_model(self, epoch_index: int, optimizer_state_dict: Any, path: Optional[str] = None) -> None:
        """Save model & optimizer state dict to file"""
        model_state = {
            "epoch": epoch_index,
            "model": self.model.state_dict(),
            "optimizer_state_dict": optimizer_state_dict,  # optimizer.state_dict()
        }
        if path is None:
            path = self.model_path
        torch.save(model_state, path)
        log.log("Model is saved.", write=True)

    def return_output_tensor(self, input_tensor: Tensor) -> Tensor:
        """Forward input tensor into model and return output tensor"""
        return self.model(input_tensor)

    def return_output_class_tensor(self, input_tensor: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward input tensor into model and return class index and class percentage"""
        output_tensor = self.return_output_tensor(input_tensor)
        softmax_tensor = F.softmax(output_tensor, dim=1)
        class_index_tensor = torch.argmax(softmax_tensor, dim=1)
        return class_index_tensor, softmax_tensor
