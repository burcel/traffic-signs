from datetime import datetime
from typing import List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from core.log import log
from core.util import Util
from image import ImageProcessor


class CustomDataset(Dataset):
    PRINT_INFO_THRESHOLD = 1000

    def __init__(self, input_list: List[Tuple[str, int]]) -> None:
        self.device = Util.return_torch_device()
        self.input_list: List[Tuple[Tensor, Tensor]] = []
        # Create dataset
        start_time = datetime.utcnow()
        log.log(f"Creating dataset.. Device: {self.device.type}")
        for index, (img_path, class_id) in enumerate(input_list, start=1):
            # Create image tensor
            img_tensor = ImageProcessor.img_to_tensor(img_path).to(self.device)
            # Create class tensor
            class_tensor = torch.tensor([class_id]).to(self.device)
            # Add tensors to list
            self.input_list.append((img_tensor, class_tensor))
            if index % self.PRINT_INFO_THRESHOLD == 0:
                log.log(f"Index: {index:,}/{len(input_list):,}")
        log.log(f"Creating dataset is finished in {Util.return_readable_time(start_time)}")

    def __len__(self) -> int:
        """Return data set size"""
        return len(self.input_list)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Return item from data set"""
        return self.input_list[idx]

    @staticmethod
    def collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        """Process each input tensor in batch and use padding with respect to max batch size"""
        # Find max size in batch
        max_h = max([input_tensor.size(1) for input_tensor, _ in batch])
        max_w = max([input_tensor.size(2) for input_tensor, _ in batch])
        batch_input_list, batch_label_list = [], []
        for input_tensor, label_tensor in batch:
            batch_input_list.append(ImageProcessor.padding_center(input_tensor, max_h, max_w))
            batch_label_list.append(label_tensor)
        return torch.stack(batch_input_list), torch.stack(batch_label_list)
