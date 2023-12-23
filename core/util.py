import os
from datetime import datetime

import humanize
import torch


class Util:
    @staticmethod
    def get_time() -> str:
        """Get current UTC time"""
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def get_time_as_filename() -> str:
        """Get current UTC time"""
        return datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")

    @staticmethod
    def create_directory(directory_name: str) -> None:
        """Create directory if it does not exist"""
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

    @staticmethod
    def return_readable_time(start_datetime: datetime) -> str:
        """Given starting datetime, return human readable time string"""
        return humanize.precisedelta((datetime.utcnow() - start_datetime).total_seconds())

    @staticmethod
    def return_torch_device() -> torch.device:
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        return device
