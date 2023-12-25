import os
from datetime import datetime
from typing import Dict, List, Tuple

import imagesize
import matplotlib.pyplot as plt
import pandas as pd

from core.config import config
from core.log import log
from core.util import Util


class Data:
    @staticmethod
    def parse_gtsrb_training() -> List[Tuple[str, int]]:
        """Parse GTSRB training set and return list of (img_path, class_id)"""
        start_datetime = datetime.utcnow()
        log.log("Reading GTSRB Training dataset...")
        input_list = []
        for folder in os.listdir(config.gtsrb_training_path):
            folder_path = os.path.join(config.gtsrb_training_path, folder)
            if os.path.isdir(folder_path):
                csv_path = os.path.join(folder_path, f"GT-{folder}.csv")
                if os.path.isfile(csv_path):
                    df = pd.read_csv(csv_path, delimiter=";")
                    for input_dict in df.to_dict("records"):
                        input_list.append((os.path.join(folder_path, input_dict["Filename"]), input_dict["ClassId"]))
        log.log(f"Reading is finished in {Util.return_readable_time(start_datetime)}")
        log.log(f"Input length: {len(input_list):,}")
        return input_list

    @staticmethod
    def parse_gtsrb_test() -> List[Tuple[str, int]]:
        """Parse GTSRB test set and return list of (img_path, class_id)"""
        log.log("Reading GTSRB Test dataset")
        input_list = []
        csv_path = os.path.join(os.path.dirname(os.path.dirname(config.gtsrb_test_path)), "GT-final_test.csv")
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path, delimiter=";")
            for input_dict in df.to_dict("records"):
                input_list.append((os.path.join(config.gtsrb_test_path, input_dict["Filename"]), input_dict["ClassId"]))
        log.log(f"Input length: {len(input_list):,}")
        return input_list

    @staticmethod
    def split_sets(input_list: List[Tuple[str, int]], percentage: int) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """Split training and validation sets and return them individually, input list must be [(image_path, class_id)]"""
        # Calculate count values for each class
        class_count: Dict[int, int] = {}
        for _, class_id in input_list:
            class_count[class_id] = class_count.get(class_id, 0) + 1
        # Determine the threshold for each class in validation set
        for class_id in class_count:
            class_count[class_id] = int(class_count[class_id] * percentage / 100)
        # Initialize training and validation sets
        training_list, validation_list = [], []
        # Traverse input list and prepare training and validation sets
        for img_path, class_id in input_list:
            if class_count[class_id] > 0:
                validation_list.append((img_path, class_id))
                class_count[class_id] -= 1
            else:
                training_list.append((img_path, class_id))
        log.log(f"Training: {len(training_list):,}, Validation: {len(validation_list):,}, Percentage: {percentage}%")
        return training_list, validation_list

    @staticmethod
    def create_img_size_histogram(img_size_list: List[int]) -> None:
        """Create histogram chart from image sizes and save it"""
        plt.figure()
        plt.title("Image Size Histogram")
        plt.xlabel("Image Size")
        plt.ylabel("Frequency")
        plt.hist(img_size_list, color="blue", alpha=0.8, ec="white")
        plt.savefig("histogram-image-size.png")
        plt.close()

    @staticmethod
    def create_class_histogram(class_list: List[int], class_count: int) -> None:
        """Create histogram chart from class ids and save it"""
        plt.figure()
        plt.title("Class Histogram")
        plt.xlabel("Class Id")
        plt.ylabel("Frequency")
        plt.hist(class_list, color="blue", alpha=0.8, bins=class_count, ec="white")
        plt.savefig("histogram-class.png")
        plt.close()

    @classmethod
    def create_gtsrb_training_histograms(cls) -> None:
        """Parse GTSRB training set and create image size & class frequency histograms"""
        c1, c2, c3 = 0, 0, 0
        img_size_list, class_list = [], []
        for img_path, class_id in cls.parse_gtsrb_training():
            class_list.append(class_id)
            h, w = imagesize.get(img_path)
            img_size = h * w
            img_size_list.append(img_size)
            if img_size > 10000:
                c1 += 1
            if img_size > 5000:
                c2 += 1
            if img_size > 2500:
                c3 += 1
        cls.create_img_size_histogram(img_size_list)
        cls.create_class_histogram(class_list, len(set(class_list)))
        log.log(f"img size > 100x100: {c1:,}")
        log.log(f"img size >   70x70: {c2:,}")
        log.log(f"img size >   50x50: {c3:,}")

    @staticmethod
    def filter_input_by_img_size(input_list: List[Tuple[str, int]], img_size: int) -> List[Tuple[str, int]]:
        """Filter given input list with respect to given image size"""
        start_datetime = datetime.utcnow()
        log.log(f"Filtering input list with respect to image size: {img_size:,}")
        res = []
        for img_path, class_id in input_list:
            h, w = imagesize.get(img_path)
            if h * w <= img_size:
                res.append((img_path, class_id))
        log.log(f"Filtering is finished in {Util.return_readable_time(start_datetime)}")
        log.log(f"Input length: {len(res):,}")
        return res


input_list = Data.parse_gtsrb_test()
print(len(input_list))
input_list = Data.filter_input_by_img_size(input_list, 71**2)
print(len(input_list))
