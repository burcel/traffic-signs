import os

import toml


class Config:
    PATH = "config.toml"

    def __init__(self):
        """Read YAML file and populate variables"""
        with open(self.PATH) as f:
            config = toml.load(f)
            # <path>
            path = config["path"]
            self.root_path = path["root"]
            self.model_path = os.path.join(self.root_path, path["model_folder"])
            self.log_path = os.path.join(self.root_path, path["log_folder"])
            self.data_path = os.path.join(self.root_path, path["data_folder"])
            # <model>
            training = config["training"]
            self.validation_split_percentage = training["validation_split_percentage"]
            self.batch_size = training["batch_size"]
            self.epoch = training["epoch"]
            self.early_stop_epoch = training["early_stop_epoch"]
            # <adam>
            adam = config["adam"]
            self.learning_rate = adam["learning_rate"]
            self.beta1 = adam["beta1"]
            self.beta2 = adam["beta2"]
            # <gtsrb>
            gtsrb = config["gtsrb"]
            self.gtsrb_training_path = os.path.join(self.data_path, gtsrb["training_folder"])
            self.gtsrb_test_path = os.path.join(self.data_path, gtsrb["test_folder"])


config = Config()
