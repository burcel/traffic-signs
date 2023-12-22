import os
from typing import TYPE_CHECKING, Optional

from core.config import config
from core.util import Util

if TYPE_CHECKING:
    from io import BufferedWriter


class Log:
    def __init__(self) -> None:
        self.log_file: Optional[BufferedWriter] = None
        self.log_name: str = ""  # Initiated in log creation time
        self.log_path: str = ""  # Initiated in log creation time

    def __del__(self):
        if self.log_file is not None:
            self.close()

    def initiate_log(self) -> None:
        """create folder and open log file"""
        # Create log folder if necessary
        Util.create_directory(config.log_path)
        self.log_name = f"{Util.get_time_as_filename()}.log"
        self.log_path = os.path.join(config.log_path, self.log_name)
        self.log_file = open(self.log_path, "wb")

    def log(self, log_str: str, write: bool = False, carriage_return: bool = False) -> None:
        """Print log and write to file if necessary"""
        full_log_str = f"{Util.get_time()}: {log_str}"

        if carriage_return is True:
            # Write to the same line
            print(full_log_str, end="\r")
        else:
            print(full_log_str)

        if write is True:
            if self.log_file is None:
                # Create new log file
                self.initiate_log()
            self.log_file.write(full_log_str.encode("utf8") + b"\n")
            self.log_file.flush()

    def close(self) -> None:
        """Close log file"""
        if self.log_file.closed is False:
            self.log_file.close()
            self.log_file = None


log = Log()
