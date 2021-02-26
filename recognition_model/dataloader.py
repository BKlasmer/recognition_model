#!/usr/bin/env python3
# -*- coding: latin-1 -*-

from pathlib import Path
from recognition_model.utils import Logging

"""Data loader class to handle the data. Functionality includes splitting the data into train and test sets
and creating image pairs and labels for the siamese network.
"""

class DataLoader(object):
    def __init__(self, root: str, logger_level: str = "INFO") -> None:
        self._logger = Logging().create_logger(logger_name="Data Loader", logger_level=logger_level)
        self._logger.info("Initialising Data Loader")
        self._root = root

        paths_per_person = []
        for team_dir in filter(Path.is_dir, Path(root).iterdir()):
            for player_dir in filter(Path.is_dir, team_dir.iterdir()):
                paths_per_person.append(list(map(str, player_dir.glob("*.jpg"))))

        self.paths_per_person = [paths for paths in paths_per_person if len(paths) > 1]