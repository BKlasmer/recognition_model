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

    def get_player_paths(self) -> dict:
        paths_per_player = {}
        for team_dir in filter(Path.is_dir, Path(self._root).iterdir()):
            team_name = str(team_dir).split('/')[-1]
            for player_dir in filter(Path.is_dir, team_dir.iterdir()):

                player_path = list(map(str, player_dir.glob("*.jpg")))

                if team_name in paths_per_player:
                    paths_per_player[team_name].append(player_path)
                else:
                    paths_per_player[team_name] = player_path

        return paths_per_player

    