#!/usr/bin/env python3
# -*- coding: latin-1 -*-

import random
from pathlib import Path
from typing import Tuple
from recognition_model.utils import Logging

"""Data loader class to handle the data. Functionality includes splitting the data into train and test sets
and creating image pairs and labels for the siamese network.
"""

class DataLoader(object):
    def __init__(self, root: str, logger_level: str = "INFO") -> None:
        self._logger = Logging().create_logger(logger_name="Data Loader", logger_level=logger_level)
        self._logger.info("Initialising Data Loader")
        self._root = root

    def _get_player_paths(self) -> dict:
        paths_per_player = {}
        for team_dir in filter(Path.is_dir, Path(self._root).iterdir()):
            team_name = str(team_dir).split('/')[-1] # Get team name from path
            for player_dir in filter(Path.is_dir, team_dir.iterdir()):
                # Get all paths to images for that player
                player_path = list(map(str, player_dir.glob("*.jpg")))

                if team_name in paths_per_player:
                    paths_per_player[team_name].append(player_path)
                else:
                    paths_per_player[team_name] = [player_path]

        return paths_per_player

    def create_train_test_split(self, test_size: float = 0.2) -> Tuple[dict, dict]:

        paths_per_player = self._get_player_paths()
        train_set, test_set = {}, {}
        # Iterate over all teams and paths to player images
        for team, all_player_paths in paths_per_player.items():
            train_set[team], test_set[team] = [], []
            
            for player_images in all_player_paths:
                # Calculate how many images in test set for the player
                n_test_images = int(len(player_images) * test_size)
                # Shuffle paths and split by the test size
                random.shuffle(player_images)
                train_set[team].append(player_images[n_test_images:])
                test_set[team].append(player_images[:n_test_images])

        return train_set, test_set

    


    