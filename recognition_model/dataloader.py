#!/usr/bin/env python3
# -*- coding: latin-1 -*-

import random
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset
from PIL import Image
from pathlib import Path
from typing import Tuple
from recognition_model.utils import Logging

"""Data loader class to handle the data. Functionality includes splitting the data into train and test sets
and creating image pairs and labels for the siamese network.
"""


class DataLoader(IterableDataset):
    def __init__(self, root: str, random_seed: int = 42, logger_level: str = "INFO") -> None:
        self._root = root
        self._logger = Logging().create_logger(logger_name="Data Loader", logger_level=logger_level)
        self._logger.info("Initialising Data Loader")
        self._logger.info(f"Setting random seed to: {random_seed}")
        self.train_set, self.test_set = self.create_train_test_split()
        self.transforms = transforms.Compose([transforms.ToTensor()])

        random.seed(random_seed)

    def __iter__(self):
        return iter(self.generate_team_image_pairs_and_labels(self.train_set))

    def _get_player_paths(self) -> dict:
        """Iterates through the directory of images and returns a dictionary of team names as keys and
        a list of paths to player images as values.

        E.g. {"team": [[player1 paths], [player2 paths], ...]}

        Returns:
            dict: Dictionary of teams and image paths of their players
        """
        paths_per_player = {}
        for team_dir in filter(Path.is_dir, Path(self._root).iterdir()):
            team_name = str(team_dir).split("/")[-1]  # Get team name from path
            for player_dir in filter(Path.is_dir, team_dir.iterdir()):
                # Get all paths to images for that player
                player_path = list(map(str, player_dir.glob("*.jpg")))

                if team_name in paths_per_player:
                    paths_per_player[team_name].append(player_path)
                else:
                    paths_per_player[team_name] = [player_path]

        return paths_per_player

    def create_train_test_split(self, test_size: float = 0.2) -> Tuple[dict, dict]:
        """Creates train and test splits

        Args:
            test_size (float, optional): Percentage size of test class. Defaults to 0.2.

        Returns:
            Tuple[dict, dict]: Train and test dictionaries in the format _get_players_paths generates
        """
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

    def generate_team_image_pairs_and_labels(self, dataset: dict) -> Tuple[str, str, int]:
        """Generator object that returns pairs of paths to images and their corresponding label

        Args:
            dataset (dict): Train or test dictionary in the format _get_players_paths provides

        Yields:
            Iterator[Tuple[str, str, int]]: player 1 image path, player 2 image path, label
        """
        while True:
            # Pick a random team and player
            team_1 = self.random_team(dataset)
            image_1 = self.random_player(dataset, team_1)

            # Generate a different random team
            team_2 = self.random_team(dataset)
            while team_1 == team_2:
                team_2 = self.random_team(dataset)

            # Radomly choose between another player from the same team, or a player from different team
            image_2, label = random.choice([(self.random_player(dataset, team_1), 1), (self.random_player(dataset, team_2), 0)])

            yield self.decode_image(image_1), self.decode_image(image_2), label

    def decode_image(self, image_path):
        return self.transforms(Image.open(image_path))

    @staticmethod
    def random_team(dataset: dict) -> str:
        return random.choice(list(dataset.keys()))

    @staticmethod
    def random_player(dataset: dict, team: str) -> str:
        return random.choice(random.choice(dataset[team]))
