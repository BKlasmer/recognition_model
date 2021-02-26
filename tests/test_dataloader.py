#!/usr/bin/env python3
# -*- coding: latin-1 -*-

import numpy as np
from recognition_model import DataLoader

""" Example boilerplate file
"""

def test_get_player_paths():
    # Setup
    desired = {"team_github": [["data/unit_test/team_github/person_0/github_logo.jpg"]]}

    # Exercise
    dataloader = DataLoader(root="data/unit_test")
    actual = dataloader._get_player_paths()

    # Verify
    assert actual == desired
    # Cleanup - none necessary
