import unittest
from pathlib import Path
import numpy as np

from IDGDataset import IDGDatasetBase

class TestIDGDatasetBase(unittest.TestCase):
    def setup(self):
        IMAGE_DIR_PATH = Path('data/')
