import unittest
from pathlib import Path
import numpy as np

from mscoco2formatted import *

class Testmscoco2formatted(unittest.TestCase):
    def setUp(self):
        MSCOCO_DATA = Path('data/captions/train/mscoco.json')
        OUTPUT_DIR = Path('data/result')

        self.assertTrue(MSCOCO_DATA.exists())
        self.assertTrue(OUTPUT_DIR.eexists())

    def test_read_mscoco(self):
        INVALID_PATH = Path('data/captions/train/invalid.json')

