import unittest
from pathlib import Path
import numpy as np

from mscoco2formatted import *

class Testmscoco2formatted(unittest.TestCase):
    def setUp(self):
        MSCOCO_DATA = Path('data/captions/original/MSCOCO_captions_en/captions_train2014.json')
        OUTPUT_DIR = Path('data/captions/formatted/MSCOCO_captions/captions_train2014_formatted.pickle')

        self.assertTrue(MSCOCO_DATA.exists())
        self.assertFalse(OUTPUT_DIR.exists())

    def test_read_mscoco(self):
        pass

    def test_make_groups(self):
        pass

    def test_make_formatted(self):
        pass

