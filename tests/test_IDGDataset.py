import unittest
from pathlib import Path
import numpy as np

from IDGDataset import IDGDatasetBase

class TestIDGDatasetBase(unittest.TestCase):
    def setup(self):
        dataset_path = 'data/captions/converted/MSCOCO_captions/train2014.pkl'
        vocab_path = 'data/vocab/mscoco_train2014_vocab.pkl'
        img_root = 'data/images/original'
        img_feature_root  = 'data/images/features/ResNet50'

        self.IDG_Dataset = IDGDatasetBase(
            dataset_path,
            vocab_path,
            img_root,
            img_feature_root,
            raw_caption=False,
            raw_img=False,
            img_mean='imagenet',
            preload_features=True
        )

    def test_get_example_preload(self):
        pass

    def test_get_raw_data(self):
        pass

    def test_index2token(self):
        pass

    def test_token2index(self):
        pass

    def test_get_word_ids(self):
        pass

    def test_get_unk_ratio(self):
        pass

    def get_configurations(self):
        pass
