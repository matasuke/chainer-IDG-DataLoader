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

    def test_attributes(self):
        pass

    def test_get_example_preload(self):
        randn = np.random.randint(0, len(self.IDG_Dataset))
        image, caption = self.IDG_Dataset.get_example(randn)

        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.shape[0], 2048)

        self.assertIsInstance(caption, np.ndarray)

    def test_get_example_raw_captions_with_no_preload(self):
        self.IDG_Dataset.preload_features= False
        self.IDG_Dataset.raw_caption = True

        randn = np.random.randint(0, len(self.IDG_Dataset))
        image, caption = self.IDG_Dataset.get_example(randn)

        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.shape[0], 2048)

        self.assertIsInstance(caption, list)

    def test_get_example_raw_image(self):
        self.IDG_Dataset.raw_img = True

        randn = np.random.randint(0, len(self.IDG_Dataset))
        image, caption = self.IDG_Dataset.get_example(randn)

        self.assetIsInstance()


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
