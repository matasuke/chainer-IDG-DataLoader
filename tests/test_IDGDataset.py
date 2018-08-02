import unittest
from pathlib import Path
import numpy as np

from IDGDataset import IDGDatasetBase


class TestIDGDatasetBase(unittest.TestCase):

    def setUp(self):
        self.dataset_path = Path('data/captions/converted/MSCOCO_captions/train2014.pkl')
        self.vocab_path = Path('data/vocab/mscoco_train2014_vocab.pkl')
        self.img_root = Path('data/images/original')
        self.img_feature_root = Path('data/images/features/ResNet50')
        self.raw_caption = True
        self.raw_img = False
        self.img_mean = "imagenet"
        self.img_size = (224, 224)
        self.preload_features = False

        self.IDG_Dataset = IDGDatasetBase(
            self.dataset_path,
            self.vocab_path,
            self.img_root,
            self.img_feature_root,
            raw_caption=self.raw_caption,
            raw_img=self.raw_img,
            img_mean=self.img_mean,
            preload_features=self.preload_features
        )

    def test_attributes(self):
        self.assertEqual(self.IDG_Dataset.img_feature_root, self.img_feature_root)
        self.assertEqual(self.IDG_Dataset.raw_caption, self.raw_caption)
        self.assertEqual(self.IDG_Dataset.raw_img, self.raw_img)
        self.assertEqual(self.IDG_Dataset.img_size, self.img_size)
        self.assertEqual(self.IDG_Dataset.preload_features, self.preload_features)

    def test_getitem_features(self):
        for i in range(len(self.IDG_Dataset)):
            img_feature, caption = self.IDG_Dataset[i]

            img_id = self.IDG_Dataset.cap2img[i]
            img_path = Path(self.IDG_Dataset.images[img_id]['file_path']).with_suffix("")
            img_feature_inter = np.load('{0}.npz'.format(self.img_feature_root / img_path))['arr_0']
            caption_inter = self.IDG_Dataset.captions[i]['caption']

            self.assertEqual(img_feature.any(), img_feature_inter.any())
            self.assertIsInstance(img_feature, np.ndarray)
            self.assertEqual(caption, caption_inter)

    def test_index2token(self):
        randn = np.random.randint(len(self.IDG_Dataset))
        tokens = self.IDG_Dataset[randn][1]

        ids = self.IDG_Dataset.index2token(tokens)
        reversed_tokens = self.IDG_Dataset.token2index(ids)

        self.assertEqual(tokens, reversed_tokens)

    def test_word_ids(self):
        for token in self.IDG_Dataset.word_ids:
            word_id = self.IDG_Dataset.word_ids[token]
            reversed_tokens = self.IDG_Dataset.inv_word_ids[word_id]
            self.assertEqual(token, reversed_tokens)


if __name__ == '__main__':
    unittest.main()
