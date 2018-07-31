import json
import pickle
from pathlib import Path
import numpy as np

import chainer

class Seq2SeqDatasetBase(chainer.dataset.DatasetMixin):
    def __init__(
            self,
            dataset,
            vocab,
            img_feature_root,
            img_root,
            img_mean='imagenet',
            preload_features=False,
    ):

        if Path(dataset).exists():
            self.dataset = load_data(dataset)


        self.dataset = dataset
        self.captions = dataset['captions']
        self.vocab = vocab['images']

    @staticmethod
    def load_data(path):
        in_path = Path(path)
        ext = in_path.suffix
        if ext == '.pickle':
            with in_path.open('rb') as f:
                dataset = pickle.load(f)
        elif ext = '.json':
            with in_path.open('r') as f:
                dataset = json.load(f)
        else:
            msg = 'File %s can not be loaded.\n \
                   choose json or pickle format' % path
            raise TypeError(msg)

        return dataset
