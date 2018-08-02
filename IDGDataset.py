import json
import pickle
from pathlib import Path

from tqdm import tqdm
import cv2
import numpy as np
import chainer

from utils.process_image import ImgProcesser


class IDGDatasetBase(chainer.dataset.DatasetMixin):
    """
    Chainer Dataset Class for image captioning generation.

    Attributes
    ----------
    word_ids : dict
        map to ids from tokens.

    inv_word_ids : dict
        map to tokens from ids.

    captions: list
        list of captions loaded from dataset.

    images: list
        list of images loadef from dataset.

    cap2img: dict
        relatinships betweein caption id and image id.

    img_root : str
        path to directory of images.

    img_feature_root : str
        path to directory of image features.

    raw_caption: bool, default False
        use raw captions(list) instead of numpy.ndarray format.

    raw_img : bool, default False,
        use raw images insted of extracted features beforehand.

    img_mean : str, default imagenet
        image mean used for preprocess images.
        imagenet mean is used as a default.

    img_size : tuple, default (224, 224)
        output image size after processing images.
        This attribute is used only when you load raw images.

    preload_features : bool, default False
        preload all image features onto RAM.

    img_proc: class
        ImgProcesser class to preprocess images.

    img_features : numpy.ndarray
        numpy.ndarray to save all image features onto RAM.
        This attribute is available only when preload_features is True.
    """
    def __init__(
            self,
            dataset_path,
            vocab_path,
            img_root="",
            img_feature_root="",
            raw_caption=False,
            raw_img=False,
            img_size=(224, 224),
            img_mean='imagenet',
            preload_features=False,
    ):
        """
        parameters
        ----------
        dataset_path : str
            path to dataset which contains image info and captions
            preprocessed by mscoco2formatted.py and preprocess_tokens.py.

        vocab_path : str
            path to vocabulary dictionary created by preprocess_tokens.py.

        img_root : str
            path to directory of images.

        img_feature_root : str
            path to directory of image features.

        raw_caption : bool, default False
            use raw captions(list) instead of numpy.ndarray format.

        raw_img : bool, default False,
            use raw images insted of extracted features beforehand.

        img_mean : str, default imagenet
            image mean used for preprocess images.
            imagenet mean is used as a default.

        img_size : tuple, default (224, 224)
            output image size after processing images.
            This attribute is used only when you load raw images.

        preload_features : bool, default False
            preload all image features onto RAM.
        """

        if Path(dataset_path).exists():
            dataset = self.load_data(dataset_path)
            self.captions = dataset['captions']
            self.images = dataset['images']
        else:
            msg = 'File %s is not found.\n' % dataset_path
            raise FileNotFoundError(msg)

        if Path(vocab_path).exists():
            self.word_ids = self.load_data(vocab_path)
        else:
            msg = 'File %s is not found.\n' % vocab_path
            raise FileNotFoundError(msg)

        self.cap2img = {
            caption['caption_idx']: caption['img_idx'] for caption in self.captions
        }
        self.inv_word_ids = {
            v: k for k, v in self.word_ids.items()
        }

        if raw_img and img_root:
            self.img_proc = ImgProcesser(mean_type=img_mean)
            self.img_root = Path(img_root)
            if not self.img_root.exists() and not self.img_root.is_dir():
                msg = "image root %s is not found\n" % str(self.img_root)
                raise FileNotFoundError(msg)
        elif not raw_img and img_feature_root:
            self.img_feature_root = Path(img_feature_root)
            if not self.img_feature_root.exists() and not self.img_feature_root.is_dir():
                msg = "image feature root %s is not found\n" % str(self.img_feature_root)
                raise FileNotFoundError(msg)
        else:
            if raw_img:
                img_path = "img_root"
                img_type = "images"
            else:
                img_path = "img_feature_root"
                img_type = "image features"

            msg = '%s has to be defined to load %s\n' % (img_path, img_type)
            raise NameError(msg)

        if preload_features and not raw_img:
            print("Loading image features...")

            self.img_features = np.array(
                [
                    np.load('{0}.npz'.format(
                        self.img_feature_root / Path(image['file_path']).with_suffix("")
                    ))['arr_0'] for image in tqdm(self.images)
                ]
            )

        self.img_size = img_size
        self.raw_caption = raw_caption
        self.raw_img = raw_img
        self.preload_features = preload_features

    def __len__(self):
        return len(self.captions)

    def get_example(self, i):
        """
        get image and caption based on caption index.

        Parameters
        ----------
        index: int
            caption index of image and caption.
            both are extraced from self.images and self.captions

        Returns
        -------
        img: numpy.ndarray
            image RGB values or image features extracted by CNN model beforehand.

        Notes
        -----
        Use Raw Images
        if self.raw_img is True, then raw image insted of extracted features is used.
        This reads each images one by one.
        So it would take much time.

        Preload Features
        if self.preload_features is True, then preloaded feature vectores are used.
        It doesn't take times, but it consumes RAM.
        Be careful to use this functions if RAM is less than 16GM(in case of MSCOCO dataset).

        Load Each Features one by one
        if self.raw_img and self.preload_features are both False,
        then each feature vectors are loaded one by one.
        if would take much time than using preloaded features,
        but doesn't require much RAM.

        Use Raw Caption
        if self.raw_caption is True, then it returns list of caption.
        otherwise, it returns ndarray of caption.
        """

        if self.raw_img:
            img_path = self.img_root / self.images[self.cap2img[i]]['file_path']
            img = self.img_proc.load_img(
                str(img_path),
                img_size=self.img_size,
                resize=True,
                expand_dim=False
            )

        else:
            if self.preload_features:
                img = self.img_features[self.cap2img[i]]
            else:
                img_path = Path(self.images[self.cap2img[i]]['file_path']).with_suffix("")
                img = np.load('{0}.npz'.format(self.img_feature_root / img_path))['arr_0']

        if self.raw_caption:
            caption = self.captions[i]['caption']
        else:
            caption = np.array(self.captions[i]['caption'])

        return img, caption

    def get_raw_data(self, index):
        """
        get raw image path and raw caption.

        Parameters
        ----------
        index: int
            caption index of image and caption.
            both are extraced from self.images and self.captions

        Returns
        -------
        img_path: str
            image path designated by caption index.
        raw_caption: list
            list of caption tokens.
        """
        img_path = self.images[self.cap2img[index]]['file_path']
        img_path = self.img_root / img_path

        caption = self.captions[index]['captions']
        raw_caption = self.index2token(caption)

        return img_path, raw_caption

    @staticmethod
    def load_data(path):
        '''load pickle and json file.'''

        in_path = Path(path)
        ext = in_path.suffix
        if ext == '.pkl':
            with in_path.open('rb') as f:
                dataset = pickle.load(f)
        elif ext == '.json':
            with in_path.open('r') as f:
                dataset = json.load(f)
        else:
            msg = 'File %s can not be loaded.\n \
                   choose json or pickle format' % path
            raise TypeError(msg)

        return dataset

    def token2index(self, tokens):
        """return indies from tokens."""

        return [self.word_ids[token] for token in tokens]

    def index2token(self, indices):
        """return tokens from indices."""
        return [self.inv_word_ids[index] for index in indices]

    def calc_unk_ratio(self, data):
        """base function for callculate <UNK> ratio"""
        unk = sum((np.array(s['caption']) == self.word_ids['<UNK>']).sum() for s in data)
        words = sum(np.array(['caption']).size for s in data)

        return round(float(unk / words), 3)

    @property
    def get_word_ids(self):
        """get word_ids"""
        return self.word_ids

    @property
    def get_unk_ratio(self):
        """get <UNK> ratio in self.captions"""
        return self.calc_unk_ratio([cap for cap in self.captions])

    @property
    def get_configurations(self):
        """get configurations"""
        res = {}

        res['vocabulary_size'] = len(self.get_word_ids)
        res['num_size'] = len(self.captions)
        res['num_size'] = len(self.images)
        res['unk_ratio'] = self.get_unk_ratio
