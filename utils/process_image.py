"""
Image preprocesser to load and save images correctly.
"""

import numpy as np
import cv2


class ImgProcesser:
    """
    Image preprocesser to load and save images properly.

    Attributes
    img_mean: numpy.ndarray
        mean values substracted from input images
    ----------
    """

    def __init__(self, mean_type=None):
        '''
        Parameters
        ----------
        mean_type: str or list of size three or None, default None
            mean type used for substracting these values from each images.
            This value has to be one of 'imagenet', None, list.

        Note
        ----
            mean type of imagenet is channel wise mean values
            calculated from all images from imagenet.
            This mean value is usually used as a pre-process for CNN.
        '''
        if mean_type is None:
            self.img_mean = np.zeros([3, 1, 1])

        # BGR order for openCV
        # This is channel wise mean values.
        elif mean_type == 'imagenet':
            self.img_mean = np.ndarray([3, 1, 1], dtype=np.float32)
            self.img_mean[0] = 103.939
            self.img_mean[1] = 116.779
            self.img_mean[2] = 123.68

        elif len(mean_type) == 3:
            self.img_mean = np.ndarray([3, 1, 1], dtype=np.float32)
            self.img_mean[0] = mean_type[0]
            self.img_mean[1] = mean_type[1]
            self.img_mean[2] = mean_type[2]

    def load_img(self,
                 img_path,
                 img_size=(224, 224),
                 resize=True,
                 expand_dim=True):
        '''
        load image and preprocess it based on self.img_mean

        Parameters
        ----------
        img_path: str
            path to image
        img_size: tuple of size 2, default (224, 244)
            expected image size to be resized.
        resize: bool, default True
            resize image or not.
        expand_dim: bool, default True
            expand dims after preprocess image.

        Returns
        -------
        img: numpy.ndarray
            ndarray of preprocessed image.

        Note
        ----
        when both img_size and resize are set,
        image is resized based on img_size.
        So when resize is False, image is not resized
        even if img_size is set.
        '''

        img = cv2.imread(img_path).astype(np.float32)
        input_size = (img.shape[0], img.shape[1])

        if resize and input_size != img_size:
            img = cv2.resize(img, img_size)

        img = img.transpose(2, 0, 1)
        img -= self.img_mean

        if expand_dim:
            img = np.expand_dims(img, axis=0)

        return img

    def save_img(self, img_array, save_path):
        '''
        save processed images.

        Parameters
        ----------
        img_array: numpy.ndarray
            processed images
        save_path: str
            path to save image.
        '''

        img = img_array.transpose(1, 2, 0)
        cv2.imwrite(save_path, img)

    def open_img(self, img_array):
        cv2.imshow('ImgProcesser', img_array)
