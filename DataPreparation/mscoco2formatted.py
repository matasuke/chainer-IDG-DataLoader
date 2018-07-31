'''
convert MSCOCO caption dataset to formatted one easier to preprocess

This script allows the user to convert MSCOCO caption dataset to formatted one,
which is easier to preprocess for image captioning task.
'''

import json
import pickle
import argparse
from pathlib import Path

from tqdm import tqdm


def read_mscoco(mscoco_path):
    '''
    read mscoco caption dataset and return annotations and images separatoly.

    Parametors
    ----------
    mscoco_path: str
        The file location of the mscoco caption dataset.
        The file has to be json format.

    Returns
    -------
    annots: dict
        A list of dictionaries that contain
        'image_id', 'caption', 'tokenized_caption', 'id' used in MSCOCO caption dataset.
    imgs: dict
        A list of dictionaries that contain
        'license', 'file_name', 'coco_url', 'height',
        'width', 'data_captured', 'flickr_url', 'id' used in MSCOCO caption dataset.
    '''

    p = Path(mscoco_path)
    with open(p) as f:
        dataset = json.load(f)

    annots = dataset['annotations']
    imgs = dataset['images']

    return annots, imgs

def make_groups(annots):
    '''
    make annotation groups based on 'image_id'.

    Parametors
    ----------
    annots: list
        A list of dictionaries for annotations returned in read_mscoco function

    Returns
    -------
    itoa: dict
        dictionary containing annotations mapped from 'img_id'.
    '''
    itoa = {}

    for a in tqdm(annots):
        img_id = a['image_id']
        if img_id not in itoa:
            itoa[img_id] = []
        itoa[img_id].append(a)

    return itoa

def make_formatted(itoa, imgs):
    EXIST_TOKEN = False
    out_data = []

    for i, img in enumerate(tqdm(imgs)):
        img_id = img['id']

        # img['file_name'] format is usually like 'COCO_train2014_0003232.jpg'
        # data_origin is where dataset comes from.
        # So train2014 should be set to data_origin by img['file_name'].split('_')[1]
        data_origin = img['file_name'].split('_')[1]

        pairs = {}
        pairs['file_path'] = str(Path(data_origin, img['file_name']))
        pairs['id'] = img_id

        sentences = []
        annots = itoa[img_id]

        if 'tokenized_caption' in annots[0]:
            EXIST_TOKEN = True
            tokenized = []

        for a in annots:
            sentences.append(a['caption'])
            if EXIST_TOKEN:
                tokenized.append(a['tokenized_caption'])

        pairs['captions'] = sentences
        if EXIST_TOKEN:
            pairs['tokenized_captions'] = tokenized

        out_data.append(pairs)

    return out_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('MSCOCO_DATASET', type=str,
                        help='path to MSCOCO caption dataset.')
    parser.add_argument('OUT', type=str,
                        help='path to output\(File format has to be json or pickle\)')
    args = parser.parse_args()

    in_path = Path(args.MSCOCO_DATASET)
    out_path = Path(args.OUT)

    if in_path.exists():
        annots, imgs = read_mscoco(in_path)
        itoa = make_groups(annots)
        out_data = make_formatted(itoa, imgs)

    with open(out_path, 'wb') as f:
        pickle.dump(out_data, f, pickle.HIGHEST_PROTOCOL)
