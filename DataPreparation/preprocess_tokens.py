"""
This module preprocess text corpus.

it pre-process corpus data into processed tokens.
mainly it tokenize sentences, lower characters, remove suffix and replace all digits with 0.
"""

import argparse
import collections
import pickle
import re
from itertools import dropwhile
from pathlib import Path
from tqdm import tqdm


class Tokenizer(object):
    """
    preprocess sentences into pieces.

    This class split sentences into morphs with
    prossesing of lower case, remove suffix and so on.

    Attributes
    ----------
    lang: str
        language to be processed.
    tokenize: bool
        tokenize sentences into pieces.
        set false if input sentence is already tokenized.
    to_lower: bool
        lower input sentences or not.
    remove_suffix: bool
        whether or not remove characters like [.,!?"'";:。、].
    replace_digits: bool
        whether or not replace digits with 0.
    removed_char: re
        regular expression to find
        some special characters that should be deleted
    split_digits: re
        regular expression to find digits
    """

    __slots__ = [
        'lang',
        'tokenize',
        'to_lower',
        'remove_suffix',
        'replace_digits',
        'removed_char',
        'split_digits',
        'ja_tokenizer',
        'segmenter'
    ]

    def __init__(
            self,
            lang='jp',
            tokenize=True,
            to_lower=True,
            remove_suffix=True,
            replace_digits=True
    ):
        """
        initialize parameters which is used for preprocessing sentences.

        Parameters
        ----------
        lang: str
            language to be processed.
            it should be one of ja, en, ch.
        tokenize: bool
            tokenize sentences into pieces.
            set false if input sentence is already tokenized.
        to_lower:
            lower input sentences or not.
        remove_suffix: bool
            whether or not remove characters like [.,!?"'";:。、].
        replace_digits: bool
            whether or not replace digits with 0.
        """
        self.lang = lang
        self.tokenize = tokenize
        self.to_lower = to_lower
        self.remove_suffix = remove_suffix
        self.replace_digits = replace_digits
        self.removed_char = re.compile(r'[.,!?"\'\";:。、]')
        self.split_digits = re.compile(r'\d')

        if self.tokenize:
            if lang == 'jp':
                from janome.tokenizer import Tokenizer
                self.ja_tokenizer = Tokenizer()
                self.segmenter = lambda sen: list(
                    token.surface for token in self.ja_tokenizer.tokenize(sen)
                )

            elif lang == 'ch':
                import jieba
                self.segmenter = lambda sen: list(jieba.cut(sen))

            elif lang == 'en':
                import nltk
                self.segmenter = lambda sen: list(nltk.word_tokenize(sen))

    def pre_process(self, sen):
        """
        pre_process sentences into pieces.

        This function pre-process sentences into tokens
        with lower_case, remove some characters and replace all digits into 0.

        Parameters
        ----------
        sen: str
            sentences to be processed.

        Returns
        -------
        self.segmenter(sen) or sen.split()
            if self.tokenize == True, then return
            tokenized sentences with some processing.
            if not, then return splitted sentences.
        """
        if self.to_lower:
            sen = sen.strip().lower()

        if self.remove_suffix:
            sen = self.removed_char.sub('', sen)

        if self.replace_digits:
            sen = self.split_digits.sub('0', sen)

        return self.segmenter(sen) if self.tokenize else sen.split()


def token2index(tokens, word_ids):
    """
    transform tokens into word_ids.

    Parameters
    ----------
    tokens: list
        list of tokens
    word_ids: list
        word_ids list

    Returns
    -------
    list of word ids
    """
    return [word_ids[token] if token in word_ids
            else word_ids['<UNK>'] for token in tokens]


def encode_captions(captions, word_index):
    '''encode captions into digits based on word_index'''
    for caption in tqdm(captions):
        caption['caption'] = token2index(caption['caption'], word_index)

    return captions


def load_pickle(in_file):
    """load pickle file."""
    in_path = Path(in_file)
    with in_path.open('rb') as f:
        row_data = pickle.load(f)
    return row_data


def save_pickle(in_file, out_file):
    """save pickle file."""
    out_path = Path(out_file)
    with out_path.open('wb') as f:
        pickle.dump(in_file, f, pickle.HIGHEST_PROTOCOL)


def create_captions(formatted_data, tokenizer):
    """
    separate image and captions from formatted data.
    and preprocess captions.

    Parameters
    ----------
    formatted_data: dict
        formated data preprocessed by mscoco2formatted.py
    tokenizer: class
        tokenizer to preprocess captions

    Returns
    -------
    captions: dict
        dict of captions which contains 'img_idx', 'caption', 'caption_idx'
        to identify each caption.
    images: dict
        dict of image information related to each captions.
        It contains 'file_path' and 'img_idx'.
    """

    img_idx = 0
    caption_idx = 0
    captions = []
    images = []

    word_counter = collections.Counter()

    # append each captions and images separately into captions and images.
    for img in tqdm(formatted_data):
        caption_type = 'tokenized_captions' if 'tokenized_captions' in img else 'captions'

        for caption in img[caption_type]:
            caption_tokens = ['<SOS>']
            caption_tokens += tokenizer.pre_process(caption)
            caption_tokens.append('<EOS>')
            captions.append(
                {'img_idx': img_idx,
                 'caption': caption_tokens,
                 'caption_idx': caption_idx}
            )
            caption_idx += 1

            # add each word in tokens into word_counter
            word_counter.update(caption_tokens)

        images.append(
            {'file_path': img['file_path'],
             'img_idx': img_idx}
        )
        img_idx += 1

    return captions, images, word_counter


def create_word_dict(word_counter, cutoff=5, vocab_size=False):
    '''
    create word dictionary

    Parameters
    ----------
    word_counter: collenctions.Counter
        Counter object returned from create_captions.
    cutoff: int
        cutoff to dispose words.l
    vocab_size: int
        designate vocabrary size saved in word dictionary.
        default value is set to False.
    '''
    word_ids = collections.Counter({
        '<UNK>': 0,
        '<SOS>': 1,
        '<EOS>': 2
    })

    # create word dictionary
    print("total distinct words:{0}".format(len(word_counter)))
    print('top 30 frequent words:')
    for word, num in word_counter.most_common(30):
        print('{0} - {1}'.format(word, num))

    # delete words less than cutoff
    for word, num in dropwhile(
            lambda word_num: word_num[1] >= cutoff, word_counter.most_common()
    ):
        del word_counter[word]

    # pick up words of vocab_size
    # minus 1 because unk is included.
    word_counter = word_counter.most_common(
        vocab_size-1 if vocab_size else len(word_counter)
    )

    for word, num in tqdm(word_counter):
        if word not in word_ids:
            word_ids[word] = len(word_ids)

    print('total distinct words more than {0} : {1}'.format(cutoff, len(word_counter)))

    return word_ids


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('INPUT', type=str,
                        help="input formatted file by mscoco2formatted.py")
    parser.add_argument('OUT_DATASET', type=str,
                        help="output file name")
    parser.add_argument('--in_vocab_path', type=str, default='',
                        help="input vocaburay name for encode \
                        captions used in validation and test dataset.")
    parser.add_argument('--out_vocab_path', type=str, default='',
                        help="output vocaburary dict name")
    parser.add_argument('--tokenize', action='store_true', default=False,
                        help='tokenize in_path file')
    parser.add_argument('--lang', type=str, choices=['jp', 'en', 'ch'],
                        help="language to be processed")
    parser.add_argument('--tolower', action='store_true', default=False,
                        help="lower all characters for all sentences.")
    parser.add_argument('--remove_suffix', action='store_true', default=False,
                        help="remove all suffix like ?,!.")
    parser.add_argument('--replace_digits', action='store_true', default=False,
                        help="replace digits to 0 for all sentences.")
    parser.add_argument('--cutoff', type=int, default=5,
                        help="cutoff words less than the number digignated")
    parser.add_argument('--vocab_size', type=int, default=0,
                        help='vocabrary size')
    args = parser.parse_args()

    # read files
    IN_PATH = Path(args.INPUT)
    OUT_PATH = Path(args.OUT_DATASET)

    TOKENIZER = Tokenizer(
        lang=args.lang,
        tokenize=args.tokenize,
        to_lower=args.tolower,
        remove_suffix=args.remove_suffix
    )

    FORMATTED_MSCOCO = load_pickle(IN_PATH)
    CAPTIONS, IMGS, WORD_COUNTER = create_captions(FORMATTED_MSCOCO, TOKENIZER)

    if args.in_vocab_path:
        WORD_INDEX = load_pickle(args.in_vocab_path)
    else:
        WORD_INDEX = create_word_dict(WORD_COUNTER, args.cutoff, args.vocab_size)

    CAPTIONS = encode_captions(CAPTIONS, WORD_INDEX)

    OUT_DATASET = {'images': IMGS, 'captions': CAPTIONS}

    save_pickle(OUT_DATASET, args.OUT_DATASET)

    if args.out_vocab_path:
        save_pickle(WORD_INDEX, args.out_vocab_path)
