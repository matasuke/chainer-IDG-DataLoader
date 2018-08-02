#bin/sh

# pre-process captions

# convert all captions into formatted one.
python  DataPreparation/mscoco2formatted.py \
    data/captions/original/MSCOCO_captions_en/captions_train2014.json \
    data/captions/formatted/MSCOCO_captions/captions_formatted_train2014.pkl

python DataPreparation/mscoco2formatted.py \
    data/captions/original/MSCOCO_captions_en/captions_val2014.json \
    data/captions/formatted/MSCOCO_captions/captions_formatted_val2014.pkl

python DataPreparation/mscoco2formatted.py \
    data/captions/original/STAIR_captions/stair_captions_v1.2_train.json \
    data/captions/formatted/STAIR_captions/STAIR_captions_formatted_train2014.pkl

python DataPreparation/mscoco2formatted.py \
    data/captions/original/STAIR_captions/stair_captions_v1.2_val.json \
    data/captions/formatted/STAIR_captions/STAIR_captions_formatted_val2014.pkl

# tokenize and separate image paths and captions.
# output dataset can be loaded by IDGDataloader

python DataPreparation/preprocess_tokens.py \
    data/captions/formatted/MSCOCO_captions/captions_formatted_train2014.pkl \
    data/captions/converted/MSCOCO_captions/train2014.pkl \
    --out_vocab_path data/vocab/mscoco_train2014_vocab.pkl \
    --tokenize \
    --lang en \
    --tolower \
    --remove_suffix \
    --replace_digits \
    --cutoff 5

python DataPreparation/preprocess_tokens.py \
    data/captions/formatted/MSCOCO_captions/captions_formatted_val2014.pkl \
    data/captions/converted/MSCOCO_captions/val2014.pkl \
    --in_vocab_path data/vocab/mscoco_train2014_vocab.pkl \
    --tokenize \
    --lang en \
    --tolower \
    --remove_suffix \
    --replace_digits \
    --cutoff 5

python DataPreparation/preprocess_tokens.py \
    data/captions/formatted/STAIR_captions/STAIR_captions_formatted_train2014.pkl \
    data/captions/converted/STAIR_captions/train2014.pkl \
    --out_vocab_path data/vocab/STAIR_train2014_vocab.pkl \
    --tokenize \
    --lang jp \
    --tolower \
    --remove_suffix \
    --replace_digits \
    --cutoff 5

python DataPreparation/preprocess_tokens.py \
    data/captions/formatted/STAIR_captions/STAIR_captions_formatted_val2014.pkl \
    data/captions/converted/STAIR_captions/val2014.pkl \
    --in_vocab_path data/vocab/STAIR_train2014_vocab.pkl \
    --tokenize \
    --lang jp \
    --tolower \
    --remove_suffix \
    --replace_digits \
    --cutoff 5
