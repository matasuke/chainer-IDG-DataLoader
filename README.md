# chainer-IDG-DataLoader
Chainer data loader for Image Captionning tasks using chainer.dataset.DatasetMixin.

Data Preprocesser for MSCOCO Captioning Dataset is also included.
But you can use other dataset only if you pre-process them beforehand.
This IDG data loader is especially for [MSCOCO Image Captioning Dataset](http://cocodataset.org/#download)

## Requirements
'''
pip install -r requirements.txt
'''

## Usage

### Clone the repository
'''
git clone https://github.com/matasukef/chainer-IDG-DataLoader
cd chainer-IDG-DataLoader
'''

### Download nltk tokenizer
'''
python
import nltk
nltk.download('punkt')
'''

### Download Images and Captions.
You just need to use shells/download.sh.
This script download MSCOCO Images and captions to 'data' directory.
For Japanese Users, STAIR caption is also available.

'''
sh shells/download.sh
'''

### Preprocess dataset.
create converted dataset that is easy to use in chainer.dataset.mixin from original dataset.
in the script, mscoco2formatted.py and preprocess_tokens.py are used.
mscoco2formatted.py is used for converting original caption dataset into formatted one.
then preprocess_tokens.py is used for tokenize captions and relate each image and captions.
And create vocabulary dictionary used in captions.

'''
sh shells/pre_process.sh
'''

### Check DataLoader
For usage, please see [example.ipynb](https://github.com/matasukef/chainer-IDG-DataLoader/blob/master/example.ipynb)
