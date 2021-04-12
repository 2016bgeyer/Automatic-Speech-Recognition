# Prefix Beam Search Decoding
End-to-End Automatic Speech Recognition Using Connectionist Temporal Classification via Prefix Beam Search Decoding

Note: This code uses TensorFlow 2.3+

# Setup
* Each folder has their own requirements to run the contained code.  Most use Tensorflow and all use Python 3.6+.  One or two have instructions for conda if you would rather not interfere with your local environents.

# Folders:  
`src`:
* Data for end_to_end only (others will need the data moved slightly)

`data`:a
* Example data for end_to_end only (others will need the data paths changed/moved slightly or have their own sample.wav)
* small_data contains a small sample dataset to test end_to_end on using main_small.py
* If you download the full train-clean-100, dev-clean, and test-clean datasets for testing, extract the folders here to data/train-clean-100/.  You can change these paths in main_big.py.

`src/end_to_end`:
* My fully custom data processing pipeline which unfortunately has some issues with the current model architecture.

`src/compare_decoders`:
* Run `compare_decoders.py` to evaluate multiple ctc decoders on trained output from a network in `train_nn.py`.

`src/Automatic-Speech-Recognition`:
* I used part of an online Python and Tensorflow library [Automatic-Speech-Recognition](https://github.com/rolczynski/Automatic-Speech-Recognition) to save the preprocessed audio files, trained models, and generate language models.  However, this library didn't have the Decoder pipline fully implemented properly so I decided to fork it and make my own version to implement it. This is under Automatic-Speech-Recognition/ with an example test file being bens_evaluation.py (just run it).


# Datasets:

`LibriSpeech ASR Corpus`:
http://www.openslr.org/12

* I use the 100, dev, clean sets (use the dev for training instead of 100 if you want speed).


`LibriSpeech Language Models`:
http://www.openslr.org/11

* I use the 3-gram pruned arpa file LM.  Smaller ones are fine, paths will just have to be adjusted


# Other Notes

* I also initially tried using Facebook's PyTorch ASR library [Fairseq](https://github.com/pytorch/fairseq) to overcome similar problems with preprocessing and evaluation but had limited success due to issues with PyTorch not being optimized for Windows use. It would have been quite nice to use since it used Google's [SentencePiece](https://github.com/google/sentencepiece) subwword Tokenizer library for training it's Language Models.  This allows up to 4-gram word and subword dictionaries to be used for decoding which can greatly increase the efficacy of the prefix beam search algorithm as well as other decoding strategies such as Word Beam Search.