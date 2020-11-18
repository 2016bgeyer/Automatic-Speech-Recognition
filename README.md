# Prefix Beam Search Decoding
End-to-End Automatic Speech Recognition Using Connectionist Temporal Classification via Prefix Beam Search Decoding

Note: This code uses TensorFlow 2.1+

# Setup
Install required libraries from `environment.yml` manually or via [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#)
```
conda env create -f environment.yml
conda activate Custom-ASR
```
Then you can run any of the following files from the command line inside the anaconda environment.

# Files:  
`train_nn_model.py`:  
* This file trains the neural network on a sample .wav file and its transcript for 75 epochs
* Then, it feeds the same file into the network to get an activation matrix prediction to be fed to a decoder to generate a predicted transcript
* Saves this outputted matrix to `ctc_output.npy`
* You can change any of the model or training parameters in the main() method at the bottom of the file
* This should work for multiple audio files with only a little tweaking.
  
`prefix_beam_search.py`:  
* This is my implementation of my prefix beam search decoding with an optional language model. This takes the timestep * character matrix output from `train_nn_model.py` and return a most likely predicted transcription.
* Called and evaluated in compare_decoders.py 

`compare_decoders.py`:
* Code to run multiple decoders (Mine, Tensorflow CTC_Beam_Search, an online Python Beam Search algorithm, and a simple greedy argmax decoding) and evaluate them on the output .npy file from train_nn_model.py
* Returns Character Error Rate (CER) using the common Levenshtein distance mtric for the predicted vs ground truth transcript for each decoder and compares my decoder to the others with theirs as the ground truth model.

`environment.yml`:
* List of libraries and versions to install to run all code
* Conda should install all of these for you if you also have pip installed.

`ctc_output.npy`:
* The saved neural network matrix output resulting in running `train_nn_model.py` with epochs = 75 for your convenience.
* Once you run `train_nn_model.py`, this file will be overwritten.

`ctc_output_50_epochs.npy`:
* Saved neural network matrix output resulting in running `train_nn_model.py` with epochs = 50.
* In this case, the network wasn't fully trained and every decoding algorithm had an error around 20%.

`ctc_output_75_epochs.npy`:
* Saved neural network matrix output resulting in running `train_nn_model.py` with epochs = 75 as a backup in case you run `train_nn_model.py` with a different epoch number.
* In this case, the network was more trained and each decoding algorithm except greedy was able to achieve a very low error rate (~1%).

`ctc_output_200_epochs.npy`:
* Saved neural network matrix output resulting in running `train_nn_model.py` with epochs = 200.
* In this case, the network has overfitted and allows the decoders except Greedy to all get 0 error since each character has very high confidence.

`BeamSearch.py`:
* A Beam Search python implementation from [CTCDecoder](https://github.com/githubharald/CTCDecoder/blob/master/src/BeamSearch.py) by Harald Scheidl, a researcher in Computer Vision who has reserached Optical Character Recognition using CTC Beam Search.  This file is for me to compare my algorithm against when I have a working language model to plug in.

# Datasets:

`LibriSpeech ASR Corpus`:
http://www.openslr.org/12

`LibriSpeech Language Models`:
http://www.openslr.org/11

* I was unable to fully use these for training in my example
* These were used by Automatic-Speech-Recognition and Fairseq to generate language models for use in CTC Decoding

# Other Notes
* I also tried using part of an online Python and Tensorflow library [Automatic-Speech-Recognition](https://github.com/rolczynski/Automatic-Speech-Recognition) to save the preprocessed audio files, trained models, and generate language models.  I am trying to use it with my Prefix Beam Search implementation to fully evaluate my algorithm on a range of the datasets and learned language models.  
 
    However, this library didn't have the Decoder pipline fully implemented properly so I decided to fork it and make my own version which I have worked on for several days but couldn't get working fully yet.   I am still having issues with some Tensorflow bugs in it so I haven't submitted the code here.  I have instead pushed it to my [GitHub](https://github.com/2016bgeyer/Automatic-Speech-Recognition/tree/fix-decoding-wip) if you want to check it out anyways.

    I talked with the TA on monday because I was still trying to get this library to work and evaluate my code properly with language models but she said to just submit what I had and fix it for Milestone 3. Because of that, I had to spend a little time going back and adding more testing to my original algorithm to evaluate it and compare it against other implementations. I apoligize for the delay on submitting this.
* I also initially tried using Facebook's PyTorch ASR library [Fairseq](https://github.com/pytorch/fairseq) to overcome similar problems with preprocessing and evaluation but had limited success due to issues with PyTorch not being optimized for Windows use. It would have been quite nice to use since it used Google's [SentencePiece](https://github.com/google/sentencepiece) subwword Tokenizer library for training it's Language Models.  This allows up to 4-gram word and subword dictionaries to be used for decoding which can greatly increase the efficacy of the prefix beam search algorithm as well as other decoding strategies such as Word Beam Search.