# Automatic-Speech-Recognition
End-to-End Automatic Speech Recognition Using Connectionist Temporal Classification

This code uses TensorFlow 2.3.1

Files:  
`train_nn_model.py`:  
* neural network training on a sample .wav  and its transcript as input using TF CTC loss
  
`prefix_beam_search.py`:  
* code for prefix beam search
* you can import the function from this file directly and use it on your ctc output  
```
from prefix_beam_search import prefix_beam_search
example_ctc_output = None  # get your ctc output from the network
alphabet = list(ascii_lowercase) + [space_token, end_token, blank_token]  # get your character vocab
lm = None  
print(prefix_beam_search(example_ctc, alphabet, blank_token, end_token, space_token, lm=lm))
```

Datasets:

`LibriSpeech`:
http://www.openslr.org/12

`LibriSpeech Language Models`:
http://www.openslr.org/11