import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tf.keras.layers import Conv1D, Conv2D, LSTM, Bidirectional, Dense
from string import ascii_lowercase
import math
import librosa

class Model(tf.keras.Model):
    '''
    Class for defining the end to end ASR model.
    This model consists of a 1D convolutional embedding layer followed by a bidirectional LSTM
    followed by a fully connected layer applied at each timestep.
    The output is a t x c matrix with probabilities for each character t each timestep
    '''
    def __init__(self, filters, kernel_size, conv_stride, conv_border, n_lstm_units, n_dense_units, optimizer):
        super(Model, self).__init__()
        self.conv_layer = Conv1D(filters, kernel_size, strides=conv_stride, padding=conv_border, activation='relu')
        self.blstm_layer = Bidirectional(LSTM(n_lstm_units, return_sequences=True, activation='tanh'))
        self.dense_layer = Dense(n_dense_units)
        self.optimizer = optimizer

    def call(self, x):
        x = self.conv_layer(x)
        x = self.blstm_layer(x)
        x = self.dense_layer(x)
        return x

def create_spectrogram(signals):
    '''
    function to create spectrogram from signals loaded from an audio file
    - signals:
    
    - return:
    '''
    stfts = tf.signal.stft(signals, frame_length=200, frame_step=80, fft_length=256)
    spectrograms = tf.math.pow(tf.abs(stfts), 0.5)
    return spectrograms


def generate_input_from_audio_file(path_to_audio_file, resample_to=8000): # maybe change to 16k?
    '''
    function to create input for our neural network from an audio file.
    The function loads the audio file using librosa, resamples it, and creates spectrogram form it
    - path_to_audio_file: path to the audio file
    - resample_to:
    
    - return: spectrogram corresponding to the input file
    '''
    # load the signals and resample them
    signal, sample_rate = librosa.core.load(path_to_audio_file)
    print('Debugging Preprocessing\nSample signal: {}\nSample sample_rate: {}'.format(signal, sample_rate))
    if signal.shape[0] == 2:
        signal = np.mean(signal, axis=0)
    signal_resampled = librosa.core.resample(signal, sample_rate, resample_to)

    # create spectrogram
    X = create_spectrogram(signal_resampled)

    # normalization
    means = tf.math.reduce_mean(X, 1, keepdims=True)
    stddevs = tf.math.reduce_std(X, 1, keepdims=True)
    X = tf.divide(tf.subtract(X, means), stddevs)
    return X


def generate_target_output_from_text(target_text, alphabet):
    '''
    Target output is an array of indices for each character in your string.
    The indices comes from a mapping that will
    be used while decoding the ctc output.
    - target_text: (str) target string
    
    - return: array of indices for each character in the string
    '''
    char_to_index = {}
    for idx, char in enumerate(alphabet):
        char_to_index[char] = idx

    y = []
    for char in target_text:
        print(char_to_index[char])
        print(alphabet.index(char))
        y.append(char_to_index[char])
    
    print('Debugging Target Output:\nLength: {}\n{}'.format(len(y), y))
    return y


def step(x, y, model):
    '''
    function perform forward and backpropagation on one batch
    - x: one batch of input
    - y: one batch of target
    - optimizer: optimizer
    - model: object of the ASR class
    
    - return: loss from this step
    '''
    with tf.GradientTape() as tape:
        logits = model(x)
        labels = y
        logits_length = [logits.shape[1]]*logits.shape[0]
        labels_length = [labels.shape[1]]*labels.shape[0]
        ctc_loss = tf.nn.ctc_loss(labels=labels, logits=logits, label_length=labels_length, logit_length=logits_length, logits_time_major=False, unique=None, blank_index=-1, name=None)
        loss = tf.reduce_mean(ctc_loss)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def train(model, X, Y, epochs, batch_size):
    '''
    function to train the model for given number of epochs
    - model: object of class Model
    - optimizer: optimizer
    - X:
    - Y:
    - epochs:
    - batch_size:
    
    - return: None
    '''
    batch_per_epoch = math.floor(len(X) / batch_size)
    for epoch in range(epochs):
        print('Epoch: ', epoch)
        for batch_num in range(batch_per_epoch):
            n = batch_num*batch_size
            loss = step(X[n: n+batch_size], Y[n: n+batch_size], model)
            print('Batch {}, Loss: {}'.format(batch_num, loss))


if __name__ == '__main__':
    sample_call = 'sample.wav'
    transcript = 'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'.lower()
    space_token = '_' # "\u2581" encoded in utf-8 but underscore in ascii
    end_token = '>'     # <\s> in fairseq
    blank_token = '%'
    alphabet = list(ascii_lowercase) + [space_token, end_token, blank_token]

    X = generate_input_from_audio_file(sample_call)
    print('Before expanded Input shape: {}'.format(X.shape))
    X = tf.expand_dims(X, axis=0)  # converting input into a batch of size 1  # when doing a bigger set, the expand_dims isn't needed
    y = generate_target_output_from_text(transcript, alphabet)
    y = tf.expand_dims(tf.convert_to_tensor(y), axis=0)  # converting output to a batch of size 1  # when doing a bigger set, the expand_dims isn't needed
    print('Input shape: {}'.format(X.shape))
    print('Target shape: {}'.format(y.shape))

    model = Model(200, 11, 2, 'valid', 200, 29, tf.keras.optimizers.Adam())
    train(model, X, y, 50, 1)

    # getting the ctc output
    test_X = X
    ctc_output = model(test_X)
    np.save('ctc_output_raw.npy', ctc_output)
    # save model output to npy and then load that in compare_decoders
    ctc_output_log_softmax = tf.nn.log_softmax(ctc_output)
    np.save('ctc_output_log_softmax.npy', ctc_output_log_softmax)

    # greedy decoding
    output_text = ''
    for timestep in ctc_output_log_softmax[0]:
        output_text += alphabet[tf.math.argmax(timestep)]
    print(output_text)
    print('\n\nNote: Applying a good decoder on this output will give you readable output')
