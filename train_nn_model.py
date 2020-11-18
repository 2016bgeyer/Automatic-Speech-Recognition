import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Conv2D, LSTM, Bidirectional, Dense
from string import ascii_lowercase
import math
import librosa

class Model(tf.keras.Model):
    ''' The output is a t * c * b matrix with probabilities for each character t each timestep for each minibatch b '''
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

def preprocess_audio(file_name, resample_to=8000):  # 
    '''
    Uses Librosa and Tensorflow to read in the file and use STFT to sample the file
    with moving window and generate a spectrogram to return to the network to train on.
    - file_name: path to the audio file
    - resample_to:  to cut features in half and speed up training but with less accuracy, use a smaller (not really tested)
    
    - return: spectrogram corresponding to the input file
    '''
    # load the signals and resample them
    signal, sample_rate = librosa.core.load(file_name)
    print('Debugging Preprocessing\nSample signal: {}\nSample sample_rate: {}'.format(signal, sample_rate))
    if signal.shape[0] == 2:
        signal = np.mean(signal, axis=0)
    signal_resampled = librosa.core.resample(signal, sample_rate, resample_to)  # resample the audio file if needed

    # create spectrogram
    stfts = tf.signal.stft(signal_resampled, frame_length=200, frame_step=80, fft_length=256)
    spectrogram = tf.math.pow(tf.abs(stfts), 0.5)

    # normalization
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    norm_spectrogram = tf.divide(tf.subtract(spectrogram, means), stddevs)   # normalize to 0 mean with magnitude being normalized via std dev
    return norm_spectrogram


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
        # print('x: ', x)
        logits = model(x)
        labels = y
        logits_length = [logits.shape[1]]*logits.shape[0]
        labels_length = [labels.shape[1]]*labels.shape[0]
        # print('logits_length: ', logits_length)
        # print('labels_length: ', labels_length)
        # print('logits: ', logits)
        # print('labels: ', labels)
        ctc_loss = tf.nn.ctc_loss(labels=labels, logits=logits, label_length=labels_length, logit_length=logits_length, logits_time_major=False, unique=None, blank_index=-1, name=None)
        # print('original ctc_loss', tf.keras.backend.eval(ctc_loss))
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

def train_network(X, y, epochs, batch_size, test_X, alphabet, save_model_output):

    # MODEL TRAINING
    model = Model(filters=200, kernel_size=11, conv_stride=2, conv_border='valid', n_lstm_units=200, n_dense_units=len(alphabet), optimizer=tf.keras.optimizers.Adam())
    
    train(model, X, y, epochs, batch_size)

    # getting the ctc output which is negative log probabilities of each character at each timestep for each training sample
    ctc_output = model(test_X)
    ctc_output_log_softmax = tf.nn.log_softmax(ctc_output)  # logsoftmax = logits - log(reduce_sum(exp(logits), axis))
    # ctc_output_softmax = tf.nn.softmax(ctc_output) 


    print('saving ctc output to : ctc_output.npy', tf.keras.backend.eval(ctc_output))
    # print('log_softmax: ', tf.keras.backend.eval(ctc_output_log_softmax))
    # print('softmax: ', tf.keras.backend.eval(ctc_output_softmax))

    # save model output to npy and then load that in compare_decoders
    if save_model_output:
        np.save('ctc_output.npy', ctc_output)
        # np.save('ctc_output_softmax.npy', ctc_output_softmax)
        # np.save('ctc_output_log_softmax.npy', ctc_output_log_softmax)
        
    

    # greedy decoding
    output_text = ''
    for timestep in ctc_output_log_softmax[0]:
        output_text += alphabet[tf.math.argmax(timestep)]
    print('Example Greedy Decoding of the file after training: ', output_text)

def main():

    sample_file_name = 'sample.wav'
    transcript = 'mister quilter is the apostle of the middle classes and we are glad to welcome his gospel'
    
    ################################################################
    # If these are changed, change them in compare_decoders as well!
    ################################################################
    space_token = ' ' # "\u2581" encoded in utf-8 but underscore in ascii
    end_token = '>'     # <\s> in fairseq <eos> in other places
    blank_token = '?'   # common blank token
    alphabet = list(ascii_lowercase) + [space_token, end_token, blank_token] # append spaces, end token (for fairseq), and blank_token for separating repeating characters

    save_model_output = True


    X = preprocess_audio(sample_file_name)  # process audio into ndarray
    X = tf.expand_dims(X, axis=0)           # converting input into a batch of size 1; when doing a multi-sample training set, the expand_dims isn't needed
    print('X: ', tf.keras.backend.eval(X))
    

    y = [alphabet.index(char) for char in transcript] # convert to character indecies in alphabet

    y = tf.expand_dims(tf.convert_to_tensor(y), axis=0)  # converting output to a batch of size 1  # when doing a bigger set, the expand_dims isn't needed
    # print('Input shape: {}'.format(X.shape))
    # print('Target shape: {}'.format(y.shape))

    test_X = X      # test sample to have the network predict on to save and decode. Using the same file is fine since we are evaluating decoders not the network

    epochs = 75 # after about 75, it has completely overfit to the data for 1 file
    batch_size = 1 # just use the entire audio file so that the bidirectional LSTM can have context for prediction
    
    
    train_network(X, y, epochs, batch_size, test_X, alphabet, save_model_output)
    

if __name__ == '__main__':
    main()