import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import pandas as pd
import math
import csv
import matplotlib.pyplot as plt
from BeamSearch import ctcBeamSearch
from prefix_beam_search import prefix_beam_search
from string import ascii_lowercase

import editdistance
	
def load_df_from_tsv(path: str):
	return pd.read_csv(
        path,
        sep="\t",
        header=0,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
        na_filter=False,
    )


def save_df_to_tsv(dataframe, path):
    dataframe.to_csv(
        path,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )

def softmax(matrix):
	'transform input probabilities into a probabil'
	num_timesteps, len_alphabet = matrix.shape
	res = np.zeros(matrix.shape)		# init empty result list
	for timestep in range(num_timesteps):
		y = matrix[timestep, :]
		e = np.exp(y)
		res[timestep, :] = e/np.sum(e)
	return res

def calc_cer(ground_truth, pred_sequence):
	'''
	Calculates CER based on the ground truth and predicted sequence.
	-results (list): list of ground truth and predicted label pairs.

	Returns the CER for the full set.
	'''
	dist = editdistance.eval(ground_truth, pred_sequence)	# use online editdistance (levenshtein distance) to compare prediction against transcript
	max_chars = max(len(pred_sequence), len(ground_truth))		# the longer of the two sequences so that a predicted sequence of 1 doesn't have a deceivingly low error
	return dist / max_chars if dist else 0.0

def compare_decoders(space_token=' ', end_token='>', blank_token='?'):
	filename = 'ctc_output.npy'

	mat_tcb = np.load(filename)	# raw output t * c * b matrix from the network
	# my nn architecture worked on a t * c * b matrix, so I need to switch the dimensions to t * b * c for tensorflow

	# print('mat_tcb.shape: ', mat_tcb.shape)
	# print('tf.keras.backend.eval(mat_tcb): ', tf.keras.backend.eval(mat_tcb))

	mat_tc = mat_tcb[0] # compare decoders for only for 1 training example
	# print('mat_tc.shape: ', mat_tc.shape)
	# print('tf.keras.backend.eval(mat_tc): ', tf.keras.backend.eval(mat_tc))

	mat_sm_tc = softmax(mat_tc) # applies softmax
	# print('mat_sm_tc.shape: ', mat_sm_tc.shape)
	# print('tf.keras.backend.eval(mat_sm_tc): ', tf.keras.backend.eval(mat_sm_tc))

	# # compare Tensorflow softmax to custom softmax
	# mat_tf_sm_tc = tf.nn.softmax(mat_tc) # applies tf softmax
	# print('mat_tf_sm_tc.shape: ', mat_tf_sm_tc.shape)
	# print('tf.keras.backend.eval(mat_tf_sm_tc): ', tf.keras.backend.eval(mat_tf_sm_tc))


	# Decoding Parameters

	alphabet = list(ascii_lowercase) + [space_token, end_token, blank_token]
	print('Alphabet: ', alphabet)

	ground_truth = list('mister quilter is the apostle of the middle classes and we are glad to welcome his gospel')

	run_prefix = True				# Test my implementation
	run_extra_beam_search = True	# Test similar online beam search to compare use of language models when I get that working
	run_tensorflow = True			# Test tensorflow
	run_greedy = True				# Test simple greedy argmax of each timestep

	lm = None 		# no LM until I can get either fairseq or ASR library language model preprocessing working
	beam_width = 25

	if run_prefix:
		# My Prefix Beam Search
		print('\nRunning My Prefix Beam Search')
		
		my_label_string = prefix_beam_search(mat_sm_tc, alphabet, space_token, end_token, blank_token, lm=lm, beamWidth=beam_width)
		my_labels = list(my_label_string)
		print('My Prediction:', my_label_string)
		
		my_cer = calc_cer(ground_truth, my_labels)
		print("My CER: {:.3f}".format(my_cer))

	if run_tensorflow:
		# Tensorflow Beam Search
		print('\nRunning Tensorflow\'s Beam Search')
		
		tensor_matrix = tf.convert_to_tensor(mat_tcb) # tensorflow might need something else...
		# sequence_lengths = [mat_tcb.shape[0]]
		

		# print('mat_tcb.shape: ', mat_tcb.shape)
		(batch_size, sequence_max_len, num_classes) = mat_tcb.shape

		# tensor_matrix = tf.compat.v1.placeholder(tf.float32, shape=mat_tcb.shape)
		tensor_matrix_transposed = tf.transpose(tensor_matrix, perm=[1, 0, 2])  # TF expects dimensions [max_time, batch_size, num_classes]
		sequence_lengths = tf.dtypes.cast(tf.fill([batch_size], sequence_max_len), tf.int32)
		
		# depending on how the network outputs its data, might need to take log first
		# log_tensor_matrix_transposed = tf.math.log(tensor_matrix_transposed)
		# tensor_matrix = log_tensor_matrix_transposed
		tensor_matrix = tensor_matrix_transposed

		
		# print('sequence_lengths: ', sequence_lengths)
		# print('tensor_matrix: ', tensor_matrix)

		tokens, log_probs = tf.nn.ctc_beam_search_decoder(tensor_matrix, sequence_length=sequence_lengths, beam_width=beam_width)
		# TODO: calculate even better error rate using tokens and each corresponding probability
		# 		so that low-confidence tokens aren't penalized as much for being wrong
		
		tf_indecies = tokens[0].values.numpy()
		tf_labels = [alphabet[c] for c in tf_indecies]
		tf_label_string = ''.join(tf_labels)

		print('TF Prediction:', tf_label_string)

		tf_cer = calc_cer(ground_truth, tf_labels)
		print("TF CER: {:.3f}".format(tf_cer))

	
		# Compare My Decoder against Tensorflow's
		cer = calc_cer(tf_labels, my_labels)
		print("\nMy CER with TF prediction used as ground truth: {:.3f}".format(cer))


	
	if run_extra_beam_search:
		# Extra Beam Search From Online for use with an LM to compare against mine in the future
		print('\nRunning Extra\'s  Beam Search')

		extra_label_string = ctcBeamSearch(mat_sm_tc, classes=alphabet[:-1], beamWidth=beam_width, lm=lm)
		extra_labels = list(extra_label_string)
		print('Extra Beam Search Prediction:', extra_label_string)

		extra_cer = calc_cer(ground_truth, extra_labels)
		print("Extra CER: {:.3f}".format(extra_cer))

		# Compare My Decoder against Extra's
		cer = calc_cer(extra_labels, my_labels)
		print("\nMy CER with Extra Beam Search prediction used as ground truth: {:.3f}".format(cer))

	if run_greedy:
		# greedy decoding
		print('\nRunning Argmax\'s  Decoding')
		mat_tc_log_softmax = tf.nn.log_softmax(mat_tc)  # logsoftmax = logits - log(reduce_sum(exp(logits), axis))
		greedy_label_string = ''
		for timestep in mat_tc_log_softmax:
			greedy_label_string += alphabet[tf.math.argmax(timestep)]
		
		greedy_label_string = greedy_label_string.replace(blank_token, '')	# remove all blank tokens
		greedy_labels = list(greedy_label_string)
		print('Greedy Prediction:', greedy_label_string)

		greedy_cer = calc_cer(ground_truth, greedy_labels)
		print("Greedy CER: {:.3f}".format(greedy_cer))

	# plot Incoming Probability matrix to show the parts of the prediction that are high confidence
	plt.imshow(mat_sm_tc.transpose(), extent= [0, mat_sm_tc.shape[0], 0, len(alphabet)], aspect=3.0)
	plt.xlabel('time')
	plt.ylabel('chars')
	plt.show()

if __name__ == '__main__':
	compare_decoders()
