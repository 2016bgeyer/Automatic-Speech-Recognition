from __future__ import division
from __future__ import print_function

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

def softmax(mat):
	"calc softmax such that labels per time-step form probability distribution"
	maxT, _ = mat.shape # dim0=t, dim1=c
	res = np.zeros(mat.shape)
	for t in range(maxT):
		y = mat[t, :]
		e = np.exp(y)
		s = np.sum(e)
		res[t, :] = e/s
	return res

def compute_cer(results):
    """
    Arguments:
        results (list): list of ground truth and
            predicted sequence pairs.

    Returns the CER for the full set.
    """
    dist = sum(editdistance.eval(label, pred)
                for label, pred in results)
    total = sum(len(label) for label, _ in results)
    return dist / total if total else 0.0

def compare_decoders(space_token='_', end_token='>', blank_token='%'):
	mat_tbc = np.load('data/testing/test_matrix.npy')
	mat_sm_tc = softmax(mat_tbc[:,0,:])
	print('mat_tbc.shape: ', mat_tbc.shape)
	print('mat_sm_tc.shape: ', mat_sm_tc.shape)
	alphabet = list(ascii_lowercase) + [space_token, end_token, blank_token]
	print('planned ascii alphabet: ', alphabet)
	if (mat_sm_tc.shape[1]-1 > len(alphabet)): 	# bad input matrix has more classes than expected so just start alphabet at capital letters
		print('bad input matrix class size is bigger than alphabet')
		alphabet = [chr(65+i) for i in range(mat_sm_tc.shape[1]-1)]
	print('Alphabet: ', alphabet)


	# # Word Beam Search
	# fake_classes = [chr(65+i) for i in range(mat_sm_tc.shape[1]-1)]
	# label_string = ctcBeamSearch(mat_sm_tc, classes=alphabet, beamWidth=25, lm=None)
	# my_labels = list(label_string)
	# my_indecies = [alphabet.index(c) for c in label_string]
	# print('My indecies:', my_indecies)
	# print('My result:', my_labels)

	# My Prefix Beam Search
	
	lm = None 		# no LM until I can get either fairseq processing working
	label_string = prefix_beam_search(mat_sm_tc, alphabet, space_token, end_token, blank_token, lm=lm, beamWidth=25)
	my_labels = list(label_string)
	my_indecies = [alphabet.index(c) for c in label_string]
	print('My indecies:', my_indecies)
	print('My result:', my_labels)

	# Tensorflow
	tensor_matrix = tf.convert_to_tensor(mat_tbc)
	tokens, log_probs = tf.nn.ctc_beam_search_decoder(tensor_matrix, [mat_tbc.shape[0]], beam_width=25)
	
	# print(tf.executing_eagerly())
	tf_indecies = tokens[0].values.numpy()
	tf_labels = [alphabet[c] for c in tf_indecies]


	print('TF indecies:', tf_indecies)
	print('TF result:', tf_labels)
	print('Is equal:', tf_labels==my_labels)

	ground_truth = tf_labels
	cer = compute_cer(zip(ground_truth, my_labels))
	print("CER with TF as ground truth: {:.3f}".format(cer))

	# plot matrix
	plt.imshow(mat_sm_tc)
	plt.xlabel('chars')
	plt.ylabel('time')
	plt.show()

if __name__ == '__main__':
	compare_decoders()
