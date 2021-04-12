
import os
# BEFORE TF IMPORT
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

class CustomModel(tf.keras.Model):
	@tf.function
	def train_step(self, data):
		'''
		function perform forward and backpropagation on one batch
		- x: one batch of processed audio input
		- y: one batch of target transcript labels
		- optimizer: optimizer
		
		- return: loss from this step
		'''
		x, y = data
		with tf.GradientTape() as tape:
			labels = y
			logits = self(x)   # predict / forward prop
			
			print(f'x: {x}\nx.shape: {x.shape}')
			print(f'logits: {logits}\nlogits.shape: {logits.shape}')
			print(f'labels: {labels}\nlabels.shape: {labels.shape}')
			# Labels has no shape for some reason??
			print(f'logits.shape[1]: {logits.shape[1]}')
			print(f'logits.shape[0]: {logits.shape[0]}')
			print(f'labels.shape[1]: {labels.shape[1]}')
			print(f'labels.shape[0]: {labels.shape[0]}')
			logits_length = [logits.shape[1]]*logits.shape[0]
			labels_length = [labels.shape[1]]*labels.shape[0]
			ctc_loss = tf.nn.ctc_loss(labels=labels, logits=logits, label_length=labels_length, logit_length=logits_length, logits_time_major=False, unique=None, blank_index=-1, name=None)
			loss = tf.reduce_mean(ctc_loss)
		gradients = tape.gradient(loss, self.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # # Update metrics (includes the metric that tracks the loss)
        # self.compiled_metrics.update_state(y, y_pred)
        # # Return a dict mapping metric names to current value
        # return {m.name: m.result() for m in self.metrics}

		return loss