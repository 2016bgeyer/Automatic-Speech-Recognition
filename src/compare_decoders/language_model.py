import kenlm
import math
import pickle
from collections import defaultdict
class LanguageModel:

    def score(self, sequence: str) -> float:
        pass


class KenLMLanguageModel(LanguageModel):
	'''
	Initializes the language model.

	Args:
		lm_file (str): Path to dictionary mapping between prefixes and lm probabilities. 
		use_log (bool): Whether to use log progabilities or not
	'''
	def __init__(self, lm_file: str, use_log=True):
		self._pickle = lm_file.endswith('p')
		self.use_log = use_log
		if lm_file.endswith('.p'):
			lm =  pickle.load(open(lm_file, 'rb'))
			self.model = defaultdict(lambda: 1e-11, lm)
		elif lm_file.endswith('.arpa'):
			self.model = kenlm.LanguageModel(lm_file)
		else:
			raise Exception('Invalid Lanuage Model File')


	def score(self, sequence: str) -> float:
		if self._pickle:
			score = self.model.score(self._preprocess(sequence))
			return score if self.use_log else math.exp(score)
			# return self._model[prefix]
		else:
			scores = self.model.full_scores(sequence, bos=True, eos=True)
			scores_list = list(scores)
			# print(f'sequence: {sequence}')
			# print(f'scores_list: {scores_list}')
			# print(f'len(scores_list): {len(scores_list)}')
			results = 10 ** scores_list[-1][0]
			print(f'results: {results}')
			exp_results = math.exp(scores_list[-1][0])
			print(f'exp_results: {exp_results}')
			two_results = 2 ** scores_list[-1][0]
			print(f'two_results: {two_results}')
			return exp_results
			

	def _preprocess(self, sequence) -> str:
		return ' '.join(sequence.replace(' ', '@'))		# might not want to do this actually

