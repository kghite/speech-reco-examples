"""
Data manager for pytorch speech recognition model implementation
"""

# External
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import numpy as np


"""
char <-> int mapping
"""
class TextTransform:
	def __init__(self):
		char_map_str = """
		' 0
		<SPACE> 1
		a 2
		b 3
		c 4
		d 5
		e 6
		f 7
		g 8
		h 9
		i 10
		j 11
		k 12
		l 13
		m 14
		n 15
		o 16
		p 17
		q 18
		r 19
		s 20
		t 21
		u 22
		v 23
		w 24
		x 25
		y 26
		z 27
		"""
		self.char_map = {}
		self.index_map = {}
		for line in char_map_str.strip().split('\n'):
			ch, index = line.split()
			self.char_map[ch] = int(index)
			self.index_map[int(index)] = ch
		self.index_map[1] = ' '

	"""
	Convert text seq to int labels
	"""
	def text_to_int(self, text):
		int_sequence = []
		for c in text:
			if c == ' ':
				ch = self.char_map['<SPACE>']
			else:
				ch = self.char_map[c]
			int_sequence.append(ch)
		return int_sequence

	"""
	Convert int labels to text seq 
	"""
	def int_to_text(self, labels):
		string = []
		for i in labels:
			string.append(self.index_map[i])
		return ''.join(string).replace('<SPACE>', ' ')


train_audio_transforms = nn.Sequential(
	torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
	torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
	torchaudio.transforms.TimeMasking(time_mask_param=35)
)
valid_audio_transforms = torchaudio.transforms.MelSpectrogram()
text_transform = TextTransform()


def data_processing(data, data_type='train'):
	spectrograms = []
	labels = []
	input_lengths = []
	label_lengths = []
	for (waveform, _, utterance, _, _, _) in data:
		if data_type == 'train':
			spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
		else:
			spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
		spectrograms.append(spec)
		label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
		labels.append(label)
		input_lengths.append(spec.shape[0]//2)
		label_lengths.append(len(label))

	spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
	labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

	return spectrograms, labels, input_lengths, label_lengths


def GreedyDecoder(output, labels, label_lengths, blank_label=28, collapse_repeated=True):
	arg_maxes = torch.argmax(output, dim=2)
	decodes = []
	targets = []
	for i, args in enumerate(arg_maxes):
		decode = []
		targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
		for j, index in enumerate(args):
			if index != blank_label:
				if collapse_repeated and j != 0 and index == args[j -1]:
					continue
				decode.append(index.item())
		decodes.append(text_transform.int_to_text(decode))
	return decodes, targets

def avg_wer(wer_scores, combined_ref_len):
	return float(sum(wer_scores)) / float(combined_ref_len)


"""
Measure seq difference
"""
def _levenshtein_distance(ref, hyp):
	m = len(ref)
	n = len(hyp)

	# special case
	if ref == hyp:
		return 0
	if m == 0:
		return n
	if n == 0:
		return m

	if m < n:
		ref, hyp = hyp, ref
		m, n = n, m

	# use O(min(m, n)) space
	distance = np.zeros((2, n + 1), dtype=np.int32)

	# initialize distance matrix
	for j in range(0,n + 1):
		distance[0][j] = j

	# calculate levenshtein distance
	for i in range(1, m + 1):
		prev_row_idx = (i - 1) % 2
		cur_row_idx = i % 2
		distance[cur_row_idx][0] = i
		for j in range(1, n + 1):
			if ref[i - 1] == hyp[j - 1]:
				distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
			else:
				s_num = distance[prev_row_idx][j - 1] + 1
				i_num = distance[cur_row_idx][j - 1] + 1
				d_num = distance[prev_row_idx][j] + 1
				distance[cur_row_idx][j] = min(s_num, i_num, d_num)

	return distance[m % 2][n]


"""
Compute word level levenshtein
"""
def word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):
	if ignore_case == True:
		reference = reference.lower()
		hypothesis = hypothesis.lower()

	ref_words = reference.split(delimiter)
	hyp_words = hypothesis.split(delimiter)

	edit_distance = _levenshtein_distance(ref_words, hyp_words)
	return float(edit_distance), len(ref_words)


"""
Compute char level levenshtein
"""
def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
	if ignore_case == True:
		reference = reference.lower()
		hypothesis = hypothesis.lower()

	join_char = ' '
	if remove_space == True:
		join_char = ''

	reference = join_char.join(filter(None, reference.split(' ')))
	hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

	edit_distance = _levenshtein_distance(reference, hypothesis)
	return float(edit_distance), len(reference)


"""
Calculate word error rate
"""
def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
	edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case,
										 delimiter)

	if ref_len == 0:
		raise ValueError('Reference\'s word number should be greater than 0.')

	wer = float(edit_distance) / ref_len
	return wer


"""
Calculate charactor error rate
"""
def cer(reference, hypothesis, ignore_case=False, remove_space=False):
	edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
										 remove_space)

	if ref_len == 0:
		raise ValueError('Length of reference should be greater than 0.')

	cer = float(edit_distance) / ref_len
	return cer