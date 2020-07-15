"""
Train pytorch speech recognition model implementation
"""

# External
import os
import time
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import numpy as np

# Internal
from model import *
from data_manager import *


"""
Iteration tracking util
"""
class IterMeter(object):

	def __init__(self):
		self.val = 0

	def step(self):
		self.val += 1

	def get(self):
		return self.val


def train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter):
	print('\nTraining ...')
	model.train()
	data_len = len(train_loader.dataset)
	start_time = time.time()
	for batch_idx, _data in enumerate(train_loader):
		spectrograms, labels, input_lengths, label_lengths = _data 
		spectrograms, labels = spectrograms.to(device), labels.to(device)

		optimizer.zero_grad()

		output = model(spectrograms)  # (batch, time, n_class)
		output = F.log_softmax(output, dim=2)
		output = output.transpose(0, 1) # (time, batch, n_class)

		loss = criterion(output, labels, input_lengths, label_lengths)
		loss.backward()

		optimizer.step()
		scheduler.step()
		iter_meter.step()
		if batch_idx % 100 == 0 or batch_idx == data_len:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(spectrograms), data_len,
				100. * batch_idx / len(train_loader), loss.item()))
	print('Epoch train time: ', time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))
	start_time = time.time()


def test(model, device, test_loader, criterion, epoch, iter_meter):
	print('\nEvaluating ...')
	start_time = time.time()
	model.eval()
	test_loss = 0
	test_cer, test_wer = [], []
	with torch.no_grad():
		for I, _data in enumerate(test_loader):
			spectrograms, labels, input_lengths, label_lengths = _data 
			spectrograms, labels = spectrograms.to(device), labels.to(device)

			output = model(spectrograms)  # (batch, time, n_class)
			output = F.log_softmax(output, dim=2)
			output = output.transpose(0, 1) # (time, batch, n_class)

			loss = criterion(output, labels, input_lengths, label_lengths)
			test_loss += loss.item() / len(test_loader)

			decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
			for j in range(len(decoded_preds)):
				test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
				test_wer.append(wer(decoded_targets[j], decoded_preds[j]))


	avg_cer = sum(test_cer)/len(test_cer)
	avg_wer = sum(test_wer)/len(test_wer)

	print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))
	print('Epoch test time: ', time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))


def main(learning_rate=5e-4, batch_size=20, epochs=10,
		train_url='train-clean-100', test_url='test-clean'):
	start_time = time.time()
	hparams = {
		'n_cnn_layers': 3,
		'n_rnn_layers': 5,
		'rnn_dim': 512,
		'n_class': 29,
		'n_feats': 128,
		'stride': 2,
		'dropout': 0.1,
		'learning_rate': learning_rate,
		'batch_size': batch_size,
		'epochs': epochs
	}

	use_cuda = torch.cuda.is_available()
	torch.manual_seed(7)
	device = torch.device('cuda' if use_cuda else 'cpu')

	if not os.path.isdir('./../data'):
		os.makedirs('./../data')
		print('\nDownloading Dataset')
	else:
		print('\nDataset Found')

	train_dataset = torchaudio.datasets.LIBRISPEECH('./../data', url=train_url, download=True)
	test_dataset = torchaudio.datasets.LIBRISPEECH('./../data', url=test_url, download=True)

	print('Loading data ...')
	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
	train_loader = data.DataLoader(dataset=train_dataset,
								batch_size=hparams['batch_size'],
								shuffle=True,
								collate_fn=lambda x: data_processing(x, 'train'),
								**kwargs)
	test_loader = data.DataLoader(dataset=test_dataset,
								batch_size=hparams['batch_size'],
								shuffle=False,
								collate_fn=lambda x: data_processing(x, 'valid'),
								**kwargs)

	print('\nCreating model ...')
	model = SpeechRecognitionModel(
		hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
		hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
		).to(device)

	print(model)
	print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

	optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
	criterion = nn.CTCLoss(blank=28).to(device)
	scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'], 
											steps_per_epoch=int(len(train_loader)),
											epochs=hparams['epochs'],
											anneal_strategy='linear')

	print('Setup time: ', time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))

	iter_meter = IterMeter()
	for epoch in range(1, epochs + 1):
		train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter)
		test(model, device, test_loader, criterion, epoch, iter_meter)


"""
Train and evaluate the model
"""
if __name__ == '__main__':
	learning_rate = 5e-4
	batch_size = 10
	epochs = 1
	libri_train_set = 'train-clean-100'
	libri_test_set = 'test-clean'
	print('Training model with the following parameters:\nLearning Rate: {}\nBatch Size: {}\nEpochs: {}'.format(learning_rate, batch_size, epochs))

	main(learning_rate, batch_size, epochs, libri_train_set, libri_test_set)