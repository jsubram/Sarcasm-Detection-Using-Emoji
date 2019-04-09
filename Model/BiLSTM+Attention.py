import numpy
import numpy as np
from numpy import asarray
from numpy import zeros
import pandas as pd

numpy.random.seed(1337)

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding
from keras.layers import LSTM, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.core import*
from keras import initializers, regularizers, constraints, Input
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras import backend as K


import codecs
import csv
from nltk import word_tokenize


from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

import emoji
import gensim
import time
import os
import sys
import json
sys.path.append('../')

seed = 7
numpy.random.seed(seed)


class Attention(Layer):
	def __init__(self,
				 W_regularizer=None, b_regularizer=None,
				 W_constraint=None, b_constraint=None,
				 bias=True, **kwargs):

		self.supports_masking = True
		self.init = initializers.get('glorot_uniform')

		self.W_regularizer = regularizers.get(W_regularizer)
		self.b_regularizer = regularizers.get(b_regularizer)

		self.W_constraint = constraints.get(W_constraint)
		self.b_constraint = constraints.get(b_constraint)

		self.bias = bias
		super(Attention, self).__init__(**kwargs)

	def build(self, input_shape):
		assert len(input_shape) == 3

		self.W = self.add_weight((input_shape[-1],),
								 initializer=self.init,
								 name='{}_W'.format(self.name),
								 regularizer=self.W_regularizer,
								 constraint=self.W_constraint)
		if self.bias:
			self.b = self.add_weight((input_shape[1],),
									 initializer='zero',
									 name='{}_b'.format(self.name),
									 regularizer=self.b_regularizer,
									 constraint=self.b_constraint)
		else:
			self.b = None

		self.built = True

	def compute_mask(self, input, input_mask=None):
		return None

	def call(self, x, mask=None):
		eij = K.squeeze(K.dot(x, K.expand_dims(self.W)), axis=-1)

		if self.bias:
			eij += self.b

		eij = K.tanh(eij)

		a = K.exp(eij)

		if mask is not None:
			a *= K.cast(mask, K.floatx())

		a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

		a = K.expand_dims(a)

		weighted_input = x * a
		return K.sum(weighted_input, axis=1)

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[-1])




def ReadOpen(filename,Labelfile):
	# sample = codecs.open(filename, "r", encoding="utf-8", errors="replace")
	# s = sample.read()
	data = []

	with codecs.open(filename, 'r',encoding="utf-8", errors="replace") as readFile:
		reader = csv.reader(readFile)
		lines = list(reader)
	count = 0
	for i in lines:
		temp = []
		sentence = ' '.join(i)
		for j in word_tokenize(sentence):
			temp.append(j.lower()) 
			count += 1
	  
		data.append(temp)

	labels_pd = pd.read_csv(Labelfile,index_col=False)
	# labels = numpy.array(labels_pd['Comments'])
	labels = numpy.asarray(labels_pd)

	return data[1:],labels, count-1


def extract_emojis(sentence):
	return [word for word in sentence.split() if str(word.encode('unicode-escape'))[2] == '\\' ]


def char_is_emoji(character):
	if character in emoji.UNICODE_EMOJI:
		return True
	else:
		return False

def Preprocess(docs,count):

	# prepare tokenizer
	t = Tokenizer()
	t.fit_on_texts(docs)
	# vocab_size = len(t.word_index) + 1
	# print(vocab_size)
	# integer encode the documents
	encoded_docs = t.texts_to_sequences(docs)
	# pad documents to a max length of 4 words
	# max_length = 4
	padded_docs = pad_sequences(encoded_docs, padding='post')
	l = len(padded_docs[0])

	# load the whole embedding into memory
	embeddings_index = dict()
	f = open('glove.6B.300d.txt')
	for line in f:
		values = line.split()
		word = values[0]
		coefs = asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	print('Loaded %s word vectors.' % len(embeddings_index))

	e2v = gensim.models.KeyedVectors.load_word2vec_format("emoji2vec.bin", binary=True)
	nf = 0
	# create a weight matrix for words in training docs
	embedding_matrix = zeros((count, 300))
	for word, i in t.word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
		else:
			# print(word)
			new_em = []
			em = extract_emojis(word)
			for ej in em:
				for c in ej:
					if char_is_emoji(c):
						new_em.append(c)
			# print(new_em)
			try:
				if new_em:
						row = []
						for e in new_em:

							row.append(e2v[e])
						embedding_matrix[i] = np.average(np.asarray(row),axis=0).tolist()
				else:
					embedding_matrix[i] = [0] * 300
			except:
				embedding_matrix[i] = [0] * 300
				nf += 1

	print(str(nf)+" words not found in vocabulary")

	return padded_docs, embedding_matrix,l

def f1(y_true, y_pred):
	def recall(y_true, y_pred):

		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
		recall = true_positives / (possible_positives + K.epsilon())
		return recall

	def precision(y_true, y_pred):

		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
		precision = true_positives / (predicted_positives + K.epsilon())
		return precision
	precision = precision(y_true, y_pred)
	recall = recall(y_true, y_pred)
	return 2*((precision*recall)/(precision+recall+K.epsilon()))

def PrepModel(count,embedding_matrix,l,lrate=0.01):
	model = Sequential()
	e = Embedding(count, 300, weights=[embedding_matrix], input_length=l, trainable=False)
	model.add(e)

	model.add(LSTM(100,kernel_initializer='he_normal', activation='sigmoid', dropout=0.5,recurrent_dropout=0.5, unroll=False, return_sequences=True))

	model.add(Attention())
	model.add(Dense(1, activation='sigmoid'))
	model.compile(optimizer=Adam(lr=lrate), loss='binary_crossentropy', metrics=[f1])

	# model.add(Dense(2))
	# model.add(Activation('softmax'))
	# adam = Adam(lr=0.001)
	# model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
	print('No of parameter:', model.count_params())

	print(model.summary())
	print(K.eval(model.optimizer.lr))
	return model

def ReadTest(filename,Labelfile):

	data = []

	with codecs.open(filename, 'r',encoding="utf-8", errors="replace") as readFile:
		reader = csv.reader(readFile)
		lines = list(reader)
	count = 0
	for i in lines:
		temp = []
		sentence = ' '.join(i)
		for j in word_tokenize(sentence):
			temp.append(j.lower()) 
			count += 1
	  
		data.append(temp)

	labels_pd = pd.read_csv(Labelfile,index_col=False)
	labels = numpy.array(labels_pd['Labels'])
	# labels = numpy.asarray(labels_pd)

	t = Tokenizer()
	t.fit_on_texts(data)
	encoded_docs = t.texts_to_sequences(data)
	padded_docs = pad_sequences(encoded_docs, padding='post',maxlen=926)

	return padded_docs,labels



if __name__ == "__main__":

	with open('settings.json') as data_file:
		data = json.load(data_file)

	lrate = data["Model_settings"]["Learning_rate"]
	num_epochs = data["Model_settings"]["Epochs"]

	filename = data["FileNames"]["Training_file"]
	Labelfile = data["FileNames"]["Label_file"]
	# data,labels = ReadFile(filename,Labelfile)


	print('Reading data...')
	data,labels,count = ReadOpen(filename,Labelfile)
	print('Getting Embeddings...')
	padded_docs, embedding_matrix,l = Preprocess(data,count)
	print('Preparing model...')
	model = PrepModel(count,embedding_matrix,l,lrate)
	print('Training...')

	X_train, X_test, y_train, y_test = train_test_split(padded_docs, labels, test_size=0.2, random_state=seed)
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
	# X_test1, y_test1 = ReadTest("Testing_data.csv","Testing_labels.csv")
	# print("length of X_test,y_test")
	# print(len(X_test),len(y_test))


	earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=1, mode='auto')
	model.fit(X_train, y_train, validation_data=(X_val,y_val), nb_epoch=num_epochs, verbose=1, callbacks=[earlyStopping])
	loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
	# loss1, accuracy1 = model.evaluate(X_test1, y_test1, verbose=1)
	print('Accuracy: %f' % (accuracy*100))
	# print('Accuracy of 10: %f' % (accuracy1*100))
