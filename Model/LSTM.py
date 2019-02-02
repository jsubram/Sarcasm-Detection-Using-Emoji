import numpy
import numpy as np
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
import pandas as pd
import RetrieveEmbeddings as c



from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Bidirectional
from keras.layers.embeddings import Embedding
import codecs
import csv

# define documents
# encoding="ISO-8859-1"

def ReadFile(filename, Labelfile):

	# emoji_dataset=pd.read_csv(filename,index_col=False,encoding="ISO-8859-1")
	# docs = emoji_dataset['Comments'].tolist()
	with codecs.open(filename, 'r',encoding="utf-8", errors="replace") as readFile:
		reader = csv.reader(readFile)
		docs = list(reader)
	# define class labels
	# labels = numpy.array([1,1,1,1,1,0,0,0,0,0])
	labels_pd = pd.read_csv(Labelfile,index_col=False)
	labels = numpy.array(labels_pd['Comments'])

	return docs, labels

def Preprocess(docs):

	# prepare tokenizer
	t = Tokenizer()
	t.fit_on_texts(docs)
	vocab_size = len(t.word_index) + 1
	# integer encode the documents
	encoded_docs = t.texts_to_sequences(docs)
	print(docs)
	# print(encoded_docs)
	# print(len(encoded_docs))
	# pad documents to a max length of 4 words
	max_length = 300
	padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
	print(padded_docs)

	# load the whole embedding into memory
	embeddings_index = dict()
	f = open('glove.6B.100d.txt')
	for line in f:
		values = line.split()
		word = values[0]
		coefs = asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	print('Loaded %s word vectors.' % len(embeddings_index))

	# create a weight matrix for words in training docs
	embedding_matrix = zeros((vocab_size, 100))
	for word, i in t.word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector

	# embedding_matrix = c.main()
	# print(type(embedding_matrix))
	# embedding_matrix = np.array(embedding_matrix)
	# vocab_size = len(embedding_matrix)

	return padded_docs, vocab_size, embedding_matrix

def PrepModel(vocab_size, embedding_matrix):
	# define model
	model = Sequential()
	e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=300, trainable=False)
	model.add(e)
	model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
	# model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	# compile the model
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
	# summarize the model
	print(model.summary())
	return model



def main():
	filename = "Data/Final_Dataset_Word2Vec_Emoji2Vec.csv"
	Labelfile = "Data/Final_Dataset_Word2Vec_Emoji2Vec_Labels.csv"
	data,labels = ReadFile(filename,Labelfile)
	padded_docs, vocab_size, embedding_matrix = Preprocess(data)
	# print(embedding_matrix.shape,vocab_size)
	# print(embedding_matrix)
	# print(vocab_size)
	# # print(len(padded_docs),len(labels))
	model = PrepModel(vocab_size,embedding_matrix)
	# fit the model
	model.fit(padded_docs, labels, epochs=50, verbose=0)
	# evaluate the model
	loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
	print('Accuracy: %f' % (accuracy*100))

#main()
