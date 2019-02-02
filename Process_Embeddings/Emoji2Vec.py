import numpy as np
import emoji
import gensim
import pandas as pd
import nltk
import csv
from nltk import word_tokenize
import codecs

def extract_emojis(sentence):
	return [word for word in sentence.split() if str(word.encode('unicode-escape'))[2] == '\\' ]

def char_is_emoji(character):
	if character in emoji.UNICODE_EMOJI:
		return True
	else:
		return False

def text_has_emoji(text):
    for character in text:
        if character in emoji.UNICODE_EMOJI:
            return True
    return False

def ReadOpen(filename):
	# sample = codecs.open(filename, "r", encoding="utf-8", errors="replace")
	# s = sample.read()

	data = []

	with codecs.open(filename, 'r',encoding="utf-8", errors="replace") as readFile:
		reader = csv.reader(readFile)
		lines = list(reader)

	for i in lines:
	    temp = []
	    sentence = ' '.join(i)
	    for j in word_tokenize(sentence):
	        temp.append(j.lower()) 
	  
	    data.append(temp)

	return data[1:]

def PandasReadData(filename):

	emoji_dataset=pd.read_csv(filename,index_col=False,encoding="ISO-8859-1")
	data = []

	for i in range(len(emoji_dataset)):
		data.append(emoji_dataset['Comments'][i].split())

	return data

def CreateEmojiList(data):

	em = []
	for row in data:
		sentence = " ".join(row)
		em.append(extract_emojis(sentence))

	return em

def FilterNonEnglish(em):

	new_em = []
	for row in em:
		ej = "".join(row)
		ef = []
		for c in ej:
			if char_is_emoji(c):
				ef.append(c)
		new_em.append(ef)

	return new_em

def GenerateEmojiVectors(emoji_list, pretrained_model):
	e2v = gensim.models.KeyedVectors.load_word2vec_format(pretrained_model, binary=True)

	Emoji_vec = []
	for i in range(len(emoji_list)):
		try:	
			row = []
			for j in emoji_list[i]:
				row.append(e2v[j])
			row = np.asarray(row)
			if len(row) < 1:
				Emoji_vec.append(np.zeros((300,)).tolist())
				continue
			Emoji_vec.append(np.average(row,axis=0).tolist())
		except:
			Emoji_vec.append(np.zeros((300,)).tolist()) 

	return Emoji_vec


def main(filename):
	# filename = "../Data/Final_Dataset_Word2Vec_Emoji2Vec.csv"
	print("Running Emoji2Vec...")
	data = ReadOpen(filename)
	# data = PandasReadData(filename)
	# print(data[:10])
	temp_emojis = CreateEmojiList(data)
	filterd_emojis = FilterNonEnglish(temp_emojis)
	# print(filterd_emojis[:10])
	Emoji_vec = GenerateEmojiVectors(filterd_emojis,"emoji2vec.bin")
	# print(Emoji_vec[:10])
	# print(len(Emoji_vec),len(data))
	return Emoji_vec
# main()
