import numpy as np
import enchant
import gensim 
from gensim.models import Word2Vec 
import pandas as pd
from nltk import word_tokenize
import codecs

d = enchant.Dict("en_US")


def ReadOpen(filename):

	# sample = open("Final_Dataset_Word2Vec_Emoji2Vec.csv", "r",encoding = "ISO-8859-1")
	# sample = codecs.open(filename, "r", encoding="utf-8", errors="replace")
	sample = codecs.open(filename, "r", errors="replace") 
	sample = open(filename, "r") 
	s = sample.read() 

	f = s.replace("\n", " ") 
	data = []

	l = s.split('\n')
	for i in l:
	    temp = [] 
	      
	    for j in word_tokenize(i):
	    	if d.check(j):
	    		temp.append(j.lower()) 
	  
	    data.append(temp)

	return data


def FilterNonEnglish(sentence):
	new_sent = []
	for i in sentence.split():
		if d.check(i):
			new_sent.append(i)
	return new_sent



def PandasReadData(filename):
	emoji_dataset=pd.read_csv(filename,index_col=False,encoding="ISO-8859-1")
	data = []

	for i in range(len(emoji_dataset['Comments'])):
		data.append(FilterNonEnglish(emoji_dataset['Comments'][i]))

	return data

def CreateAndTrainModel(data,size=300):
	model = gensim.models.Word2Vec(data,size=size,window=10,min_count=2,workers=10)
	model.train(data, total_examples=len(data), epochs=10)
	return model
	# print(model.wv.most_similar(positive='sarcasm'))


def AverageVectorPerTweet(data,model):

	avg = []
	unused = []
	for i in range(len(data)):
		try:	
			row = []
			for j in data[i]:
				row.append(model[j])
			row = np.asarray(row)
			if len(row) < 1:
				avg.append(np.zeros((300,)).tolist())
				continue
			avg.append((np.average(row,axis=0)).tolist())
		except:
			avg.append(np.zeros((300,)).tolist())

	return avg


def main(filename):
	# filename = "../Data/Final_Dataset_Word2Vec_Emoji2Vec.csv"
	print("Running Word2Vec...")
	# data = ReadOpen(filename)
	data = PandasReadData(filename)
	model = CreateAndTrainModel(data)
	avg = AverageVectorPerTweet(data,model)
	# print(avg)
	# print(len(data),len(avg))
	return avg

# main()	
