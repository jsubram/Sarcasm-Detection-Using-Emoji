import Word2Vec as w
import Emoji2Vec as ex
import numpy as np
import pandas as pd

def JoinVectors(wv,ev,n):

	Concatenated_Vector = []
	if len(wv) == len(ev):
		for i in range(n):
			Concatenated_Vector.append(np.hstack((wv[i],ev[i])))
	else:
		return False

	return Concatenated_Vector

def ReadLabels(filename):
	ClassLabels = pd.read_csv(filename,index_col=False)

	return list(ClassLabels['Comments'])
	

def main():
	word_vec = w.main()
	Emoji_vec = ex.main()
	print("Concatenating...")
	Concatenated_Vector = JoinVectors(word_vec,Emoji_vec,len(word_vec))
	return Concatenated_Vector

#main()
