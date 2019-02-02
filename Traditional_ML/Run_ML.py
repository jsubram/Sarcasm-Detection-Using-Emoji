import numpy as np
from sklearn import svm

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score 
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier

import random
import matplotlib.pyplot as plt
import sys
sys.path.append('../Process_Embeddings')
import RetrieveEmbeddings as r

import pandas as pd

def ReadLabels(filename):
	ClassLabels = pd.read_csv(filename,index_col=False)

	return list(ClassLabels['Comments'])

def ML(X,Y):

	print("Training...")
	# print(len(X),len(Y))
	X = np.array(X)
	Y = np.array(Y)
	# print(X)
	# for i in range(len(X)):
	# 	# print(i)
	# 	# print(X[i])
	# 	for j in X[i]:
	# 		j = float(j)

	# for i in Y:
	# 	for j in i:
	# 		j = float(j)


	kfold = KFold(10, True, 1)
	for train_index, test_index in kfold.split(X):
	    x_train, x_test = X[train_index], X[test_index]
	    y_train, y_test = Y[train_index], Y[test_index]

	# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

	clf1 = svm.SVC(kernel = 'rbf')
	clf1.fit(x_train, y_train)
	predictions = clf1.predict(x_test)
	predictions[0:30]
	print("SVM RBF Kernel")
	rbf_score=f1_score(y_test, predictions, average = 'macro')
	print(rbf_score)

	clftree = tree.DecisionTreeClassifier()
	model2 = clftree.fit(x_train, y_train)
	predictions = clftree.predict(x_test)
	predictions[0:30]
	print("Decision Tree Classifier")
	dt_score=f1_score(y_test, predictions, average = 'macro')
	print(dt_score)

	clf4 = RandomForestClassifier()
	clf4.fit(x_train, y_train)
	predictions = clf4.predict(x_test)
	predictions[0:30]
	print("Random Forest classifier")
	rf_score=f1_score(y_test,predictions, average = 'macro')
	print(rf_score)

	clf5 = AdaBoostClassifier()
	clf5.fit(x_train, y_train)
	predictions = clf5.predict(x_test)
	predictions[0:30]
	print("AdaBoost classifier")
	ad_score = f1_score(y_test,predictions, average = 'macro')
	print(ad_score)

	clf6 = GradientBoostingClassifier()
	clf6.fit(x_train, y_train)
	predictions = clf6.predict(x_test)
	predictions[0:30]
	print("GradientBoosting Classifier")
	gd_score = f1_score(y_test,predictions, average = 'macro')
	print(gd_score)

	clf7 = KNeighborsClassifier()
	clf7.fit(x_train, y_train)
	predictions = clf7.predict(x_test)
	predictions[0:30]
	print("KNN Classifier")
	knn_score= f1_score(y_test,predictions, average = 'macro')
	print(knn_score)

	print("SGDClassifier")
	clf = SGDClassifier()
	clf.fit(x_train, y_train)
	predictions = clf.predict(x_test)
	print(clf.score(x_test, y_test))

	print("Bayesian Classifier")
	clf = GaussianNB()
	clf.fit(x_train, y_train)
	predictions = clf.predict(x_test)
	print(f1_score(y_test, predictions, average = 'macro'))

	print("Extra Tree Classifier")
	clf = ExtraTreesClassifier()
	clf.fit(x_train, y_train)
	predictions = clf.predict(x_test)
	print(f1_score(y_test, predictions, average = 'macro'))


if __name__ == "__main__":	
	data = r.main()
	labelfile = "../Data/Final_Dataset_Word2Vec_Emoji2Vec_Labels.csv"
	labels = ReadLabels(labelfile)
	ML(data,labels)

# main()
