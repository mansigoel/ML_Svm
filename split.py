import os
import csv
import struct
import numpy as np
from sklearn import datasets, svm, metrics
import scipy
import matplotlib.pyplot
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
from sklearn.multiclass import OneVsRestClassifier
from matplotlib import pyplot
import matplotlib as mpl
from sklearn.utils import shuffle
from sklearn.preprocessing import label_binarize
from roc import generate_roc
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score

clf = joblib.load('part(b)_model.pkl')
i=0
for model in clf.estimators_:
	print i
	joblib.dump(model, 'part(b)-'+str(i)+'.pkl', compress =1)
	i+=1
