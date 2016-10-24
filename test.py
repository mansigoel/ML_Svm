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
import pickle
"""
Source to read dataset: https://gist.github.com/akesling/5358964

"""

def read(dataset = "training", path = "."):
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows*cols)

    img,lbl = shuffle(img,lbl)
    img = (img/(255.0))
    return img,lbl


xtrain, ytrain = read('training')
xtest, ytest = read('testing')

size = len(ytrain)
sizet = len(ytest)

Xtrain, Ytrain = [],[]
Xtest, Ytest = [],[]

labels = list(set(ytrain))

for lb in labels:
    count = 0
    for i in range(0,size):
        if ytrain[i]==lb:
            Xtrain.append(xtrain[i])
            Ytrain.append(ytrain[i])
            count+=1
        if(count==2000):
            break

for lb in labels:
    count = 0
    for i in range(0,sizet):
        if ytest[i]==lb:
            Xtest.append(xtest[i])
            Ytest.append(ytest[i])
            count+=1
        if(count==500):
            break

Xtrain = np.array(Xtrain)
Ytrain = np.array(Ytrain)
Xtest = np.array(Xtest)
Ytest = np.array(Ytest)

Xtrain,Ytrain = shuffle(Xtrain,Ytrain)
Xtest, Ytest = shuffle(Xtest,Ytest)

Ytbrain = label_binarize(Ytrain, classes = np.unique(Ytrain))
Ytbest = label_binarize(Ytest, classes = np.unique(Ytest))
all1 = []
all1.append(['Analysis For Multi class'])
print "multi-class"

# grid_params = {'kernel': ['rbf'], 'C': [1e-5, 1e-2, 1, 10, 100, 1000],'gamma': [1e-5, 1e-2,1e-1,1, 10, 100, 1000]}
# score = 'accuracy'
# print("# Scoring parameter is %s" % score)
# all1.append([score])

# clf = GridSearchCV(SVC(max_iter = 10000), grid_params, cv=5, scoring='%s' % score)
# clf.fit(Xtrain, Ytrain)

# print("Best parameters set: ")
# print(clf.best_params_)
# best = clf.best_params_
# all1.append([str(clf.best_params_)])

data1 = {'train': {'X': Xtrain,'y': Ytrain},'test': {'X': Xtest,'y': Ytest}}
all1.append([''])
all1.append(['Accuracy Values'])
try:
    clf = OneVsRestClassifier(SVC(probability=False,cache_size=200,kernel="linear", max_iter = 10000, C=1),-1)
    req = len(data1['train']['X'])
    clf.fit(data1['train']['X'][:req], data1['train']['y'][:req])
    yt =  clf.predict(data1['test']['X'])
    acc = accuracy_score(data1['test']['y'], yt)
    print acc
    all1.append([acc])
    all1.append(['recommended C value'])
    print "recommended C value"
    joblib.dump(clf, 'part(b).model',compress=1)
    i=0
    for model in clf.estimators_:
        print i
        joblib.dump(model, 'part(b)-'+str(i)+'.model', compress =1)
        i+=1

    smatrix = clf.decision_function(data1['test']['X'])
    smatrix = np.reshape(smatrix, (smatrix.shape[0],10))
    print "shape is ", smatrix.shape
    val1,val2,val3 = generate_roc(smatrix,Ytbest,nROCpts =100 ,plotROC = 'true', name= 'Roc_1(b).png')
    all1.append([str(val1),str(val2),str(val3)])
    all1.append([''])

    print "--------------------------------------------------------------------------------------------------"
except Exception,e:
    print e

with open('analysis(1b).csv', 'w') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(all1)

print "end@@@@@@@@@@@"
