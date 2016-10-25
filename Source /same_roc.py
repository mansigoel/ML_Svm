import os
import csv
import struct
import numpy as np
from sklearn import datasets, svm, metrics
import scipy
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand
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
from pprint import pformat

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

multi = joblib.load('2014062_Models/part(b).model')
rbf = joblib.load('2014062_Models/part(c).model')
data1 = {'train': {'X': Xtrain,'y': Ytrain},'test': {'X': Xtest,'y': Ytest}}

smatrix = multi.decision_function(data1['test']['X'])
smatrix = np.reshape(smatrix, (smatrix.shape[0],10))
print "shape is ", smatrix.shape
# val1,val2,val3 = generate_roc(smatrix,Ytbest,nROCpts =100 ,plotROC = 'true', name= 'Roc_1(b).png')
trueLabels = Ytbest
scoreMatrix = smatrix
nROCpts = 100
tpr = np.zeros([1,nROCpts]) 
fpr = np.zeros([1,nROCpts]) 
nTrueLabels = np.count_nonzero(trueLabels) 
nFalseLabels = np.size(trueLabels) - nTrueLabels 

minScore = np.min(scoreMatrix)
maxScore = np.max(scoreMatrix);
rangeScore = maxScore - minScore;


thdArr = minScore + rangeScore*np.arange(0,1,float(1)/(nROCpts))
#print thdArr
for thd_i in range(0,nROCpts):
    thd = thdArr[thd_i]
    ind = np.where(scoreMatrix>=thd) 
    thisLabel = np.zeros([np.size(scoreMatrix,0),np.size(scoreMatrix,1)])
    thisLabel[ind] = 1
    tpr_mat = np.multiply(thisLabel,trueLabels)
    tpr[0,thd_i] = np.sum(tpr_mat)/nTrueLabels
    fpr_mat = np.multiply(thisLabel, 1-trueLabels)
    fpr[0,thd_i] = np.sum(fpr_mat)/nFalseLabels
    
print "okay"
plt.xlim(min(fpr[0,:]), max(fpr[0,:]))
plt.plot(fpr[0,:],tpr[0,:], 'b.-', label="Multilinear")


print "multi done"
smatrix1 = rbf.decision_function(data1['test']['X'])
smatrix1 = np.reshape(smatrix1, (smatrix1.shape[0],10))


scoreMatrix = smatrix1
nROCpts = 100
tpr = np.zeros([1,nROCpts]) 
fpr = np.zeros([1,nROCpts]) 
nTrueLabels = np.count_nonzero(trueLabels) 
nFalseLabels = np.size(trueLabels) - nTrueLabels 

minScore = np.min(scoreMatrix)
maxScore = np.max(scoreMatrix);
rangeScore = maxScore - minScore;


thdArr = minScore + rangeScore*np.arange(0,1,float(1)/(nROCpts))
#print thdArr
for thd_i in range(0,nROCpts):
    thd = thdArr[thd_i]
    ind = np.where(scoreMatrix>=thd) 
    thisLabel = np.zeros([np.size(scoreMatrix,0),np.size(scoreMatrix,1)])
    thisLabel[ind] = 1
    tpr_mat = np.multiply(thisLabel,trueLabels)
    tpr[0,thd_i] = np.sum(tpr_mat)/nTrueLabels
    fpr_mat = np.multiply(thisLabel, 1-trueLabels)
    fpr[0,thd_i] = np.sum(fpr_mat)/nFalseLabels
    
print "rbf done"
plt.xlim(min(fpr[0,:]), max(fpr[0,:]))
plt.plot(fpr[0,:],tpr[0,:], 'k.-', label="Multi RBF")

plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.title("Cross Multi and RBF ROC Curve")
plt.legend(loc=4)
plt.savefig("Combined_ROC.png")