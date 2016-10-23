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

X1train, Y1train = [],[]
X1test, Y1test = [],[]
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

for lb in [3,8]:
    count = 0
    for i in range(0,size):
        if ytrain[i]==lb:
            X1train.append(xtrain[i])
            Y1train.append(ytrain[i])
            count+=1
        if(count==2000):
            break

for lb in [3,8]:
    count = 0
    for i in range(0,sizet):
        if ytest[i]==lb:
            X1test.append(xtest[i])
            Y1test.append(ytest[i])
            count+=1
        if(count==500):
            break

Xtrain = np.array(Xtrain)
Ytrain = np.array(Ytrain)
X1train = np.array(X1train)
Y1train = np.array(Y1train)

Xtest = np.array(Xtest)
Ytest = np.array(Ytest)
X1test = np.array(X1test)
Y1test = np.array(Y1test)

Y1btrain = label_binarize(Y1train, classes = np.unique(Y1train))
Ytbrain = label_binarize(Ytrain, classes = np.unique(Ytrain))
Ytbest = label_binarize(Ytest, classes = np.unique(Ytest))
Y1btest = label_binarize(Y1test, classes = np.unique(Y1test))

Xtrain,Ytrain = shuffle(Xtrain,Ytrain)
X1train, Y1train = shuffle(X1train,Y1train)
Xtest, Ytest = shuffle(Xtest,Ytest)
X1test, Y1test = shuffle(X1test,Y1test)


all1 = []
all1.append(['Analysis For 2 class'])
print Y1train
print "binary-CLASS"
grid_params = {'kernel': ['linear'], 'C': [1e-6, 1e-3, 1, 10, 100,1000]}

scores = ['accuracy', 'recall_macro']
best = 0
for score in scores:
    print("# Scoring parameter is %s" % score)
    all1.append([score])
    clf = GridSearchCV(SVC(C=1), grid_params, cv=5, scoring='%s' % score)
    clf.fit(X1train, Y1train)

    print("Best parameters set: ")
    print(clf.best_params_)
    all1.append([str(clf.best_params_['C'])])
    if score=='accuracy':
        best = clf.best_params_

all1.append([''])
all1.append(['Accuracy Values'])
data1 = {'train': {'X': X1train,'y': Y1train},'test': {'X': X1test,'y': Y1test}}

for item in grid_params['C']:
    print "for C = " + str(item) + "accuracy ="
    all1.append([str(item)])
    
    clf = SVC(probability=False,kernel="linear", C=item)
    req = len(data1['train']['X'])
    clf.fit(data1['train']['X'][:req], data1['train']['y'][:req])

    yt =  clf.predict(data1['test']['X'])
    acc = accuracy_score(data1['test']['y'], yt)
    print acc
    all1.append([acc])
    if item==best['C']:
        all1.append(['recommended C value'])
        print "recommended C value"
        joblib.dump(clf, 'part(a)_model.pkl')
        smatrix = clf.decision_function(data1['test']['X'])
        print "shape is ", smatrix.shape
        val1,val2,val3 = generate_roc(smatrix,Y1btest,nROCpts =100 ,plotROC = 'true', name= 'Roc_1(a).png')
        all1.append([str(val1),str(val2),str(val3)])
    all1.append([''])

    print "--------------------------------------------------------------------------------------------------"

with open('analysis(1a).csv', 'w') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(all1)

print "end@@@@@@@@@@@"