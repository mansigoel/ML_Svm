import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand


# Generate ROC 
def generate_roc(scoreMatrix,trueLabels,nROCpts =100 ,plotROC = 'false',name='roc.png'):

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
        
        #print fpr
       # print tpr  
    if(plotROC == 'true'):
        plt.xlim(min(fpr[0,:]), max(fpr[0,:]))
        plt.plot(fpr[0,:],tpr[0,:], 'b.-')
        plt.savefig(name)
        # plt.show()

    return fpr,tpr,thdArr
                             




# Driving script Checked with random intialization
# n_classes = 3
# n_pts = 5
# trueLabels  = np.zeros([n_classes,n_pts])
# scoreMatrix = np.zeros([n_classes,n_pts])

# scoreMatrix[0,0]= 1
# scoreMatrix[0,1]= 2
# scoreMatrix[0,2]= 3
# scoreMatrix[0,3]= 4
# scoreMatrix[0,4]= 5


# scoreMatrix[1,0]= 2
# scoreMatrix[1,1]= 4
# scoreMatrix[1,2]= 5
# scoreMatrix[1,3]= 3
# scoreMatrix[1,4]= 1

# scoreMatrix[2,0]= 4
# scoreMatrix[2,1]= 5
# scoreMatrix[2,2]= 1
# scoreMatrix[2,3]= 2
# scoreMatrix[2,4]= 3

# trueLabels[0,0]= 0
# trueLabels[0,1]= 0
# trueLabels[0,2]= 1
# trueLabels[0,3]= 0
# trueLabels[0,4]= 0

# trueLabels[1,0]= 0
# trueLabels[1,1]= 1
# trueLabels[1,2]= 0
# trueLabels[1,3]= 0
# trueLabels[1,4]= 1

# trueLabels[2,0]= 1
# trueLabels[2,1]= 0
# trueLabels[2,2]= 0
# trueLabels[2,3]= 1
# trueLabels[2,4]= 0

# print scoreMatrix
# print trueLabels
# fpr,tpr,thdArr = generate_roc(scoreMatrix,trueLabels,nROCpts =100 ,plotROC = 'true')



