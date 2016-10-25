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
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.title(name)
        plt.savefig(name)
        # plt.show()

    return fpr,tpr,thdArr
                             

