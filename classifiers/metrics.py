from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np

def confusion_matrix(Y, T):
    """
        Y    ndarray
             predicted labels
        T    ndarray
             target labels
             
        @cfm DataFrame
             confusion matrix
    """
    
    if len(Y) != len(T):
        raise ValueError("Wrong prediction and target length!")
    
    classes = np.unique(T)
    n_classes = len(classes)
    
    cfm = pd.DataFrame(np.zeros((n_classes, n_classes)), index=classes, columns=classes, dtype=int)
    
    Tidx = [T == c for c in classes]
    for c in classes:
        pred_idx = Y == c
        cfm.loc[c, :] = [np.sum(np.logical_and(pred_idx, tidx)) for tidx in Tidx]
    
    return cfm

def accuracy(Y,T):
    return np.sum(Y==T)/len(Y)

def precision(cfm):
    cfm = cfm.as_matrix()
    cfm = np.float64(cfm)
    return cfm[1,1] / (cfm[1,1]+cfm[1,0])

def recall(cfm):
    cfm = cfm.as_matrix()
    cfm = np.float64(cfm)
    return cfm[1, 1] / (cfm[1, 1] + cfm[0,1])
def specificity(cfm):
    cfm = cfm.as_matrix()
    cfm = np.float64(cfm)
    return cfm[0,0]/(cfm[0,0]+cfm[1,0])

def f1Score(cfm):
    cfm = cfm.as_matrix()
    cfm = np.float64(cfm)
    return cfm[1,1] / cfm[1,1] + ((cfm[0,1]+cfm[1,0])/2)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1.02])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

def roc(prob,Y):
    fpr, tpr, thresholds = roc_curve(Y, prob)
    plt.figure(figsize=(8, 6))
    plot_roc_curve(fpr, tpr)
    plt.show()

    
def roc_auc(prob,Y):
    return roc_auc_score(Y, prob)

def plotClassifedResult(Y,T):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(T, 'o', color='b', label='Real',linewidth=5) 
    ax.plot(Y, '.', color='r', label='Predicted')
    ax.set_xlabel('Index')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.show()
                                                     
def allStats(Y,T):
    cfm = confusion_matrix(Y,T)
    display(cfm)
    display(pd.DataFrame([['TN','FN'],['FP','TP']],index=["-","+"],columns=["-","+"]))
    stats = [accuracy(Y,T),precision(cfm),recall(cfm),specificity(cfm),f1Score(cfm),roc_auc(Y,T)]
    display(pd.DataFrame(stats,index=["Accuracy","Precision","Recall","Specificity","F1 Score","ROC_AUC"],columns=["Stats"]))
    plotClassifedResult(Y,T)
    roc(Y,T)
    return stats 

def compStats(Y,T):
    cfm = confusion_matrix(Y,T)
    stats = [accuracy(Y,T),precision(cfm),recall(cfm),specificity(cfm),f1Score(cfm),MCC(cfm),roc_auc(Y,T)]
    return stats
