
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

input_dir = 'C:/Users/jocky/OneDrive/Dokumentumok/egyetem/onlab_2/sgcc_data/data/wide_deep/input/'
probs = np.load(input_dir+'probs.npy')
y_test = np.load(input_dir+'y_test_22_us.npy')
thr = 0.44

precision = []
recall = []


for i in range(100):
    threshold = i/100
    predictions = []
    for j in range(len(probs)):
        predictions.append(probs[j]>(threshold))
    precision.append(sklearn.metrics.precision_score(y_test,predictions))
    recall.append(sklearn.metrics.recall_score(y_test,predictions))



auc = roc_auc_score(y_test,probs,average = "macro")
preds = []
for prob in range(len(probs)):
   preds.append(bool(probs[prob]>(thr)))
f1_score = f1_score(y_test,preds)

x_diff = 0.022
y_diff = -0.02

fpr, tpr, threshold = roc_curve(y_test, probs)
roc_auc = sklearn.metrics.auc(fpr, tpr)

## draw ROC Curve
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
#plt.plot([0.13, 0.13], [0, 1], 'y--')
#plt.plot([0.28, 0.28], [0, 1], 'y--')
#plt.plot([0.07, 0.07], [0, 1], 'y--')
plt.plot([0.068],[0.46], 'r+')
plt.text(0.068+x_diff,0.46+y_diff, 'P1 (FPR = 0.068, TPR = 0.46)')
plt.plot([0.125],[0.57], 'r+')
plt.text(0.125+x_diff, 0.57+y_diff, 'P2 (FPR = 0.125, TPR = 0.57)')
plt.plot([0.28],[0.77], 'r+')
plt.text(0.28+x_diff, 0.77+y_diff, 'P3 (FPR = 0.28, TPR = 0.77)')
plt.xlim([0, 1])
plt.ylim([0, 1])


plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()