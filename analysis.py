from scipy import stats
import numpy as np
from utils import extract_gt_mRS
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

mRS = extract_gt_mRS()
volumetric_pred = np.load('./volumetric_pred_absolute.npy')
spatial_pred = np.load('./spatial_pred_absolute.npy')
morphological_pred = np.load('./morphological_pred_absolute.npy')
HarvardOxfordCort_pred = np.load('./HarvardOxfordCort_pred_absolute.npy')
aal_pred = np.load('./aal_pred_absolute.npy')
Wdsi_end_roi_pred = np.load('./Wdsi_end_roi_absolute.npy')


y = mRS.reshape((37,1))
all_decision = np.concatenate((volumetric_pred, spatial_pred, morphological_pred, HarvardOxfordCort_pred, Wdsi_end_roi_pred), axis=1)

majority_vote = Wdsi_end_roi_pred
majority_vote[12] = 2
majority_vote[19] = 2
majority_vote[20] = 2
majority_vote[26] = 2
majority_vote[28] = 1
majority_ae = np.absolute(y-majority_vote)
print(y)
#volumetric_ae = np.absolute(y-volumetric_pred)
#tractographic_ae = np.absolute(y-Wdsi_end_roi_pred)
#spatial_ae = np.absolute(y-spatial_pred)
#morphological_ae = np.absolute(y-morphological_pred)
#HarvardOxfordCort_ae = np.absolute(y-HarvardOxfordCort_pred)
#aal_ae = np.absolute(y-aal_pred)
#print(stats.ttest_rel(aal_ae, spatial_ae))

#print(np.median(HarvardOxfordCort_ae))
cnf_matrix_volume = confusion_matrix(mRS, majority_vote)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix_volume, classes=[0, 1, 2, 3, 4],
                      title='Confusion matrix(Tractographic Features)')
