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

#volumetric_pred = np.load('./rfr_loo/rfr_volumetric_pred_loo.npy')
#spatial_pred = np.load('./rfr_loo/rfr_spatial_pred_loo.npy')
#morphological_pred = np.load('./rfr_loo/rfr_morphological_pred_loo.npy')
#HarvardOxfordCort_pred = np.load('./rfr_loo/rfr_HarvardOxfordCort_pred_loo.npy')
#aal_pred = np.load('./rfr_loo/rfr_aal_pred_loo.npy')
#aal_Wdsi_end_roi_pred = np.load('./rfr_loo/rfr_aal_Wdsi_end_roi_pred_loo.npy')


volumetric_pred = np.load('./rfr_loo/rfr_volumetric_pred_loo.npy')
spatial_pred = np.load('./rfr_loo/rfr_spatial_pred_loo.npy')
morphological_pred = np.load('./rfr_loo/rfr_morphological_pred_loo.npy')
HarvardOxfordCort_pred = np.load('./rfr_loo/rfr_HarvardOxfordCort_pred_loo.npy')
aal_pred = np.load('./rfr_loo/rfr_aal_pred_loo.npy')
aal_ori_pred = np.load('./rfr_loo/rfr_aal_ori_pred_loo.npy')
aal_Wdsi_end_roi_pred = np.load('./rfr_loo/rfr_aal_Wdsi_end_roi_pred_loo.npy')
oskar_isles2016_pred = np.load('./rfr_loo/rfr_oskar_pred_loo.npy')
y = mRS.reshape((37,1))

tractographic_ae = np.absolute(y-aal_Wdsi_end_roi_pred)
volumetric_ae = np.absolute(y-volumetric_pred)
spatial_ae = np.absolute(y-spatial_pred)
morphological_ae = np.absolute(y-morphological_pred)
HarvardOxfordCort_ae = np.absolute(y-HarvardOxfordCort_pred)
aal_ae = np.absolute(y-aal_pred)
aal_ori_ae = np.absolute(y-aal_ori_pred)
oskar_ae = np.absolute(y-oskar_isles2016_pred)
print(stats.ttest_rel(tractographic_ae, oskar_ae))
#print(np.median(HarvardOxfordCort_ae))

# confusion_matrix(y_true, y_pred)
#cnf_matrix_volume = confusion_matrix(y, oskar_isles2016_pred)
#np.set_printoptions(precision=2)
#plt.figure()
#plot_confusion_matrix(cnf_matrix_volume, classes=[0, 1, 2, 3, 4],
#                      title='Confusion matrix(Oskar ISLES2016)')