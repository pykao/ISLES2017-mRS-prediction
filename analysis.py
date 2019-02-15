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
        cm = np.round(100*cm).astype(int)
        print("Normalized confusion matrix")
        plt.imshow(cm, clim=(0,100), interpolation='nearest', cmap=cmap)
    else:
        print('Confusion matrix, without normalization')
        plt.imshow(cm, interpolation='nearest', cmap=cmap)

    print(cm)
    

    
    plt.title(title, fontsize = 30)
    cb = plt.colorbar()
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=20)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize = 30)
    plt.yticks(tick_marks, classes, rotation=45, fontsize = 30)

    #fmt = '.2f' if normalize else 'd'
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize = 30)

    plt.ylabel('True label', fontsize = 30)
    plt.xlabel('Predicted label', fontsize = 30)
    plt.tight_layout()
    plt.show()

mRS = extract_gt_mRS()

y = mRS.reshape((37,1))

# Original Features
ori_morphological_pred = np.load('./predicted_labels/morphological_pred_loo.npy')
ori_aal_ori_pred = np.load('./predicted_labels/ori_aal_pred_loo.npy')
ori_tract_pred = np.load('./predicted_labels/ori_tract_nrm_end_aal_pred_loo.npy')
ori_oskar_isles2016_pred = np.load('./predicted_labels/oskar_pred_loo.npy')


# absolute error of original features
ori_tractographic_ae = np.absolute(y-ori_tract_pred)
ori_aal_ori_ae = np.absolute(y-ori_aal_ori_pred)
ori_morphological_ae = np.absolute(y-ori_morphological_pred)
ori_oskar_ae = np.absolute(y-ori_oskar_isles2016_pred)



# Features with feature selection

volumetric_pred = np.load('./predicted_labels/rfecv_volumetric_pred_loo.npy')
spatial_pred = np.load('./predicted_labels/rfecv_spatial_pred_loo.npy')
morphological_pred = np.load('./predicted_labels/rfecv_morphological_pred_loo.npy')
aal_ori_pred = np.load('./predicted_labels/rfecv_ori_aal_pred_loo.npy')
aal_tract_pred = np.load('./predicted_labels/rfecv_ori_tract_nrm_end_aal_pred_loo.npy')
oskar_isles2016_pred = np.load('./predicted_labels/rfecv_oskar_pred_loo.npy')

# absolute error of features with feature selection
tractographic_ae = np.absolute(y-aal_tract_pred)
volumetric_ae = np.absolute(y-volumetric_pred)
spatial_ae = np.absolute(y-spatial_pred)
morphological_ae = np.absolute(y-morphological_pred)
aal_ori_ae = np.absolute(y-aal_ori_pred)
oskar_ae = np.absolute(y-oskar_isles2016_pred)


## p value of tractographic feature vs other features 

#print(stats.ttest_rel(aal_tract_pred, oskar_isles2016_pred))


## p value of original feature vs feature with feature selection
#print(stats.ttest_rel(ori_tractographic_ae, tractographic_ae))
#print(stats.ttest_rel(ori_aal_ori_ae, aal_ori_ae))
#print(stats.ttest_rel(ori_morphological_ae, morphological_ae))
#print(stats.ttest_rel(ori_oskar_ae, oskar_ae))



# confusion_matrix(y_true, y_pred)
cnf_matrix_volume = confusion_matrix(y, oskar_isles2016_pred)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix_volume, classes=[0, 1, 2, 3, 4],
                      title="Confusion matrix (Oskar Feature)")
