import paths
import csv
import os
import logging
import argparse

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, LeaveOneOut
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from utils_40 import ReadImage, find_list, threshold_connectivity_matrix, weight_conversion, get_lesion_weights, get_train_dataset
from utils_40 import extract_gt_mRS, extract_volumetric_features, extract_tractographic_features, extract_spatial_features
from utils_40 import extract_volumetric_spatial_features, extract_morphological_features, extract_modified_volumetric_spatial_features
from utils_40 import extract_new_tractographic_features, extract_tract_features

from xgboost import XGBRegressor

# =========================================== setup logs ====================================== #
log = os.path.join(os.getcwd(), 'log.txt')
fmt = '%(asctime)s %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt, filename=log)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter(fmt))
logging.getLogger('').addHandler(console)

# ======================================= Parameters Setting ================================== #
use_original_feature = False
rfecv_feature_selection = True

# ======================================== Select Feature  ==================================== #

#feature_type = 'volumetric'
#feature_type = 'spatial'
#feature_type = 'morphological'

# lesion in brain parcellation regions and the remaining region
#feature_type = 'volumetric_spatial'
# lesion in brain parcellation regions 
feature_type = 'original_volumetric_spatial'
# modified volumetric and spatial feature
#feature_type = 'modified_volumetric_spatial'
#atlas_name = 'HarvardOxfordSub'
#atlas_name = 'HarvardOxfordCort'
atlas_name = 'aal'
#atlas_name = 'JHU-WhiteMatter-labels-1mm'
#atlas_name = 'MNI'
#atlas_name = 'OASIS_TRT_20'

#feature_type = 'oskar'

#feature_type = 'num_fibers'

#feature_type = 'new_fib_pass'
#feature_type = 'new_fib_end'
# original
#weight_type = 'original'
# modified
#weight_type = 'modified'
# one
#weight_type = 'one'

#aal_regions = 116



#feature_type = 'tract_dsi_pass'
#feature_type = 'tract_nrm_pass'
#feature_type = 'tract_bin_pass'
#feature_type = 'tract_dsi_end'
#feature_type = 'tract_nrm_end'
#feature_type = 'tract_bin_end'
# original
#weight_type = 'original'
# modified
#weight_type = 'modified'
# one
#weight_type = 'one'

#aal_regions = 116

# =================================== Groundtruth of mRS scores ================================ #
logging.info('Extracting mRS scores...')
mRS_gt = extract_gt_mRS()


# ======================================== Feature Extraction ================================= #

# ======================================== Voluemtric Feature ================================= #
if 'volume' in feature_type and 'spatial' not in feature_type:
    logging.info('Extracting volumetric features...')
    features, features_list  = extract_volumetric_features()
if 'spatial' in feature_type and 'volume' not in feature_type:
    logging.info('Extracting spatial features...')
    features, features_list = extract_spatial_features()
if 'morpho' in feature_type:
    logging.info('Extracting morphological features...')
    features, features_list = extract_morphological_features()
if  'volume' in feature_type and 'spatial' in feature_type:
    logging.info('Extracting volumetric and spatial features...')
    logging.info(atlas_name)
    if 'ori' not in feature_type and 'mod' not in feature_type:
        logging.info('Included the lesion outside brain parcellation regions...')
        features, features_list = extract_volumetric_spatial_features(atlas_name)
    if 'ori' in feature_type:
        logging.info('Removing the lesion outside brain parcellation regions...')
        features, features_list = extract_volumetric_spatial_features(atlas_name)
        features = features[:,1:]
        del features_list[0]
    if 'mod' in feature_type:
        logging.info('Using modified version of volumetric and spatial features...')
        features, features_list = extract_modified_volumetric_spatial_features(atlas_name)
if 'tract' in feature_type:
    logging.info('Extracting tractographic features...')
    logging.info('Feature type: %s' %feature_type)
    logging.info('Weight type: %s' %weight_type)
    W_dsi_pass, W_nrm_pass, W_bin_pass, W_dsi_end, W_nrm_end, W_bin_end, tract_list = extract_tractographic_features(weight_type, aal_regions)
    if 'dsi' in feature_type and 'pass' in feature_type:
    	features, features_list = W_dsi_pass, tract_list
    elif 'nrm' in feature_type  and 'pass' in feature_type:
    	features, features_list = W_nrm_pass, tract_list
    elif 'bin' in feature_type and 'pass' in feature_type:
    	features, features_list = W_bin_pass, tract_list
    elif 'dsi' in feature_type and 'end' in feature_type:
    	features, features_list = W_dsi_end, tract_list
    elif 'nrm' in feature_type and 'end' in feature_type:
    	features, features_list = W_nrm_end, tract_list
    else:
    	features, features_list = W_bin_end, tract_list

if 'oskar' in feature_type:
    logging.info('Extracting Oskar features...')
    features = np.load('./features/oskar_features_40.npy')
    features_list = list(range(1662))


if 'num_fibers' in feature_type:
    logging.info('Extracting number of tracts...')
    features, features_list = extract_tract_features()

if 'new_fib' in feature_type:
    logging.info('Extracting new tractographic features...')
    logging.info('Feature type: %s' %feature_type)
    logging.info('Weight type: %s' %weight_type)
    W_pass_histogram_features, W_end_histogram_features, tractographic_list = extract_new_tractographic_features(weight_type, aal_regions)
    if 'pass' in feature_type:
        features, features_list = W_pass_histogram_features, tractographic_list
    else:
        features, features_list = W_end_histogram_features, tractographic_list

# =============================== Save Original Features ======================================= #


print(features[0,:])
logging.info('Saving Features...')
if not os.path.exists('./features/'):
	os.mkdir('./features/')
#np.save('./features/volumetric_features.npy', features)
#np.save('./features/spatial_features.npy', features)
#np.save('./features/morphological_features.npy', features)
#np.save('./features/ori_aal_features.npy', features)
#np.save('./features/ori_tract_nrm_end_aal_features.npy', features)
#np.save('./features/mod_tract_bin_end_aal_features.npy', features)

# ==============================  Feature Normalization ========================================= #

logging.info('Normalizing features...')
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# =========================== Remove features with low variance ================================= # 

logging.info('Removing features with low variance...')
sel = VarianceThreshold(0.85*(1-0.85))
selected_normalized_features = sel.fit_transform(normalized_features)
selected_features_list = [name for idx, name in enumerate(features_list) if sel.get_support()[idx]]

# =========================== Select which feature to use =========================================== #


if use_original_feature:
    # ====================================== Original Features ====================================== #
    logging.info('Using original features...')
    X, X_list = normalized_features, features_list
elif rfecv_feature_selection:
    # ======================================= RFECV Features ======================================== #
    logging.info('Using RFECV features...')
    # Cross Validation Model
    loo = LeaveOneOut()
    estimator_rfecv = RandomForestRegressor(n_estimators=300, max_depth=3, random_state=1989, n_jobs=-1)
    rfecv = RFECV(estimator_rfecv, step=1, cv=loo, scoring='neg_mean_absolute_error', n_jobs=-1)
    rfecv_selected_normalized_features = rfecv.fit_transform(selected_normalized_features, mRS_gt)
    rfecv_selected_features_list = [name for idx, name in enumerate(selected_features_list) if rfecv.get_support()[idx]]
    X, X_list = rfecv_selected_normalized_features, rfecv_selected_features_list
else:
    # ======================================= Features with high variance =========================== #
    logging.info('Using features with high variance...')
    X, X_list = selected_normalized_features, selected_features_list

# =============================================== Start Prediction ========================================= #

logging.info('Predicting...')

y = mRS_gt

# Cross Validation Model
loo = LeaveOneOut()

accuracy = np.zeros((40,1), dtype=np.float32)
y_pred_label = np.zeros((40,1), dtype=np.float32)
y_abs_error = np.zeros((40,1), dtype=np.float32)
##y_pred_proba = np.zeros((40,5), dtype=np.float32)

subject_feature_importances = np.zeros((40,len(X_list)), dtype=np.float32)

idx = 0
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # RF Regression
    estimator_pred = RandomForestRegressor(n_estimators=300, max_depth=3, random_state=1989, n_jobs=-1)
    estimator_pred.fit(X_train, y_train)
    subject_importance = estimator_pred.feature_importances_
    subject_feature_importances[idx,:] = subject_importance
    y_pred_label[idx] = np.round(estimator_pred.predict(X_test))
    ##y_pred_proba[idx, :] = estimator_pred.predict_proba(X_test)
    accuracy[idx] = accuracy_score(np.round(estimator_pred.predict(X_test)), y_test)
    y_abs_error[idx] = np.absolute(y_pred_label[idx]-y_test)
    idx += 1

logging.info(feature_type+" features with RF Regressor - Accuracy: %0.4f , MAE: %0.4f (+/- %0.4f)" %(np.mean(accuracy), np.mean(y_abs_error), np.std(y_abs_error)))

# =========================== Save Predicted Label ============================== #

logging.info('Saving Predicted Labels...')

if not os.path.exists('./predicted_labels/'):
	os.mkdir('./predicted_labels/')
#np.save('./predicted_labels/rfecv_volumetric_40_pred_loo.npy', y_pred_label) 
#np.save('./predicted_labels/rfecv_spatial_40_pred_loo.npy', y_pred_label)
#np.save('./predicted_labels/rfecv_morphological_40_pred_loo.npy', y_pred_label)
#np.save('./predicted_labels/morphological_pred_loo.npy', y_pred_label)
np.save('./predicted_labels/rfecv_ori_aal_40_pred_loo.npy', y_pred_label)
#np.save('./predicted_labels/ori_aal_pred_loo.npy', y_pred_label)
#np.save('./predicted_labels/rfecv_ori_tract_nrm_end_aal_pred_loo.npy', y_pred_label)
#np.save('./predicted_labels/ori_tract_nrm_end_aal_40_pred_loo.npy', y_pred_label)
#np.save('./predicted_labels/oskar_40_pred_loo.npy', y_pred_label)
#np.save('./predicted_labels/mod_tract_bin_end_aal_pred_loo.npy', y_pred_label)

# =========================== Feature Importance ================================ #

importances = np.round(np.mean(subject_feature_importances, axis=0),decimals=2) 
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(X_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[logging.info('Variable: {:30} Importance: {}'.format(*pair)) for pair in feature_importances]

logging.info(feature_type+" features with RF Regressor - Accuracy: %0.2f , MAE: %0.2f (+/- %0.2f)" %(np.mean(accuracy), np.mean(y_abs_error), np.std(y_abs_error)))

#x_range = np.arange(volumetric_spatial_features.shape[1])
#lesion_histogram = volumetric_spatial_features.sum(axis=0)
#fig, ax = plt.subplots()
#plt.bar(x_range, lesion_histogram)
#plt.show()

'''# AAL top 7 features
regions_importances = [subject_feature_importances[:,88],
                       subject_feature_importances[:,6],
                       subject_feature_importances[:,17],
                       subject_feature_importances[:,72],
                       subject_feature_importances[:,64],
                       subject_feature_importances[:,15],
                       subject_feature_importances[:,13]]
fig, ax = plt.subplots(1, 1)
ax.boxplot(regions_importances, showmeans=True)
ax.set_title('Region Importance', fontsize = 60)
plt.ylabel('Importance', fontsize = 40)
region_names = ['LITG', 'LMFG', 'RRO', 'LP', 'LAG', 'ORIFG', 'TRIFG'] 
plt.xticks(np.arange(1,8), region_names,  fontsize = 20)
plt.yticks(fontsize = 20)
plt.show()'''



'''# AAL selected features
regions_importances = [subject_feature_importances[:,6],
                       subject_feature_importances[:,3],
                       subject_feature_importances[:,0],
                       subject_feature_importances[:,2],
                       subject_feature_importances[:,1],
                       subject_feature_importances[:,4],
                       subject_feature_importances[:,5],
                       subject_feature_importances[:,7]]
fig, ax = plt.subplots(1, 1)
ax.boxplot(regions_importances, showmeans=True)
ax.set_title('Region Importance', fontsize = 60)
#plt.xlabel('Region', fontsize = 20)
plt.ylabel('Importance', fontsize = 40)
region_names = ['LITG',
                'RRO',
                'LMFG',
                'ORIFG',
                'TRIFG',
                'LAG',
                'LP',                
                'RITG'] 
plt.xticks(np.arange(1,9), region_names,  fontsize = 30)
plt.yticks(fontsize = 20)
plt.show()'''
