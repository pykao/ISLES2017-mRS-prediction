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
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from utils import ReadImage, find_list, threshold_connectivity_matrix, weight_conversion, get_lesion_weights, get_train_dataset
from utils import extract_gt_mRS, extract_volumetric_features, extract_tractographic_features, extract_spatial_features
from utils import extract_volumetric_spatial_features, extract_morphological_features, extract_modified_volumetric_spatial_features

from xgboost import XGBRegressor

# =========================================== setup logs ====================================== #
log = os.path.join(os.getcwd(), 'log_new.txt')
fmt = '%(asctime)s %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt, filename=log)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter(fmt))
logging.getLogger('').addHandler(console)

# ======================================= Parameters Setting ================================== #
use_original_feature = True
rfecv_feature_selection = False


# ======================================== Select Feature  ==================================== #

#feature_type = 'volumetric'
#feature_type = 'spatial'
#feature_type = 'morphological'

# lesion in brain parcellation regions and the remaining region
#feature_type = 'volumetric_spatial'
# lesion in brain parcellation regions 
#feature_type = 'original_volumetric_spatial'
#atlas_name = 'HarvardOxfordSub'
#atlas_name = 'HarvardOxfordCort'
#atlas_name = 'aal'
#atlas_name = 'JHU-WhiteMatter-labels-1mm'
#atlas_name = 'MNI'
#atlas_name = 'OASIS_TRT_20'

feature_type = 'oskar'

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
    if 'ori' not in feature_type:
        logging.info('Included the lesion outside brain parcellation regions...')
    features, features_list = extract_volumetric_spatial_features(atlas_name)
    if 'ori' in feature_type:
    	logging.info('Removing the lesion outside brain parcellation regions...')
    	features = features[:,1:]
    	del features_list[0]
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
    features = np.load('./features/oskar_features.npy')
    features_list = list(range(1662))

print(features[0,:])

# =============================== Save Original Features ======================================= #

logging.info('Saving Features...')
if not os.path.exists('./features/'):
	os.mkdir('./features/')
#np.save('./features/volumetric_features.npy', features)
#np.save('./features/spatial_features.npy', features)
#np.save('./features/morphological_features.npy', features)
#np.save('./features/ori_aal_features.npy', features)
#np.save('./features/ori_tract_nrm_end_aal_features.npy', features)


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
    X, X_list = features, features_list
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

accuracy = np.zeros((37,1), dtype=np.float32)
y_pred_label = np.zeros((37,1), dtype=np.float32)
y_abs_error = np.zeros((37,1), dtype=np.float32)
##y_pred_proba = np.zeros((37,5), dtype=np.float32)

subject_feature_importances = np.zeros((37,len(X_list)), dtype=np.float32)

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
#np.save('./predicted_labels/rfecv_volumetric_pred_loo.npy', y_pred_label) 
#np.save('./predicted_labels/rfecv_spatial_pred_loo.npy', y_pred_label)
#np.save('./predicted_labels/rfecv_morphological_pred_loo.npy', y_pred_label)
#np.save('./predicted_labels/morphological_pred_loo.npy', y_pred_label)
#np.save('./predicted_labels/rfecv_ori_aal_pred_loo.npy', y_pred_label)
#np.save('./predicted_labels/ori_aal_pred_loo.npy', y_pred_label)
#np.save('./predicted_labels/rfecv_ori_tract_nrm_end_aal_pred_loo.npy', y_pred_label)
#np.save('./predicted_labels/ori_tract_nrm_end_aal_pred_loo.npy', y_pred_label)
np.save('./predicted_labels/oskar_pred_loo.npy', y_pred_label)


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

# AAL top 5 features
#region_7_importances = subject_feature_importances[:,6]
#region_14_importances = subject_feature_importances[:,13]
#region_41_importances = subject_feature_importances[:,40]
#region_65_importances = subject_feature_importances[:,64]
#region_89_importances = subject_feature_importances[:,88]
#regions_importances = [region_89_importances, region_7_importances, region_65_importances, region_41_importances, region_14_importances]
#fig, ax = plt.subplots(1, 1)
#ax.boxplot(regions_importances, showmeans=True)
#ax.set_title('Region Importance', fontsize = 30)
#plt.ylabel('Importance', fontsize = 20)
#region_names = ['Left\ninferior temporal gyrus', 'Left\nmiddle frontal gyrus', 'Left\nangular gyrus', 'Left amygdala', 'Triangular part of\nright inferior frontal gyrus'] 
#plt.xticks(np.arange(1,6), region_names,  fontsize = 20)
#plt.yticks(fontsize = 20)
#plt.show()

# AAL selected features
#region_16_importances = subject_feature_importances[:,0]
#region_18_importances = subject_feature_importances[:,1]
#region_41_importances = subject_feature_importances[:,2]
#region_89_importances = subject_feature_importances[:,3]
#regions_importances = [region_89_importances, region_18_importances, region_41_importances, region_16_importances]
#fig, ax = plt.subplots(1, 1)
#ax.boxplot(regions_importances, showmeans=True)
#ax.set_title('Region Importance', fontsize = 30)
##plt.xlabel('Region', fontsize = 20)
#plt.ylabel('Importance', fontsize = 20)
#region_names = ['Left \ninferior temporal gyrus', 'Right \nRolandic operculum', 'Left amygdala', 'Orbital part of \nright inferior frontal gyrus'] 
#plt.xticks(np.arange(1,5), region_names,  fontsize = 20)
#plt.yticks(fontsize = 20)
#plt.show()


'''# ================================= Tractographic Features =================================================== #

tractographic_feature_list = ['W_dis_pass', 'W_nrm_pass', 'W_bin_pass', 'W_dsi_end', 'W_nrm_end', 'W_bin_end']

all_tractographic_features = np.stack((W_dsi_pass, W_nrm_pass, W_bin_pass, W_dsi_end, W_nrm_end, W_bin_end), axis=-1)

for i in np.arange(6):
    tractographic_feature = all_tractographic_features[:, :, i]
    logging.info('Using Tractographic Features')
    logging.info(tractographic_feature_list[i])
    
    logging.info('Features normalization...')
    scaler = StandardScaler()
    normalized_tractographic_feature = scaler.fit_transform(tractographic_feature)
    logging.info('Completed features normalization...')
    
    logging.info('Remove features with low variance...')
    sel = VarianceThreshold(0)
    selected_normalized_tractographic_feature = sel.fit_transform(normalized_tractographic_feature)
    
    X = selected_normalized_tractographic_feature
    y = mRS_gt

    # Cross Validation Model
    loo = LeaveOneOut()
    
    accuracy = np.zeros((37,1), dtype=np.float32)
    y_pred_label = np.zeros((37,1), dtype=np.float32)
    y_abs_error = np.zeros((37,1), dtype=np.float32)
    
    logging.info('RFECV Feature selection...')
    # rfecv 
    estimator = RandomForestRegressor(n_estimators=300, max_depth=3, random_state=1989, n_jobs=-1)
    #estimator = RandomForestClassifier(n_estimators=300, max_depth=3, random_state=1989, n_jobs=-1, class_weight="balanced_subsample")
    #rfecv = RFECV(estimator, step=1, cv=loo, scoring='neg_mean_squared_error', n_jobs=-1)
    rfecv = RFECV(estimator, step=1, cv=loo, scoring='neg_mean_absolute_error', n_jobs=-1)
    rfecv.fit(X, y)
    X_rfecv = rfecv.transform(X)
    logging.info('Random Forest Regressor, Optimal number of features: %d' % X_rfecv.shape[1])
    #X_rfecv = normalized_tractographic_feature

    logging.info('Predicting...')
    idx = 0
    for train_index, test_index in loo.split(X_rfecv):
        X_train, X_test = X_rfecv[train_index], X_rfecv[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Random Forest Regressor
        rfr = RandomForestRegressor(n_estimators=300, max_depth=3, random_state=1989, n_jobs=-1)
        rfr.fit(X_train, y_train)
        # Random Froest Classifier
        #rfr = RandomForestClassifier(n_estimators=300, max_depth=3, random_state=1989, n_jobs=-1, class_weight="balanced_subsample")
        #rfr.fit(X_train, y_train)
        y_pred_label[idx] = np.round(rfr.predict(X_test))
        accuracy[idx] = accuracy_score(np.round(rfr.predict(X_test)), y_test)
        y_abs_error[idx] = np.absolute(y_pred_label[idx]-y_test)
        idx += 1
    logging.info("Best Scores of features - Using Random Forest Regressor - Accuracy: %0.4f , MAE: %0.4f (+/- %0.4f)" %(np.mean(accuracy), np.mean(y_abs_error), np.std(y_abs_error)))
'''
