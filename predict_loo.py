import paths
import csv
import os
import logging

import SimpleITK as sitk
import numpy as np

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
from utils import extract_volumetric_spatial_features, extract_morphological_features

from xgboost import XGBRegressor

# =========================================== setup logs ====================================== #
log = os.path.join(os.getcwd(), 'log_loo.txt')
fmt = '%(asctime)s %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt, filename=log)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter(fmt))
logging.getLogger('').addHandler(console)

# ============================================ The target ===================================== #
logging.info('Extracting mRS scores...')
mRS_gt = extract_gt_mRS()

# ======================================== Feature Extraction ================================= #

#logging.info('Extracting volumetric features...')
#volumetric_features, volumetric_list = extract_volumetric_features()

#logging.info('Extracting spatial features...')
#spatial_features, spatial_list = extract_spatial_features()

#logging.info('Extracting morphological features...')
#morphological_features, morphological_list = extract_morphological_features()


logging.info('Extracting volumetric and spatial features...')
#atlas_name = 'HarvardOxfordSub'
#atlas_name = 'HarvardOxfordCort'
atlas_name = 'aal'
#atlas_name = 'JHU-WhiteMatter-labels-1mm'
#atlas_name = 'MNI'
#atlas_name = 'OASIS_TRT_20'
volumetric_spatial_features, volumetric_spatial_list = extract_volumetric_spatial_features(atlas_name)

# original atlas
volumetric_spatial_features = volumetric_spatial_features[:, 1:]
del volumetric_spatial_list[0]
#print(volumetric_spatial_features.shape, len(volumetric_spatial_list))

#logging.info('Extracting tractographic features...')
#region_type='roi'
#logging.info(region_type)
#W_dsi_pass, W_nrm_pass, W_bin_pass, W_dsi_end, W_nrm_end, W_bin_end, tract_list = extract_tractographic_features(region_type)


logging.info('Completed feature extraction...')



# ==============================  Feature Normalization ========================================= #
logging.info('Features normalization...')
scaler = StandardScaler()

#normalized_volumetric_features = scaler.fit_transform(volumetric_features)

#normalized_spatial_features = scaler.fit_transform(spatial_features)

#normalized_morphological_features = scaler.fit_transform(morphological_features)

normalized_volumetric_spatial_features =scaler.fit_transform(volumetric_spatial_features)

##normalized_W_dsi_pass_histogram_features = scaler.fit_transform(W_dsi_pass)
##normalized_W_nrm_pass = scaler.fit_transform(W_nrm_pass)

#normalized_W_bin_pass = scaler.fit_transform(W_bin_pass)

#normalized_W_dsi_end = scaler.fit_transform(W_dsi_end)

##normalized_W_nrm_end_histogram_features = scaler.fit_transform(W_nrm_end)

#normalized_W_bin_end = scaler.fit_transform(W_bin_end)

#logging.info('Completed features normalization...')

# ======================================== Remove features with all zeros ======================= # 

# Perforamce Feature Selection
# Remove features with all zeros
logging.info('Remove features with all zeros...')
sel = VarianceThreshold(0)

#selected_normalized_volumetric_features = sel.fit_transform(normalized_volumetric_features)
#selected_volumetric_list = [name for idx, name in enumerate(volumetric_list) if sel.get_support()[idx]]

#selected_normalized_spatial_features = sel.fit_transform(normalized_spatial_features)
#selected_spatial_list = [name for idx, name in enumerate(spatial_list) if sel.get_support()[idx]]

#selected_normalized_morphological_features = sel.fit_transform(normalized_morphological_features)
#selected_morphological_list = [name for idx, name in enumerate(morphological_list) if sel.get_support()[idx]]

selected_normalized_volumetric_spatial_features = sel.fit_transform(normalized_volumetric_spatial_features)
selected_volumetric_spatial_list = [name for idx, name in enumerate(volumetric_spatial_list) if sel.get_support()[idx]]

##selected_normalized_W_dsi_pass_histogram_features = sel.fit_transform(normalized_W_dsi_pass_histogram_features)
##selected_normalized_W_nrm_pass = sel.fit_transform(normalized_W_nrm_pass)
##selected_W_nrm_pass_list = [name for idx, name in enumerate(tract_list) if sel.get_support()[idx]]

#selected_normalized_W_bin_pass= sel.fit_transform(normalized_W_bin_pass)
#selected_W_bin_pass_list = [name for idx, name in enumerate(tract_list) if sel.get_support()[idx]]

#selected_normalized_W_dsi_end = sel.fit_transform(normalized_W_dsi_end)
#selected_W_dsi_end_list = [name for idx, name in enumerate(tract_list) if sel.get_support()[idx]]

##selected_normalized_W_nrm_end_histogram_features = sel.fit_transform(normalized_W_nrm_end_histogram_features)

#selected_normalized_W_bin_end= sel.fit_transform(normalized_W_bin_end)
#selected_W_bin_end_list = [name for idx, name in enumerate(tract_list) if sel.get_support()[idx]]

# ======================================= Select which feature to use ===================================== #

#logging.info('Using volumetric features')
#X = selected_normalized_volumetric_features
#feature_list = selected_volumetric_list

#logging.info('Using spatial features')
#X = selected_normalized_spatial_features
#feature_list = selected_spatial_list

#logging.info("Using morphological features...")
#X = selected_normalized_morphological_features
#feature_list = selected_morphological_list

logging.info('Using Volumetric and Spatial Features....')
X = selected_normalized_volumetric_spatial_features
feature_list = selected_volumetric_spatial_list

#logging.info('Using Tractographic Features')
##X = selected_normalized_W_dsi_pass_histogram_features
##X = selected_normalized_W_nrm_pass
##feature_list = selected_W_nrm_pass_list
#X = selected_normalized_W_bin_pass
#feature_list = selected_W_bin_pass_list

#X = selected_normalized_W_dsi_end
#feature_list = selected_W_dsi_end_list

##X = selected_normalized_W_nrm_end_histogram_features
#X = selected_normalized_W_bin_end
#feature_list = selected_W_bin_end_list

# ========================================== RFECV feature selection ====================================== #

# Cross Validation Model
#loo = LeaveOneOut()
#estimator = RandomForestRegressor(n_estimators=300, max_depth=3, random_state=1989, n_jobs=-1)
#rfecv = RFECV(estimator, step=1, cv=loo, scoring='neg_mean_absolute_error', n_jobs=-1)
##rfecv = RFECV(estimator,step=1, cv=loo, scoring='neg_mean_squared_error', n_jobs=-1)

#rfecv_selected_normalized_volumetric_features = rfecv.fit_transform(selected_normalized_volumetric_features, mRS_gt)
#rfecv_selected_volumetric_list = [name for idx, name in enumerate(selected_volumetric_list) if rfecv.get_support()[idx]]
#print(len(rfecv_selected_volumetric_list))

#rfecv_selected_normalized_spatial_features = rfecv.fit_transform(selected_normalized_spatial_features, mRS_gt)
#rfecv_selected_spatial_list = [name for idx, name in enumerate(selected_spatial_list) if rfecv.get_support()[idx]]
#print(len(rfecv_selected_spatial_list))

#rfecv_selected_normalized_morphological_features = rfecv.fit_transform(selected_normalized_morphological_features, mRS_gt)
#rfecv_selected_morphological_list = [name for idx, name in enumerate(selected_morphological_list) if rfecv.get_support()[idx]]
#print(len(rfecv_selected_morphological_list))

#rfecv_selected_normalized_volumetric_spatial_features = rfecv.fit_transform(selected_normalized_volumetric_spatial_features, mRS_gt)
#rfecv_selected_volumetric_spatial_list = [name for idx, name in enumerate(selected_volumetric_spatial_list) if rfecv.get_support()[idx]]
#print(len(rfecv_selected_volumetric_spatial_list))

#rfecv_selected_normalized_W_dsi_end = rfecv.fit_transform(selected_normalized_W_dsi_end, mRS_gt)
#rfecv_selected_W_dsi_end_list = [name for idx, name in enumerate(selected_W_dsi_end_list) if rfecv.get_support()[idx]]
#print(len(rfecv_selected_W_dsi_end_list))

#rfecv_all_features = np.concatenate((rfecv_selected_normalized_volumetric_features, 
#	rfecv_selected_normalized_spatial_features, 
#	rfecv_selected_normalized_morphological_features, 
#	rfecv_selected_normalized_volumetric_spatial_features, 
#	rfecv_selected_normalized_W_dsi_end), axis=1)
#rfecv_feature_list = rfecv_selected_volumetric_list + rfecv_selected_spatial_list + rfecv_selected_morphological_list + rfecv_selected_volumetric_spatial_list + rfecv_selected_W_dsi_end_list
#X_rfecv = rfecv_all_features
#print(rfecv_all_features.shape, len(rfecv_feature_list))

# =============================================== Start Prediction ========================================= #

y = mRS_gt

# Cross Validation Model
loo = LeaveOneOut()

accuracy = np.zeros((37,1), dtype=np.float32)
y_pred_label = np.zeros((37,1), dtype=np.float32)
y_abs_error = np.zeros((37,1), dtype=np.float32)
##y_pred_proba = np.zeros((37,5), dtype=np.float32)

# ======================== Volumetric, Spatial, Morphological, Volumetric Spatial Features ================= #
#estimator = RandomForestRegressor(n_estimators=300, max_depth=3, random_state=1989, n_jobs=-1)
#estimator = RandomForestRegressor(n_estimators=300, random_state=1989, n_jobs=-1)
estimator = RandomForestClassifier(n_estimators=300, max_depth=3, random_state=1989, n_jobs=-1, class_weight="balanced")
logging.info('RFECV Feature selection...')
# rfecv 
##rfecv = RFECV(estimator, step=1, cv=rskf, scoring='neg_mean_absolute_error', n_jobs=-1)
##rfecv = RFECV(estimator, step=1, cv=loo, scoring='neg_mean_squared_error', n_jobs=-1)
rfecv = RFECV(estimator, step=1, cv=loo, scoring='neg_mean_absolute_error', n_jobs=-1)
X_rfecv = rfecv.fit_transform(X, y)
logging.info('Random Froest Classifier, Optimal number of features: %d' % X_rfecv.shape[1])
#logging.info('Random Froest Regressor, Optimal number of features: %d' % X_rfecv.shape[1])
rfecv_feature_list = [name for idx, name in enumerate(feature_list) if rfecv.get_support()[idx]]

##X_rfecv=X
##rfecv_feature_list = feature_list

subject_feature_importances = np.zeros((37,len(rfecv_feature_list)), dtype=np.float32)

logging.info('Predicting...')
idx = 0
for train_index, test_index in loo.split(X_rfecv):
    X_train, X_test = X_rfecv[train_index], X_rfecv[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # RF Regression
    #estimator = RandomForestRegressor(n_estimators=300, max_depth=3, random_state=1989, n_jobs=-1)
    #estimator = RandomForestRegressor(n_estimators=300, random_state=1989, n_jobs=-1)    
    # RF Classifier with balanced weights
    estimator = RandomForestClassifier(n_estimators=300, max_depth=3, random_state=1989, n_jobs=-1, class_weight="balanced")
    estimator.fit(X_train, y_train)
    subject_importance = estimator.feature_importances_
    subject_feature_importances[idx,:] = subject_importance
    y_pred_label[idx] = np.round(estimator.predict(X_test))
    #y_pred_proba[idx, :] = estimator.predict_proba(X_test)
    accuracy[idx] = accuracy_score(np.round(estimator.predict(X_test)), y_test)
    y_abs_error[idx] = np.absolute(y_pred_label[idx]-y_test)
    idx += 1

logging.info("Best Scores of features  - Using RF Classifier - Accuracy: %0.4f , MAE: %0.4f (+/- %0.4f)" %(np.mean(accuracy), np.mean(y_abs_error), np.std(y_abs_error)))
#logging.info("Best Scores of features  - Using RF Regressor - Accuracy: %0.4f , MAE: %0.4f (+/- %0.4f)" %(np.mean(accuracy), np.mean(y_abs_error), np.std(y_abs_error)))

#np.save('./rfc_volumetric_pred_loo.npy', y_pred_label) 
#np.save('./rfc_spatial_pred_loo.npy', y_pred_label)
#np.save('./rfc_morphological_pred_loo.npy', y_pred_label)
#np.save('./rfc_HarvardOxfordSub_pred_loo.npy', y_pred_label)
#np.save('./rfr_HarvardOxfordCort_ori_pred_loo.npy', y_pred_label)
np.save('./rfc_aal_ori_pred_loo.npy', y_pred_label)
#np.save('./rfc_JHU_pred_loo.npy', y_pred_label)
#np.save('./rfc_MNI_pred_loo.npy', y_pred_label)
#np.save('./rfc_OASIS_pred_loo.npy', y_pred_label)
#np.save('./rfc_aal_Wbin_pass_roi_pred_loo.npy', y_pred_label)
#np.save('./rfc_aal_Wdsi_end_roi_pred_loo.npy', y_pred_label)


importances = np.round(np.mean(subject_feature_importances, axis=0),decimals=2) 
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(rfecv_feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[logging.info('Variable: {:30} Importance: {}'.format(*pair)) for pair in feature_importances]


'''
# ================================= Tractographic Features =================================================== #

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
    #estimator = RandomForestRegressor(n_estimators=300, max_depth=3, random_state=1989, n_jobs=-1)
    estimator = RandomForestClassifier(n_estimators=300, max_depth=3, random_state=1989, n_jobs=-1, class_weight="balanced_subsample")
    #rfecv = RFECV(estimator, step=1, cv=loo, scoring='neg_mean_squared_error', n_jobs=-1)
    rfecv = RFECV(estimator, step=1, cv=loo, scoring='neg_mean_absolute_error', n_jobs=-1)
    rfecv.fit(X, y)
    X_rfecv = rfecv.transform(X)
    logging.info('Random Forest Classifier, Optimal number of features: %d' % X_rfecv.shape[1])
    #X_rfecv = normalized_tractographic_feature

    logging.info('Predicting...')
    idx = 0
    for train_index, test_index in loo.split(X_rfecv):
        X_train, X_test = X_rfecv[train_index], X_rfecv[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Random Forest Regressor
        #rfr = RandomForestRegressor(n_estimators=300, max_depth=3, random_state=1989, n_jobs=-1)
        #rfr.fit(X_train, y_train)
        # Random Froest Classifier
        rfr = RandomForestClassifier(n_estimators=300, max_depth=3, random_state=1989, n_jobs=-1, class_weight="balanced_subsample")
        rfr.fit(X_train, y_train)
        y_pred_label[idx] = np.round(rfr.predict(X_test))
        accuracy[idx] = accuracy_score(np.round(rfr.predict(X_test)), y_test)
        y_abs_error[idx] = np.absolute(y_pred_label[idx]-y_test)
        idx += 1
    logging.info("Best Scores of features - Using Random Forest Regressor - Accuracy: %0.4f , MAE: %0.4f (+/- %0.4f)" %(np.mean(accuracy), np.mean(y_abs_error), np.std(y_abs_error)))
    '''