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

# setup logs
log = os.path.join(os.getcwd(), 'log.txt')
fmt = '%(asctime)s %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt, filename=log)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter(fmt))
logging.getLogger('').addHandler(console)

logging.info('Extracting mRS scores...')
mRS_gt = extract_gt_mRS()

#logging.info('Extracting volumetric features...')
#volumetric_features = extract_volumetric_features()

logging.info('Extracting tractographic features...')
region_type='roi'
W_dsi_pass_histogram_features, W_nrm_pass_histogram_features, W_bin_pass_histogram_features, W_dsi_end_histogram_features, W_nrm_end_histogram_features, W_bin_end_histogram_features = extract_tractographic_features(region_type)

#logging.info('Extracting spatial features...')
#spatial_features = extract_spatial_features()

#logging.info('Extracting morphological features...')
#morphological_features = extract_morphological_features()


#logging.info('Extracting volumetric and spatial features...')
#HarvardOxfordSub_name = 'HarvardOxfordSub.nii.gz'
#HarvardOxfordCort_name = 'HarvardOxfordCort.nii.gz'
#aal_name = 'aal.nii.gz'
#JHU_WhiteMatter_labels_1mm_name = 'JHU-WhiteMatter-labels-1mm.nii.gz'
#MNI_name = 'MNI.nii.gz'
#OASIS_TRT_20_name = 'OASIS_TRT_20.nii.gz'

#volumetric_spatial_features = extract_volumetric_spatial_features(OASIS_TRT_20_name)
#print(volumetric_spatial_features.shape)
logging.info('Completed feature extraction...')



# Normalize Training Features
logging.info('Features normalization...')
scaler = StandardScaler()
normalized_W_dsi_pass_histogram_features = scaler.fit_transform(W_dsi_pass_histogram_features)
normalized_W_nrm_pass_histogram_features = scaler.fit_transform(W_nrm_pass_histogram_features)
normalized_W_bin_pass_histogram_features = scaler.fit_transform(W_bin_pass_histogram_features)
normalized_W_dsi_end_histogram_features = scaler.fit_transform(W_dsi_end_histogram_features)
normalized_W_nrm_end_histogram_features = scaler.fit_transform(W_nrm_end_histogram_features)
normalized_W_bin_end_histogram_features = scaler.fit_transform(W_bin_end_histogram_features)

#normalized_volumetric_features = scaler.fit_transform(volumetric_features)

#normalized_spatial_features = scaler.fit_transform(spatial_features)

#normalized_morphological_features = scaler.fit_transform(morphological_features)

#normalized_volumetric_spatial_features =scaler.fit_transform(volumetric_spatial_features)

logging.info('Completed features normalization...')



# Perforamce Feature Selection
# Remove features with low variance
logging.info('Remove features with low variance...')
sel = VarianceThreshold(0)
selected_normalized_W_dsi_pass_histogram_features = sel.fit_transform(normalized_W_dsi_pass_histogram_features)
selected_normalized_W_nrm_pass_histogram_features = sel.fit_transform(normalized_W_nrm_pass_histogram_features)
selected_normalized_W_bin_pass_histogram_features = sel.fit_transform(normalized_W_bin_pass_histogram_features)
selected_normalized_W_dsi_end_histogram_features = sel.fit_transform(normalized_W_dsi_end_histogram_features)
selected_normalized_W_nrm_end_histogram_features = sel.fit_transform(normalized_W_nrm_end_histogram_features)
selected_normalized_W_bin_end_histogram_features = sel.fit_transform(normalized_W_bin_end_histogram_features)

#selected_normalized_volumetric_features = sel.fit_transform(normalized_volumetric_features)

#selected_normalized_spatial_features = sel.fit_transform(normalized_spatial_features)

#selected_normalized_morphological_features = sel.fit_transform(normalized_morphological_features)

#selected_normalized_volumetric_spatial_features = sel.fit_transform(normalized_volumetric_spatial_features)

#logging.info('Using Volumetric and Spatial Features....')
logging.info('Using Tractographic Features')
X = selected_normalized_W_dsi_pass_histogram_features
#X = selected_normalized_W_nrm_pass_histogram_features
#X = selected_normalized_W_bin_pass_histogram_features
#X = selected_normalized_W_dsi_end_histogram_features
#X = selected_normalized_W_nrm_end_histogram_features
#X = selected_normalized_W_bin_end_histogram_features

#logging.info('Using volumetric features')
#X = selected_normalized_volumetric_features

#X = selected_normalized_spatial_features
#logging.info("Using morphological features...")
#X = selected_normalized_morphological_features

#X = selected_normalized_volumetric_spatial_features

y = mRS_gt

# Cross Validation Model
loo = LeaveOneOut()

accuracy = np.zeros((37,1), dtype=np.float32)
y_pred_label = np.zeros((37,1), dtype=int)


logging.info('RFECV Feature selection...')
# rfecv 
#estimator = LogisticRegression(penalty='l2', class_weight='balanced', random_state=0, multi_class='multinomial', solver='lbfgs', n_jobs=-1)
estimator = RandomForestRegressor(n_estimators=300, criterion='mse', random_state=0, n_jobs=2)
#estimator = RandomForestClassifier(n_estimators=100, n_jobs=-1)
#rfecv = RFECV(estimator, step=1, cv=loo, scoring='accuracy', n_jobs = -1)
rfecv = RFECV(estimator, step=1, cv=loo, scoring='neg_mean_absolute_error', n_jobs = 2)
rfecv.fit(X, y)
X_rfecv = rfecv.transform(X)
#logging.info('Logistic Regression, Optimal number of features: %d' % X_rfecv.shape[1])
logging.info('Random Forest Regressior, Optimal number of features: %d' % X_rfecv.shape[1])


logging.info('Predicting...')
idx = 0
for train_index, test_index in loo.split(X_rfecv):

	X_train, X_test = X_rfecv[train_index], X_rfecv[test_index]
	y_train, y_test = y[train_index], y[test_index]

	# Random Forest Classifier
	#rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1)
	#rfc.fit(X_train, y_train)
	#accuracy[idx] = rfc.score(X_test, y_test)
	#y_pred_label[idx] = rfc.predict(X_test)
	#print(rfc.predict(X_test))
	
	# Random Forest Regressior
	rfr = RandomForestRegressor(n_estimators=300, criterion='mse', random_state=0, n_jobs=2)
	rfr.fit(X_train, y_train)
	#accuracy[idx] = rfr.score(X_test, y_test)
	y_pred_label[idx] = np.round(rfr.predict(X_test))
	accuracy[idx] = accuracy_score(y_pred_label[idx], y_test)


	#lr = LogisticRegression(penalty='l2', class_weight='balanced', random_state=0, multi_class='multinomial', solver='lbfgs', n_jobs=-1)
	#lr.fit(X_train, y_train)
	#accuracy[idx] = lr.score(X_test, y_test)
	#y_pred_label[idx] = lr.predict(X_test)
	

	idx += 1

logging.info("Best Scores of features  - Using RFR - Accuracy: %0.4f (+/- %0.4f), MAE: %0.4f (+/- %0.4f)" %(np.mean(accuracy), np.std(accuracy), np.mean(np.absolute(y-y_pred_label)), np.std(np.absolute(y-y_pred_label))))