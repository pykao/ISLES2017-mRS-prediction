import paths
import csv
import os
import logging

import SimpleITK as sitk
import numpy as np

from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, LeaveOneOut
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFECV
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from utils import ReadImage, find_list, threshold_connectivity_matrix, weight_conversion, get_lesion_weights


# setup logs
log = os.path.join(os.getcwd(), 'log.txt')
fmt = '%(asctime)s %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt, filename=log)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter(fmt))
logging.getLogger('').addHandler(console)



# The CSV file for train dataset
train_mRS_file = "ISLES2017_Training.csv"
train_mRS_path = os.path.join(paths.isles2017_dir, train_mRS_file)
# The ground truth lesion in subject space
gt_subject_paths = [os.path.join(root, name) for root, dirs, files in os.walk(paths.isles2017_training_dir) for name in files if '.OT.' in name and '__MACOSX' not in root and name.endswith('.nii')]
# The connectivity matrices location 
connectivity_train_dir = os.path.join(paths.dsi_studio_path, 'connectivity', 'gt_stroke')
# pass type locations
connectivity_pass_files = [os.path.join(root, name) for root, dirs, files in os.walk(connectivity_train_dir) for name in files if 'count' in name and 'ncount' not in name and 'connectivity' in name  and 'pass' in name and name.endswith('.mat')]
connectivity_pass_files.sort()
# end type locations
connectivity_end_files = [os.path.join(root, name) for root, dirs, files in os.walk(connectivity_train_dir) for name in files if 'count' in name and 'ncount' not in name and 'connectivity' in name  and 'end' in name and name.endswith('.mat')]
connectivity_end_files.sort()
# The ground truth lesions in MNI space
stroke_mni_dir = os.path.join(paths.dsi_studio_path, 'gt_stroke')
stroke_mni_paths = [os.path.join(root, name) for root, dirs, files in os.walk(stroke_mni_dir) for name in files if name.endswith('nii.gz')]
stroke_mni_paths.sort()
assert(len(connectivity_pass_files) == len(connectivity_end_files) == len(stroke_mni_paths) == 43)
assert(os.path.isfile(train_mRS_path))



# Read CSV file for Train dataset
train_dataset = {}
with open(train_mRS_path, 'rt') as csv_file:
    csv_reader = csv.reader(csv_file)
    for line in csv_reader:
        if line[2] == '90': # 90 days 
            subject_name = line[0]
            gt_file = [file for file in gt_subject_paths if '/'+subject_name+'/' in file]
            if gt_file:
                train_dataset[subject_name]={}
                train_dataset[subject_name]['mRS'] = line[1]
                train_dataset[line[0]]['TICI'] = line[3]
                train_dataset[line[0]]['TSS'] = line[4]
                train_dataset[line[0]]['TTT'] = line[5]
                train_dataset[line[0]]['ID'] = gt_file[0][-10:-4]


# Ground truth 
mRS_gt = np.zeros(37)

# Feature initialization
# Tractographic Features
W_dsi_pass_histogram_features = np.zeros((37, 116), dtype=np.float32)
W_nrm_pass_histogram_features = np.zeros((37, 116), dtype=np.float32)
W_bin_pass_histogram_features = np.zeros((37, 116), dtype=np.float32)

W_dsi_end_histogram_features = np.zeros((37, 116), dtype=np.float32)
W_nrm_end_histogram_features = np.zeros((37, 116), dtype=np.float32)
W_bin_end_histogram_features = np.zeros((37, 116), dtype=np.float32)
# Volumetric Features
volumetric_features = np.zeros((37,1), dtype = int)


# Feature extraction
logging.info('Feature extraction...')
for idx, subject_name in enumerate(train_dataset.keys()):
    mRS_gt[idx] = train_dataset[subject_name]['mRS']
    print(train_dataset[subject_name]['TICI'])
    subject_id = train_dataset[subject_name]['ID']
    connectivity_pass_file = find_list(subject_id, connectivity_pass_files)
    connectivity_pass_obj = loadmat(connectivity_pass_file)
    weighted_connectivity_pass = threshold_connectivity_matrix(connectivity_pass_obj['connectivity'], 0)
    W_nrm_pass, W_bin_pass = weight_conversion(weighted_connectivity_pass)

    connectivity_end_file = find_list(subject_id, connectivity_end_files)
    connectivity_end_obj = loadmat(connectivity_end_file)
    weighted_connectivity_end = threshold_connectivity_matrix(connectivity_end_obj['connectivity'], 0)
    W_nrm_end, W_bin_end = weight_conversion(weighted_connectivity_end)

    stroke_mni_path = find_list(subject_id, stroke_mni_paths)

    #volumetric features
    stroke_mni_nda = ReadImage(stroke_mni_path)
    volumetric_features[idx] = np.count_nonzero(stroke_mni_nda)

    # Get the lesion weights
    lesion_weights = get_lesion_weights(stroke_mni_path)

    # weighted connectivity histogram
    W_dsi_pass_histogram_features[idx, :] = np.multiply(np.sum(weighted_connectivity_pass, axis=0), lesion_weights)
    W_nrm_pass_histogram_features[idx, :] = np.multiply(np.sum(W_nrm_pass, axis=0), lesion_weights)
    W_bin_pass_histogram_features[idx, :] = np.multiply(np.sum(W_bin_pass, axis=0), lesion_weights)

    W_dsi_end_histogram_features[idx, :] = np.multiply(np.sum(weighted_connectivity_end, axis=0), lesion_weights)
    W_nrm_end_histogram_features[idx, :] = np.multiply(np.sum(W_nrm_end, axis=0), lesion_weights)
    W_bin_end_histogram_features[idx, :] = np.multiply(np.sum(W_bin_end, axis=0), lesion_weights)
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

normalized_volumetric_features = scaler.fit_transform(volumetric_features)

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

selected_normalized_volumetric_features = sel.fit_transform(normalized_volumetric_features)

print(selected_normalized_W_dsi_pass_histogram_features.shape)
print(selected_normalized_W_nrm_pass_histogram_features.shape)
print(selected_normalized_W_bin_pass_histogram_features.shape)
print(selected_normalized_W_dsi_end_histogram_features.shape)
print(selected_normalized_W_nrm_end_histogram_features.shape)
print(selected_normalized_W_bin_end_histogram_features.shape)

print(selected_normalized_volumetric_features.shape)



#logging.info('Using volumetric features')
logging.info('Using Tractographic Features')
X = selected_normalized_W_dsi_pass_histogram_features
#X = selected_normalized_W_nrm_pass_histogram_features
#X = selected_normalized_W_bin_pass_histogram_features
#X = selected_normalized_W_dsi_end_histogram_features
#X = selected_normalized_W_nrm_end_histogram_features
#X = selected_normalized_W_bin_end_histogram_features

#X = selected_normalized_volumetric_features

print(X.shape)

y = mRS_gt

# Cross Validation Model
loo = LeaveOneOut()

accuracy = np.zeros((37,1), dtype=np.float32)
y_pred_label = np.zeros((37,1), dtype=int)


logging.info('RFECV Feature selection')
# rfecv 
#estimator = LogisticRegression(penalty='l2', class_weight='balanced', random_state=0, multi_class='multinomial', solver='lbfgs', n_jobs=-1)
estimator = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0, n_jobs=-1)
rfecv = RFECV(estimator, step=1, cv=loo, scoring='neg_mean_absolute_error', n_jobs = -1)
rfecv.fit(X, y)
X_rfecv = rfecv.transform(X)
#logging.info('Logistic Regression, Optimal number of features: %d' % X_rfecv.shape[1])
logging.info('Random Forest Regressior, Optimal number of features: %d' % X_rfecv.shape[1])


logging.info('Prediction')
idx = 0
for train_index, test_index in loo.split(X_rfecv):

	X_train, X_test = X_rfecv[train_index], X_rfecv[test_index]
	y_train, y_test = y[train_index], y[test_index]
	
	# Regressior
	rfr = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0, n_jobs=-1)
	rfr.fit(X_train, y_train)
	y_pred_label[idx] = rfr.predict(X_test)
	print(rfr.predict(X_test))
	#lr = LogisticRegression(penalty='l2', class_weight='balanced', random_state=0, multi_class='multinomial', solver='lbfgs', n_jobs=-1)
	#lr.fit(X_train, y_train)
	#accuracy[idx] = lr.score(X_test, y_test)
	#y_pred_label[idx] = lr.predict(X_test)
	idx += 1

#logging.info("Best Scores of features  - Using LR - Accuracy: %0.4f (+/- %0.4f), MAE: %0.4f (+/- %0.4f)" %(np.mean(accuracy), np.std(accuracy), np.mean(np.absolute(y-y_pred_label)), np.std(np.absolute(y-y_pred_label))))