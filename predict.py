import paths
import csv
import os
import SimpleITK as sitk
import numpy as np

from scipy.io import loadmat

def find_list(subject_id, list):
	files = [file for file in list if subject_id in file]
	return files[0]

train_mRS_file = "ISLES2017_Training.csv"

train_mRS_path = os.path.join(paths.isles2017_dir, train_mRS_file)

gt_subject_paths = [os.path.join(root, name) for root, dirs, files in os.walk(paths.isles2017_training_dir) for name in files if '.OT.' in name and '__MACOSX' not in root and name.endswith('.nii')]

connectivity_train_dir = os.path.join(paths.dsi_studio_path, 'connectivity', 'gt_stroke')

connectivity_pass_files = [os.path.join(root, name) for root, dirs, files in os.walk(connectivity_train_dir) for name in files if 'count' in name and 'ncount' not in name and 'connectivity' in name  and 'pass' in name and name.endswith('.mat')]
connectivity_pass_files.sort()

connectivity_end_files = [os.path.join(root, name) for root, dirs, files in os.walk(connectivity_train_dir) for name in files if 'count' in name and 'ncount' not in name and 'connectivity' in name  and 'end' in name and name.endswith('.mat')]
connectivity_end_files.sort()

stroke_mni_dir = os.path.join(paths.dsi_studio_path, 'gt_stroke')

stroke_mni_paths = [os.path.join(root, name) for root, dirs, files in os.walk(stroke_mni_dir) for name in files if name.endswith('nii.gz')]
stroke_mni_paths.sort()

print(len(connectivity_pass_files), len(connectivity_end_files), len(stroke_mni_paths))



assert(os.path.isfile(train_mRS_path))

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

mRS_gt = np.zeros(37)

W_dsi_pass_histogram_features = np.zeros((37, 116), dtype=np.float32)
W_nrm_pass_histogram_features = np.zeros((37, 116), dtype=np.float32)
W_bin_pass_histogram_features = np.zeros((37, 116), dtype=np.float32)

W_dsi_end_histogram_features = np.zeros((37, 116), dtype=np.float32)
W_nrm_end_histogram_features = np.zeros((37, 116), dtype=np.float32)
W_bin_end_histogram_features = np.zeros((37, 116), dtype=np.float32)


for i, subject_name in enumerate(train_dataset.keys()):
	mRS_gt[i] = train_dataset[subject_name]['mRS']
	subject_id = train_dataset[subject_name]['ID']
	print(find_list(subject_id, connectivity_pass_files))
	print(train_dataset[subject_name]['mRS'], subject_id)