import SimpleITK as sitk
import numpy as np
import os
import paths
import csv

from scipy.io import loadmat

def ReadImage(path):
    ''' This code returns the numpy nd array for a MR image at path'''
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)

def find_list(subject_id, list):
    files = [file for file in list if subject_id in file]
    return files[0]

def reshape_by_padding_upper_coords(image, new_shape, pad_value=None):
    shape = tuple(list(image.shape))
    new_shape = tuple(np.max(np.concatenate((shape, new_shape)).reshape((2,len(shape))), axis=0))
    if pad_value is None:
        if len(shape)==2:
            pad_value = image[0,0]
        elif len(shape)==3:
            pad_value = image[0, 0, 0]
        else:
            raise ValueError("Image must be either 2 or 3 dimensional")
    res = np.ones(list(new_shape), dtype=image.dtype) * pad_value
    if len(shape) == 2:
        res[0:0+int(shape[0]), 0:0+int(shape[1])] = image
    elif len(shape) == 3:
        res[0:0+int(shape[0]), 0:0+int(shape[1]), 0:0+int(shape[2])] = image
    return res

def threshold_connectivity_matrix(connectivity_matrix, threshold=0.01):
    ''' threshold the connectiivty matrix in order to remove the noise'''
    thresholded_connectivity_matrix= np.copy(connectivity_matrix)
    thresholded_connectivity_matrix[connectivity_matrix <= threshold*np.amax(connectivity_matrix)] = 0
    return thresholded_connectivity_matrix


def weight_conversion(W):
    ''' convert to the normalized version and binary version'''
    W_bin = np.copy(W)
    W_bin[W!=0]=1
    W_nrm = np.copy(W)
    W_nrm = W_nrm/np.amax(np.absolute(W))
    return W_nrm, W_bin

def get_lesion_weights(whole_tumor_mni_path):
    ''' get the weight vector'''
    #print(whole_tumor_mni_path)
    aal_path = os.path.join(paths.dsi_studio_path, 'atlas', 'aal.nii.gz')
    aal_nda = ReadImage(aal_path)
    aal_182_218_182 = reshape_by_padding_upper_coords(aal_nda, (182,218,182), 0)
    whole_tumor_mni_nda = ReadImage(whole_tumor_mni_path)
    weights = np.zeros(int(np.amax(aal_182_218_182)), dtype=float)
    for bp_number in range(int(np.amax(aal_182_218_182))):
        mask = np.zeros(aal_182_218_182.shape, aal_182_218_182.dtype)
        mask[aal_182_218_182==(bp_number+1)]=1
        bp_size = float(np.count_nonzero(mask))
        whole_tumor_in_bp = np.multiply(mask, whole_tumor_mni_nda)
        whole_tumor_in_bp_size = float(np.count_nonzero(whole_tumor_in_bp))
        weights[bp_number] = whole_tumor_in_bp_size/bp_size
    return weights

def get_train_dataset():    
    gt_subject_paths = [os.path.join(root, name) for root, dirs, files in os.walk(paths.isles2017_training_dir) for name in files if '.OT.' in name and '__MACOSX' not in root and name.endswith('.nii')]
    # The CSV file for train dataset
    train_mRS_file = "ISLES2017_Training.csv"
    train_mRS_path = os.path.join(paths.isles2017_dir, train_mRS_file)
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
    return train_dataset   

def extract_gt_mRS():
    # Ground truth 
    mRS_gt = np.zeros((37, 1))
    train_dataset = get_train_dataset()
    for idx, subject_name in enumerate(train_dataset.keys()):
        mRS_gt[idx] = train_dataset[subject_name]['mRS']
    return mRS_gt

def extract_volumetric_features():
    # The ground truth lesions in MNI space
    stroke_mni_dir = os.path.join(paths.dsi_studio_path, 'gt_stroke')
    stroke_mni_paths = [os.path.join(root, name) for root, dirs, files in os.walk(stroke_mni_dir) for name in files if name.endswith('nii.gz')]
    stroke_mni_paths.sort()
    assert(len(stroke_mni_paths) == 43)
    # Volumetric Features
    volumetric_features = np.zeros((37,1))
    train_dataset = get_train_dataset()
    for idx, subject_name in enumerate(train_dataset.keys()):
        subject_id = train_dataset[subject_name]['ID']
        stroke_mni_path = find_list(subject_id, stroke_mni_paths)

        #volumetric features
        stroke_mni_nda = ReadImage(stroke_mni_path)
        volumetric_features[idx] = np.count_nonzero(stroke_mni_nda)
    return volumetric_features


def extract_tractographic_features():
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
    train_dataset = get_train_dataset()
    # Tractographic Features
    W_dsi_pass_histogram_features = np.zeros((37, 116), dtype=np.float32)
    W_nrm_pass_histogram_features = np.zeros((37, 116), dtype=np.float32)
    W_bin_pass_histogram_features = np.zeros((37, 116), dtype=np.float32)

    W_dsi_end_histogram_features = np.zeros((37, 116), dtype=np.float32)
    W_nrm_end_histogram_features = np.zeros((37, 116), dtype=np.float32)
    W_bin_end_histogram_features = np.zeros((37, 116), dtype=np.float32)

    for idx, subject_name in enumerate(train_dataset.keys()):
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

        # Get the lesion weights
        lesion_weights = get_lesion_weights(stroke_mni_path)

        # weighted connectivity histogram
        W_dsi_pass_histogram_features[idx, :] = np.multiply(np.sum(weighted_connectivity_pass, axis=0), lesion_weights)
        W_nrm_pass_histogram_features[idx, :] = np.multiply(np.sum(W_nrm_pass, axis=0), lesion_weights)
        W_bin_pass_histogram_features[idx, :] = np.multiply(np.sum(W_bin_pass, axis=0), lesion_weights)

        W_dsi_end_histogram_features[idx, :] = np.multiply(np.sum(weighted_connectivity_end, axis=0), lesion_weights)
        W_nrm_end_histogram_features[idx, :] = np.multiply(np.sum(W_nrm_end, axis=0), lesion_weights)
        W_bin_end_histogram_features[idx, :] = np.multiply(np.sum(W_bin_end, axis=0), lesion_weights)
    return W_dsi_pass_histogram_features, W_nrm_pass_histogram_features, W_bin_pass_histogram_features, W_dsi_end_histogram_features, W_nrm_end_histogram_features, W_bin_end_histogram_features