import SimpleITK as sitk
import numpy as np
import os
import paths
import csv
import math

from scipy.io import loadmat
from skimage.measure import regionprops, marching_cubes_classic, mesh_surface_area

def divide_hcp(connectivity_matrix, hcp_connectivity):
    ''' divide the connectivity matrix by the hcp matrix'''
    assert(connectivity_matrix.shape == hcp_connectivity.shape)
    output_matrix = np.zeros(connectivity_matrix.shape)
    for i in range(connectivity_matrix.shape[0]):
        for j in range(connectivity_matrix.shape[1]):
            if hcp_connectivity[i,j] != 0:
                output_matrix[i,j] = connectivity_matrix[i,j]/hcp_connectivity[i,j]
    return output_matrix


def get_hcp_connectivity_matrice(hcp_connectivity_matrices_path = paths.hcp_connectivity_matrices_path):
    '''Get the pass-type and end-type connectivity matrices from HCP1021 subjects'''
    end_matrix_path = os.path.join(hcp_connectivity_matrices_path, 'HCP1021.1mm.fib.gz.aal.count.end.connectivity.mat')
    
    pass_matrix_path = os.path.join(hcp_connectivity_matrices_path, 'HCP1021.1mm.fib.gz.aal.count.pass.connectivity.mat')

    end_obj = loadmat(end_matrix_path)

    end_matrix = end_obj['connectivity']

    pass_obj = loadmat(pass_matrix_path)

    pass_matrix = pass_obj['connectivity']

    return pass_matrix, end_matrix

def ReadImage(path):
    ''' This code returns the numpy nd array for a MR image at path'''
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)

def find_list(subject_id, list):
    ''' this is used to find the stroke lesion for a subject name '''
    files = [file for file in list if subject_id in file]
    return files[0]

def find_3d_surface(mask, voxel_spacing=(1.0,1.0,1.0)):
    ''' find the surface for a 3D object '''
    verts, faces = marching_cubes_classic(volume=mask, spacing=voxel_spacing)
    return mesh_surface_area(verts, faces)

def find_3d_roundness(mask):
    ''' find the roundess of a 3D object '''
    mask_region_props = regionprops(mask.astype(int))
    mask_area = mask_region_props[0].area
    mask_equivDiameter = (6.0*mask_area/math.pi)**(1.0/3.0)
    mask_major_axis_length = mask_region_props[0].major_axis_length
    return mask_equivDiameter**2/mask_major_axis_length**2

def reshape_by_padding_upper_coords(image, new_shape, pad_value=None):
    ''' reshape the 3d matrix '''
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

# ======================= Tools for connectivity matrix ============================================= #

def threshold_connectivity_matrix(connectivity_matrix, threshold=0.01):
    ''' threshold the connectiivty matrix in order to remove the noise'''
    thresholded_connectivity_matrix= np.copy(connectivity_matrix)
    thresholded_connectivity_matrix[connectivity_matrix <= threshold*np.amax(connectivity_matrix)] = 0.0
    return thresholded_connectivity_matrix


def weight_conversion(W):
    ''' convert to the normalized version and binary version'''
    W_bin = np.copy(W)
    W_bin[W!=0]=1
    W_nrm = np.copy(W)
    W_nrm = W_nrm/np.amax(np.absolute(W))
    return W_nrm, W_bin

def get_lesion_weights(stroke_mni_path):
    ''' get the weight vector(workshop paper)'''
    aal_path = os.path.join(paths.dsi_studio_path, 'atlas', 'aal.nii.gz')
    aal_nda = ReadImage(aal_path)
    aal_182_218_182 = reshape_by_padding_upper_coords(aal_nda, (182,218,182), 0)
    stroke_mni_nda = ReadImage(stroke_mni_path)
    weights = np.zeros(int(np.amax(aal_182_218_182)), dtype=float)
    for bp_number in range(int(np.amax(aal_182_218_182))):
        mask = np.zeros(aal_182_218_182.shape, aal_182_218_182.dtype)
        mask[aal_182_218_182==(bp_number+1)]=1
        bp_size = float(np.count_nonzero(mask))
        stroke_in_bp = np.multiply(mask, stroke_mni_nda)
        stroke_in_bp_size = float(np.count_nonzero(stroke_in_bp))
        weights[bp_number] = stroke_in_bp_size/bp_size
        #weights[bp_number] = stroke_in_bp_size
    return weights

def get_modified_lesion_weights(stroke_mni_path):
    ''' get the modified weight vector'''
    aal_path = os.path.join(paths.dsi_studio_path, 'atlas', 'aal.nii.gz')
    aal_nda = ReadImage(aal_path)
    aal_182_218_182 = reshape_by_padding_upper_coords(aal_nda, (182,218,182), 0)
    stroke_mni_nda = ReadImage(stroke_mni_path)
    stroke_volume = float(np.count_nonzero(stroke_mni_nda))
    weights = np.zeros(int(np.amax(aal_182_218_182)), dtype=float)
    for bp_number in range(int(np.amax(aal_182_218_182))):
        mask = np.zeros(aal_182_218_182.shape, aal_182_218_182.dtype)
        mask[aal_182_218_182==(bp_number+1)]=1
        #bp_size = float(np.count_nonzero(mask))
        stroke_in_bp = np.multiply(mask, stroke_mni_nda)
        stroke_volume_in_bp = float(np.count_nonzero(stroke_in_bp))
        #weights[bp_number] = 1.0 + stroke_volume_in_bp/stroke_volume
        weights[bp_number] = stroke_volume_in_bp/stroke_volume
    #remaining_volume = stroke_volume - np.sum(weights)
    #print(remaining_volume)
    return weights

def get_train_dataset():    
    '''Give you the training dataset'''
    gt_subject_paths = [os.path.join(root, name) for root, dirs, files in os.walk(paths.isles2017_training_dir) for name in files if '.OT.' in name and '__MACOSX' not in root and name.endswith('.nii')]
    gt_subject_paths.sort()
    # The CSV file for train dataset
    train_mRS_file = "ISLES2017_Training.csv"
    train_mRS_path = os.path.join(paths.isles2017_dir, train_mRS_file)
    assert(os.path.isfile(train_mRS_path))
    # Read CSV file for Train dataset
    train_dataset = {}
    with open(train_mRS_path, 'rt') as csv_file:
        csv_reader = csv.reader(csv_file)
        for line in csv_reader:
            if line[2] == '90' or line[2] == '88' or line[2] == '96' or line[2] == '97': # 90 days 
                subject_name = line[0]
                gt_file = [file for file in gt_subject_paths if '/'+subject_name+'/' in file]
                if gt_file:
                    train_dataset[subject_name]={}
                    train_dataset[subject_name]['mRS'] = line[1]
                    train_dataset[line[0]]['TICI'] = line[3]
                    train_dataset[line[0]]['TSS'] = line[4]
                    train_dataset[line[0]]['TTT'] = line[5]
                    train_dataset[line[0]]['ID'] = gt_file[0][-10:-4]
                    train_dataset[line[0]]['tracts'] = line[6]
    return train_dataset

# Get the mRS for training subject from training_1 to training_48
def extract_gt_mRS():
    '''extract the mRS for training subjects from training_1 to training_48'''  
    mRS_gt = np.zeros((40, ))
    train_dataset = get_train_dataset()
    for idx, subject_name in enumerate(train_dataset.keys()):
        mRS_gt[idx] = train_dataset[subject_name]['mRS']
    return mRS_gt

def extract_tract_features():
	''' extract number of tracts'''
	train_dataset = get_train_dataset()
	tracts = np.zeros((40, 1))
	for idx, subject_name in enumerate(train_dataset.keys()):
		tracts[idx] = train_dataset[subject_name]['tracts']
	return tracts, ['tracts']


# Extract the volume of stroke in MNI152 space
def extract_volumetric_features():
    # The ground truth lesions in MNI space
    volumetric_list = ["volume"]
    stroke_mni_dir = os.path.join(paths.dsi_studio_path, 'gt_stroke')
    stroke_mni_paths = [os.path.join(root, name) for root, dirs, files in os.walk(stroke_mni_dir) for name in files if name.endswith('nii.gz')]
    stroke_mni_paths.sort()
    assert(len(stroke_mni_paths) == 43)
    # Volumetric Features
    volumetric_features = np.zeros((40,1))
    train_dataset = get_train_dataset()
    for idx, subject_name in enumerate(train_dataset.keys()):
        subject_id = train_dataset[subject_name]['ID']
        stroke_mni_path = find_list(subject_id, stroke_mni_paths)

        #volumetric features
        stroke_mni_nda = ReadImage(stroke_mni_path)
        volumetric_features[idx] = np.count_nonzero(stroke_mni_nda)
    return volumetric_features, volumetric_list

def extract_spatial_features():
    # The ground truth lesions in MNI space
    stroke_mni_dir = os.path.join(paths.dsi_studio_path, 'gt_stroke')
    stroke_mni_paths = [os.path.join(root, name) for root, dirs, files in os.walk(stroke_mni_dir) for name in files if name.endswith('nii.gz')]
    stroke_mni_paths.sort()
    assert(len(stroke_mni_paths) == 43)
    spatial_list = ["centroid_z", "centroid_y", "centroid_x"]
    # Volumetric Features
    spatial_features = np.zeros((40,3))
    train_dataset = get_train_dataset()
    for idx, subject_name in enumerate(train_dataset.keys()):
        subject_id = train_dataset[subject_name]['ID']
        stroke_mni_path = find_list(subject_id, stroke_mni_paths)
        stroke_mni_nda = ReadImage(stroke_mni_path)
        stroke_regions = regionprops(stroke_mni_nda.astype(int))
        stroke_centroid = stroke_regions[0].centroid
        spatial_features[idx, :] = stroke_centroid
    return spatial_features, spatial_list

def extract_morphological_features():
    # The ground truth lesions in MNI space
    stroke_mni_dir = os.path.join(paths.dsi_studio_path, 'gt_stroke')
    stroke_mni_paths = [os.path.join(root, name) for root, dirs, files in os.walk(stroke_mni_dir) for name in files if name.endswith('nii.gz')]
    stroke_mni_paths.sort()
    assert(len(stroke_mni_paths) == 43)
    morphological_list = ["major", "minor", "major/minor", "surface", "solidity", "roundness"]
    # Volumetric Features
    morphological_features = np.zeros((40,6), dtype=np.float32)
    train_dataset = get_train_dataset()
    for idx, subject_name in enumerate(train_dataset.keys()):
        subject_id = train_dataset[subject_name]['ID']
        stroke_mni_path = find_list(subject_id, stroke_mni_paths)
        stroke_mni_nda = ReadImage(stroke_mni_path)
        stroke_regions = regionprops(stroke_mni_nda.astype(int))
        stroke_major_axis_length = stroke_regions[0].major_axis_length
        stroke_minor_axis_length = stroke_regions[0].minor_axis_length
        stroke_surface = find_3d_surface(stroke_mni_nda.astype(int))
        stroke_roundness = find_3d_roundness(stroke_mni_nda.astype(int))
        morphological_features[idx, :] = stroke_major_axis_length, stroke_minor_axis_length, stroke_major_axis_length/stroke_minor_axis_length, stroke_surface, stroke_regions[0].solidity, stroke_roundness
    return morphological_features, morphological_list

def extract_tractographic_features(weight_type, aal_regions=116):
    # The ground truth lesion in subject space
    gt_subject_paths = [os.path.join(root, name) for root, dirs, files in os.walk(paths.isles2017_training_dir) for name in files if '.OT.' in name and '__MACOSX' not in root and name.endswith('.nii')]
    # New connectivity matrices location
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
    tractographic_list = ["tract_aal_"+str(i) for i in range(1, aal_regions+1)]
    assert(len(connectivity_pass_files) == len(connectivity_end_files) == len(stroke_mni_paths) == 43)
    train_dataset = get_train_dataset()
    # Tractographic Features
    W_dsi_pass_histogram_features = np.zeros((40, aal_regions), dtype=np.float32)
    W_nrm_pass_histogram_features = np.zeros((40, aal_regions), dtype=np.float32)
    W_bin_pass_histogram_features = np.zeros((40, aal_regions), dtype=np.float32)

    W_dsi_end_histogram_features = np.zeros((40, aal_regions), dtype=np.float32)
    W_nrm_end_histogram_features = np.zeros((40, aal_regions), dtype=np.float32)
    W_bin_end_histogram_features = np.zeros((40, aal_regions), dtype=np.float32)

    for idx, subject_name in enumerate(train_dataset.keys()):
        subject_id = train_dataset[subject_name]['ID']
        connectivity_pass_file = find_list(subject_id, connectivity_pass_files)
        connectivity_pass_obj = loadmat(connectivity_pass_file)
        thresholded_connectivity_pass = threshold_connectivity_matrix(connectivity_pass_obj['connectivity'], 0)
        W_nrm_pass, W_bin_pass = weight_conversion(thresholded_connectivity_pass)

        connectivity_end_file = find_list(subject_id, connectivity_end_files)
        connectivity_end_obj = loadmat(connectivity_end_file)
        thresholded_connectivity_end = threshold_connectivity_matrix(connectivity_end_obj['connectivity'], 0)
        W_nrm_end, W_bin_end = weight_conversion(thresholded_connectivity_end)

        stroke_mni_path = find_list(subject_id, stroke_mni_paths)

        # =================================== Weight Vector ========================================== #
        # Get the lesion weights
        if 'ori' in weight_type:
            lesion_weights = get_lesion_weights(stroke_mni_path)
        # Get the modified lesion weights
        if 'mod' in weight_type:
            lesion_weights = get_modified_lesion_weights(stroke_mni_path)
        # No weight
        if 'one' in weight_type:
            lesion_weights = np.ones((1,aal_regions), dtype=np.float32)


        # weighted connectivity histogram
        W_dsi_pass_histogram_features[idx, :] = np.multiply(np.sum(thresholded_connectivity_pass, axis=0), lesion_weights)
        W_nrm_pass_histogram_features[idx, :] = np.multiply(np.sum(W_nrm_pass, axis=0), lesion_weights)
        W_bin_pass_histogram_features[idx, :] = np.multiply(np.sum(W_bin_pass, axis=0), lesion_weights)

        W_dsi_end_histogram_features[idx, :] = np.multiply(np.sum(thresholded_connectivity_end, axis=0), lesion_weights)
        W_nrm_end_histogram_features[idx, :] = np.multiply(np.sum(W_nrm_end, axis=0), lesion_weights)
        W_bin_end_histogram_features[idx, :] = np.multiply(np.sum(W_bin_end, axis=0), lesion_weights)

    return W_dsi_pass_histogram_features, W_nrm_pass_histogram_features, W_bin_pass_histogram_features, W_dsi_end_histogram_features, W_nrm_end_histogram_features, W_bin_end_histogram_features, tractographic_list


def extract_volumetric_spatial_features(atlas_name):
    '''extract volumetric spatial features'''
    stroke_mni_dir = os.path.join(paths.dsi_studio_path, 'gt_stroke')
    stroke_mni_paths = [os.path.join(root, name) for root, dirs, files in os.walk(stroke_mni_dir) for name in files if name.endswith('nii.gz')]
    stroke_mni_paths.sort()
    train_dataset = get_train_dataset()
    atlas_path = os.path.join(paths.dsi_studio_path, 'atlas', atlas_name+'.nii.gz')
    atlas_nda = ReadImage(atlas_path)
    if atlas_name == 'aal':
        atlas_nda = reshape_by_padding_upper_coords(atlas_nda, (182,218,182), 0)
    volumetric_spatial_features = np.zeros((40, int(np.amax(atlas_nda))+1), dtype=float)
    for idx, subject_name in enumerate(train_dataset.keys()):
        subject_id = train_dataset[subject_name]['ID']
        stroke_mni_path = find_list(subject_id, stroke_mni_paths)
        stroke_mni_nda = ReadImage(stroke_mni_path)
        whole_stroke_volume = float(np.count_nonzero(stroke_mni_nda))
        for bp_number in range(1, int(np.amax(atlas_nda)+1)):
            mask = np.zeros(atlas_nda.shape, atlas_nda.dtype)
            mask[atlas_nda==(bp_number)]=1
            stroke_in_bp = np.multiply(mask, stroke_mni_nda)
            stroke_in_bp_volume = np.count_nonzero(stroke_in_bp)
            volumetric_spatial_features[idx, bp_number] = stroke_in_bp_volume
        total_stroke_volume_bp = np.sum(volumetric_spatial_features[idx, :])
        volumetric_spatial_features[idx, 0] = whole_stroke_volume - total_stroke_volume_bp
    volumetric_spatial_list =['volume_'+atlas_name+'_'+str(i) for i in range(0, int(np.amax(atlas_nda)+1))]
    return volumetric_spatial_features, volumetric_spatial_list

def extract_modified_volumetric_spatial_features(atlas_name):
    '''extract volumetric spatial features considering the total volume of the stroke lesion'''
    stroke_mni_dir = os.path.join(paths.dsi_studio_path, 'gt_stroke')
    stroke_mni_paths = [os.path.join(root, name) for root, dirs, files in os.walk(stroke_mni_dir) for name in files if name.endswith('nii.gz')]
    stroke_mni_paths.sort()
    train_dataset = get_train_dataset()
    atlas_path = os.path.join(paths.dsi_studio_path, 'atlas', atlas_name+'.nii.gz')
    atlas_nda = ReadImage(atlas_path)
    if atlas_name == 'aal':
        atlas_nda = reshape_by_padding_upper_coords(atlas_nda, (182,218,182), 0)
    modified_volumetric_spatial_features = np.zeros((40, int(np.amax(atlas_nda))), dtype=float)
    for idx, subject_name in enumerate(train_dataset.keys()):
        subject_id = train_dataset[subject_name]['ID']
        stroke_mni_path = find_list(subject_id, stroke_mni_paths)
        stroke_mni_nda = ReadImage(stroke_mni_path)
        whole_stroke_volume = float(np.count_nonzero(stroke_mni_nda))
        for bp_number in range(1, int(np.amax(atlas_nda))+1):
            mask = np.zeros(atlas_nda.shape, atlas_nda.dtype)
            mask[atlas_nda==(bp_number)]=1
            stroke_in_bp = np.multiply(mask, stroke_mni_nda)
            stroke_in_bp_volume = float(np.count_nonzero(stroke_in_bp))
            modified_volumetric_spatial_features[idx, bp_number-1] = stroke_in_bp_volume / whole_stroke_volume
    volumetric_spatial_list =['volume_'+atlas_name+'_'+str(i) for i in range(1, int(np.amax(atlas_nda))+1)]
    assert((len(volumetric_spatial_list))==modified_volumetric_spatial_features.shape[1])
    return modified_volumetric_spatial_features, volumetric_spatial_list


def extract_new_tractographic_features(weight_type, aal_regions=116):
    # The ground truth lesion in subject space
    gt_subject_paths = [os.path.join(root, name) for root, dirs, files in os.walk(paths.isles2017_training_dir) for name in files if '.OT.' in name and '__MACOSX' not in root and name.endswith('.nii')]
    # New connectivity matrices location
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
    tractographic_list = ["tract_aal_"+str(i) for i in range(1, aal_regions+1)]
    assert(len(connectivity_pass_files) == len(connectivity_end_files) == len(stroke_mni_paths) == 43)
    train_dataset = get_train_dataset()
    # Tractographic Features
    W_pass_histogram_features = np.zeros((40, aal_regions), dtype=np.float32)
    W_end_histogram_features = np.zeros((40, aal_regions), dtype=np.float32)

    for idx, subject_name in enumerate(train_dataset.keys()):
        HCP_pass, HCP_end = get_hcp_connectivity_matrice()
        subject_id = train_dataset[subject_name]['ID']
        connectivity_pass_file = find_list(subject_id, connectivity_pass_files)
        connectivity_pass_obj = loadmat(connectivity_pass_file)
        connectivity_pass_matrix = connectivity_pass_obj['connectivity']
        #normalized_pass_matrix = divide_hcp(connectivity_pass_matrix, HCP_pass)

        connectivity_end_file = find_list(subject_id, connectivity_end_files)
        connectivity_end_obj = loadmat(connectivity_end_file)
        connectivity_end_matrix = connectivity_end_obj['connectivity']
        #normalized_end_matrix = divide_hcp(connectivity_pass_matrix, HCP_end)


        stroke_mni_path = find_list(subject_id, stroke_mni_paths)

        # =================================== Weight Vector ========================================== #
        # Get the lesion weights
        if 'ori' in weight_type:
            lesion_weights = get_lesion_weights(stroke_mni_path)
        # Get the modified lesion weights
        if 'mod' in weight_type:
            lesion_weights = get_modified_lesion_weights(stroke_mni_path)
        # No weight
        if 'one' in weight_type:
            lesion_weights = np.ones((1,aal_regions), dtype=np.float32)
        
        normalized_pass_matrix = np.divide(np.sum(connectivity_pass_matrix, axis=0), np.sum(HCP_pass, axis=0))
        normalized_end_matrix = np.divide(np.sum(connectivity_end_matrix, axis=0), np.sum(HCP_end, axis=0))

        # weighted connectivity histogram
        W_pass_histogram_features[idx, :] = np.multiply(normalized_pass_matrix, lesion_weights)
        
        W_end_histogram_features[idx, :] = np.multiply(normalized_end_matrix, lesion_weights)

    return W_pass_histogram_features, W_end_histogram_features, tractographic_list
