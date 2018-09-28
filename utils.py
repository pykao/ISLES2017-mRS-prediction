import SimpleITK as sitk
import numpy as np
import os

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
