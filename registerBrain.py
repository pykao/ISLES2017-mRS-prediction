import paths
import os
import subprocess

from multiprocessing import Pool
from natsort import natsorted

def ISLES2017TrainingADCPaths(isles_train_dir=paths.isles2017_training_dir):
    adc_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(isles_train_dir) 
    for name in files if 'MR_ADC' in name and '__MACOSX' not in root and name.endswith('.nii')]
    return natsorted(adc_filepaths)

def ISLES2017TrainingGTPaths(isles_train_dir=paths.isles2017_training_dir):
    gt_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(isles_train_dir) 
    for name in files if '.OT.' in name and '__MACOSX' not in root and name.endswith('.nii')]
    return natsorted(gt_filepaths)

def ISLES2017TrainingInVol2RefVolPaths(isles_train_dir=paths.isles2017_training_dir):
    mat_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(isles_train_dir) 
    for name in files if 'invol2refvol' in name and name.endswith('.mat')]
    return natsorted(mat_filepaths)

def RegisterBrain(adc_path):
    path, adc_file = os.path.split(adc_path)
    mni_path = os.path.join(path, 'MNI152')
    if not os.path.exists(mni_path):
        os.makedirs(mni_path)
    subject2mni_mat = os.path.join(mni_path, adc_file[:adc_file.index('.nii')]+'_MNI152_1mm_invol2refvol.mat')
    mni2subject_mat = os.path.join(mni_path, adc_file[:adc_file.index('.nii')]+'_MNI152_1mm_refvol2invol.mat')
    adc_mni = os.path.join(mni_path, adc_file[:adc_file.index('.nii')]+'_MNI152_1mm.nii.gz')
    #print(adc_path, paths.mni152_1mm_path, subject2mni_mat, mni2subject_mat)
    # Create the affine transformation matrix from subject space to MNI152 1mm space
    subprocess.call(["flirt", "-in", adc_path, "-ref", paths.mni152_1mm_path, "-omat", subject2mni_mat])
    # Output ADC in MNI 152 1mm space
    #subprocess.call(["flirt", "-in", adc_path, "-ref", paths.mni152_1mm_path, "-out", adc_mni, "-omat", subject2mni_mat])
    subprocess.call(["convert_xfm", "-omat", mni2subject_mat, "-inverse", subject2mni_mat])
    print('Finish this subject: %s'  %adc_path)

def Lesions2MNI152(GTInVol, omat):
	new_name_prob = GTInVol[:-4] + "_prob_MNI152_T1_1mm.nii.gz"
	new_name_bin = GTInVol[:-4] + "_MNI152_T1_1mm.nii.gz"
	subprocess.call(["flirt", "-in", GTInVol, "-ref", refVol, "-out", new_name_prob, "-init", omat, "-applyxfm"])
	subprocess.call(["fslmaths", new_name_prob, "-thr", "0.5", "-bin", new_name_bin])




def Lesions2MNI152_star(lesion_dirs):
	return Lesions2MNI152(*lesion_dirs)

global refVol

refVol = paths.mni152_1mm_path

adc_filepaths = ISLES2017TrainingADCPaths()

gt_filepaths = ISLES2017TrainingGTPaths()

omat_filepaths = ISLES2017TrainingInVol2RefVolPaths()

pool = Pool(6)

pool.map(RegisterBrain, adc_filepaths)

pool.map(Lesions2MNI152_star, zip(gt_filepaths, omat_filepaths))
