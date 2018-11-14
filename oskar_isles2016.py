import medpy
from medpy.io import load, header, save
from medpy.features.intensity import intensities, local_mean_gauss, hemispheric_difference, local_histogram
import os
import paths
import utils
import numpy as np
import math
from skimage.morphology import dilation,disk
import nibabel as nib
import csv

def get_train_dataset():    
    gt_subject_paths = [os.path.join(root, name) for root, dirs, files in os.walk(paths.isles2017_training_dir) for name in files if '.OT.' in name and '__MACOSX' not in root and name.endswith('.nii')]
    gt_subject_paths.sort()
    # The CSV file for train dataset
    train_mRS_file = "ISLES2017_Training.csv"
    train_mRS_path = os.path.join(paths.isles2017_dir, train_mRS_file)
    assert(os.path.isfile(train_mRS_path))
    # Read CSV file for Train dataset
    train_dataset = {}
    with open(train_mRS_path, 'rU') as csv_file:
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

training_dataset = get_train_dataset()

mRS_gt = np.zeros((37,))

for idx, training_folder in enumerate(training_dataset.keys()):
    # adc direction    
    adc_temp = [os.path.join(root, name) for root, dirs, files in os.walk(os.path.join(paths.isles2017_training_dir,training_folder))
    for name in files if 'ADC' in name and 'MNI' not in name and name.endswith('.nii')]
    adc_dir = adc_temp[0]
    # Ground-truth mRS scores
    mRS_gt[idx] = training_dataset[training_folder]['mRS']
    # Ground-truth lesion mask
    stroke_temp = [os.path.join(root, name) for root, dirs, files in os.walk(os.path.join(paths.isles2017_training_dir, training_folder))
    for name in files if '.OT.' in name and 'MNI' not in name and name.endswith('.nii')]
    stroke_dir = stroke_temp[0]
    
    # find the three region
    band = 5.0
    stroke_mask, stroke_header = load(stroke_dir)
    img = nib.load(stroke_dir)
    #print(img.affine)
    adc_data, adc_header = load(adc_dir)


    pixel_spacing = header.get_pixel_spacing(stroke_header)
    x_band, y_band, z_band = math.ceil(band/pixel_spacing[0]), math.ceil(band/pixel_spacing[1]), math.ceil(band/pixel_spacing[2])
    print(x_band, y_band, z_band)
    #print(disk(x_band))
    
    dilated_stroke_mask = np.copy(stroke_mask)
    # dialation in 2D plane 
    for z_idx in range(stroke_mask.shape[2]):
        dilated_stroke_mask[:,:,z_idx] = dilation(stroke_mask[:,:,z_idx], disk(x_band))

    #print(np.count_nonzero(dilated_stroke_mask), np.count_nonzero(stroke_mask))
    brain_mask = np.zeros(stroke_mask.shape)
    brain_mask[adc_data>0] = 1
    #print(np.count_nonzero(brain_mask))
    
    first_region = stroke_mask
    second_region = dilated_stroke_mask - stroke_mask
    third_region = brain_mask - dilated_stroke_mask
    #print(np.count_nonzero(first_region), np.count_nonzero(second_region), np.count_nonzero(third_region))
    
    first_region_intensities = intensities(adc_data, mask=first_region)
    first_region_local_mean_gauss_3mm = local_mean_gauss(adc_data, sigma=3, voxelspacing=pixel_spacing, mask=first_region)
    first_region_local_mean_gauss_5mm = local_mean_gauss(adc_data, sigma=5, voxelspacing=pixel_spacing, mask=first_region)
    first_region_local_mean_gauss_7mm = local_mean_gauss(adc_data, sigma=7, voxelspacing=pixel_spacing, mask=first_region)
    first_region_hemispheric_difference = hemispheric_difference(adc_data, voxelspacing=pixel_spacing, mask=first_region)
    first_region_local_histogram= local_histogram(adc_data, bins=20, footprint=(x_band, y_band, z_band),  mask=first_region)
    print(np.mean(first_region_intensities), np.mean(first_region_local_mean_gauss_3mm), np.mean(first_region_local_mean_gauss_5mm), np.mean(first_region_local_mean_gauss_7mm), np.mean(first_region_hemispheric_difference))
    print(first_region_local_histogram)
'''
    hemispheric_difference_data = hemispheric_difference(adc_dir)


first_region_img = nib.Nifti1Image(first_region, img.affine)
nib.save(first_region_img, './first.nii')
second_region_img = nib.Nifti1Image(second_region, img.affine)
nib.save(second_region_img, './second.nii')
third_region_img = nib.Nifti1Image(third_region, img.affine)
nib.save(third_region_img, './third.nii')
'''
