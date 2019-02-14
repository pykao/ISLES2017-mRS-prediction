import os
import math
import medpy
import csv

import nibabel as nib
import numpy as np

from medpy.io import load, header, save
from medpy.features.intensity import intensities, local_mean_gauss, hemispheric_difference, local_histogram

from skimage.morphology import dilation,disk
from skimage.measure import regionprops, marching_cubes_classic, mesh_surface_area


import paths

def get_train_dataset(isles2017_dir, isles2017_training_dir):    
    gt_subject_paths = [os.path.join(root, name) for root, dirs, files in os.walk(isles2017_training_dir) for name in files if '.OT.' in name and '__MACOSX' not in root and name.endswith('.nii')]
    gt_subject_paths.sort()
    # The CSV file for train dataset
    train_mRS_file = "ISLES2017_Training.csv"
    train_mRS_path = os.path.join(isles2017_dir, train_mRS_file)
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

def twenty_two_statistics_of_feature(feature):
    statistics = np.zeros((1, 22))
    idx = 0 
    for p in np.linspace(0,100,10):
        statistics[0, idx] = np.percentile(feature, p)
        idx += 1
    statistics[0, idx] = np.std(feature)
    idx += 1
    statistics[0, idx] = np.var(feature)
    idx += 1
    statistics[0, idx:] = np.histogram(feature)[0]
    return statistics

def statistics_of_a_feature(image_features):
    for idx in range(image_features.shape[0]):
        if idx == 0:
            statistics = twenty_two_statistics_of_feature(image_features[idx,:])
        else:
            statistics = np.hstack((statistics, twenty_two_statistics_of_feature(image_features[idx,:])))
    return statistics

def statistics_of_features(first_region_image_features, second_region_image_features, third_region_image_features):
    first_region_statistics_of_features = statistics_of_a_feature(first_region_image_features)
    second_region_statistics_of_features = statistics_of_a_feature(second_region_image_features)
    third_region_statistics_of_features = statistics_of_a_feature(third_region_image_features)
    all_statistics_of_features = np.hstack((first_region_statistics_of_features, second_region_statistics_of_features))
    all_statistics_of_features = np.hstack((all_statistics_of_features, third_region_statistics_of_features))
    return all_statistics_of_features

def image_features(img, pixel_spacing, bands, mask):
    img_intensities = intensities(img, mask=mask).reshape(1, -1)
    img_local_mean_gauss_3mm = local_mean_gauss(img, sigma=3, voxelspacing=pixel_spacing, mask=mask).reshape(1, -1)
    all_image_features = np.vstack((img_intensities,img_local_mean_gauss_3mm))
    img_local_mean_gauss_5mm = local_mean_gauss(img, sigma=5, voxelspacing=pixel_spacing, mask=mask).reshape(1, -1)
    all_image_features = np.vstack((all_image_features,img_local_mean_gauss_5mm))
    img_local_mean_gauss_7mm = local_mean_gauss(img, sigma=7, voxelspacing=pixel_spacing, mask=mask).reshape(1, -1)
    all_image_features = np.vstack((all_image_features,img_local_mean_gauss_7mm))
    img_hemispheric_difference = hemispheric_difference(img, voxelspacing=pixel_spacing, mask=mask).reshape(1, -1)
    all_image_features = np.vstack((all_image_features,img_hemispheric_difference))
    img_local_histogram= local_histogram(img, bins=20, size=bands, mask=mask).reshape(20, -1)
    all_image_features = np.vstack((all_image_features,img_local_histogram))
    return all_image_features

def find_3d_surface(mask, voxel_spacing):
	verts, faces = marching_cubes_classic(volume=mask, spacing=voxel_spacing)
	return mesh_surface_area(verts, faces)

def find_3d_roundness(mask):
	mask_region_props = regionprops(mask.astype(int))
	mask_area = mask_region_props[0].area
	mask_equivDiameter = (6.0*mask_area/math.pi)**(1.0/3.0)
	mask_major_axis_length = mask_region_props[0].major_axis_length
	return mask_equivDiameter**2/mask_major_axis_length**2

isles2017_dir = paths.isles2017_dir
isles2017_training_dir = paths.isles2017_training_dir

training_dataset = get_train_dataset(isles2017_dir, isles2017_training_dir)

mRS_gt = np.zeros((37,))

all_features = np.zeros((37, 1662))

train_dataset_keys = training_dataset.keys()

sorted_train_dataset_keys = sorted(train_dataset_keys, key=lambda x:int(x[9:]))

for idx, training_folder in enumerate(sorted_train_dataset_keys):
    print(training_folder)
    # adc direction    
    adc_temp = [os.path.join(root, name) for root, dirs, files in os.walk(os.path.join(isles2017_training_dir,training_folder))
    for name in files if 'ADC' in name and 'MNI' not in name and name.endswith('.nii')]
    adc_dir = adc_temp[0]
    # Ground-truth mRS scores
    mRS_gt[idx] = training_dataset[training_folder]['mRS']
    # Ground-truth lesion mask
    stroke_temp = [os.path.join(root, name) for root, dirs, files in os.walk(os.path.join(isles2017_training_dir, training_folder))
    for name in files if '.OT.' in name and 'MNI' not in name and name.endswith('.nii')]
    stroke_dir = stroke_temp[0]
    
    # find the three region
    band = 5.0 
    stroke_mask, stroke_header = load(stroke_dir)
    adc_data, adc_header = load(adc_dir)


    pixel_spacing = header.get_pixel_spacing(stroke_header)
    voxel_volume = pixel_spacing[0]*pixel_spacing[1]*pixel_spacing[2]
    bands = (int(math.ceil(band/pixel_spacing[0])), int(math.ceil(band/pixel_spacing[1])), int(math.ceil(band/pixel_spacing[2])))
    
    dilated_stroke_mask = np.copy(stroke_mask)
    # dialation in 2D plane 
    for z_idx in range(stroke_mask.shape[2]):
        dilated_stroke_mask[:,:,z_idx] = dilation(stroke_mask[:,:,z_idx], disk(bands[0]))

    brain_mask = np.zeros(stroke_mask.shape)
    brain_mask[adc_data>0] = 1
    
    # Create three regions

    first_region = stroke_mask
    second_region = dilated_stroke_mask - stroke_mask
    third_region = brain_mask - dilated_stroke_mask

    # Extract 1650 image features
    first_region_image_features = image_features(adc_data, pixel_spacing, bands, first_region)
    second_region_image_features = image_features(adc_data, pixel_spacing, bands, second_region)
    third_region_image_features = image_features(adc_data, pixel_spacing, bands, third_region)
    all_features[idx, 0:1650] = statistics_of_features(first_region_image_features, second_region_image_features, third_region_image_features)


    # Extract 12 shape features
    first_region_props = regionprops(first_region.astype(int))
    first_region_area = first_region_props[0].area*voxel_volume
    first_region_equivDiameter = (6.0*first_region_area/math.pi)**(1.0/3.0)
    first_region_surface = find_3d_surface(first_region, pixel_spacing)
    first_region_roundness = find_3d_roundness(first_region)
    second_region_props = regionprops(second_region.astype(int))
    second_region_area = second_region_props[0].area*voxel_volume
    second_region_equivDiameter = (6.0*second_region_area/math.pi)**(1.0/3.0)
    second_region_surface = find_3d_surface(second_region, pixel_spacing)
    second_region_roundness = find_3d_roundness(second_region)
    third_region_props = regionprops(third_region.astype(int))
    third_region_area = third_region_props[0].area*voxel_volume
    third_region_equivDiameter = (6.0*third_region_area/math.pi)**(1.0/3.0)
    third_region_surface = find_3d_surface(third_region, pixel_spacing)
    third_region_roundness = find_3d_roundness(third_region)
    all_features[idx, 1650:1653] = first_region_area, second_region_area, third_region_area
    all_features[idx, 1653:1656] = first_region_equivDiameter, second_region_equivDiameter, third_region_equivDiameter
    all_features[idx, 1656:1659] = first_region_surface, second_region_surface, third_region_surface
    all_features[idx, 1659:1662] = first_region_roundness, second_region_roundness, third_region_roundness

# save oskar features and ground-truth mRS scores into numpy array
np.save('./oskar_features.npy', all_features)
np.save('./ISLES2017_gt.npy', mRS_gt)