import medpy
from medpy.io import load
from medpy.io import header
import os
import paths
import utils
import numpy as np
import math
from skimage.morphology import dilation,disk


training_dataset = utils.get_train_dataset()

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

    adc_data, adc_header = load(adc_dir)


    pixel_spacing = header.get_pixel_spacing(stroke_header)
    x_band, y_band, z_band = math.ceil(band/pixel_spacing[0]), math.ceil(band/pixel_spacing[1]), math.ceil(band/pixel_spacing[2])
    #print(x_band, y_band, z_band)
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
    print(np.count_nonzero(first_region), np.count_nonzero(second_region), np.count_nonzero(third_region))
