import numpy as np
import os
import SimpleITK as sitk
import csv
import paths


from utils import get_train_dataset, ReadImage, reshape_by_padding_upper_coords, find_list

atlas_name = 'aal'

stroke_mni_dir = os.path.join(paths.dsi_studio_path, 'gt_stroke')
stroke_mni_paths = [os.path.join(root, name) for root, dirs, files in os.walk(stroke_mni_dir) for name in files if name.endswith('nii.gz')]
stroke_mni_paths.sort()


train_dataset = get_train_dataset()
atlas_path = os.path.join(paths.dsi_studio_path, 'atlas', atlas_name+'.nii.gz')
atlas_img = sitk.ReadImage(atlas_path)
atlas_nda = sitk.GetArrayFromImage(atlas_img)
atlas_nda = reshape_by_padding_upper_coords(atlas_nda, (182,218,182), 0)
tf_atlas_nda = atlas_nda > 0 
binary_atlas_nda = np.zeros((182,218,182), dtype=np.float32)
binary_atlas_nda[tf_atlas_nda]=3.0
binary_atlas_img = sitk.GetImageFromArray(binary_atlas_nda)
sitk.WriteImage(binary_atlas_img,'aal_binary.nii.gz')
stroke_heatmap = np.zeros((182,218,182), dtype=np.float32)

for idx, subject_name in enumerate(train_dataset.keys()):
	subject_id = train_dataset[subject_name]['ID']
	stroke_mni_path = find_list(subject_id, stroke_mni_paths)
	stroke_mni_nda = ReadImage(stroke_mni_path)
	stroke_heatmap = stroke_heatmap+stroke_mni_nda

stroke_heatmap_img = sitk.GetImageFromArray(stroke_heatmap)
sitk.WriteImage(stroke_heatmap_img, 'stroke_heatmap.nii.gz')