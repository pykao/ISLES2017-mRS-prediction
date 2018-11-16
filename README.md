# Prediction of Modified Rankin Scale (mRS) in Stroke Patients with Tractographic Features

## How to run the codes

These codes are wrote in Python 3.6

### 1. Change the paths.py

set `isles2017_dir` to the path you save the clinical parameters .csv file

set `isles2017_training_dir` to the path you save ISLES2017 training data

set `mni152_1mm_path` to the path store the MNI152_T1_1mm_brain.nii.gz

set `dsi_studio_path` to the dsistudio directory

### 2. registerBrain.py

This script registers the MR adc images from subject space to MNI152 1mm space and registers the brain lesions to MNI 152 1mm space

### 3. gt_fiber.py

This script generates the fiber tracts for the subject

You are able to change the region properties on line #61

`--seed=`: create tract seed inside the lesion 

`--roi=`: seed in the whole brain region and find the fiber tracts pass through the lesion region

`--roa=`: seed in the whole brain region and find the fiber tracts avoid the lesion region 

## 4. predict_loo.py

Perform mRS prediction on features extracted from the lesion region with-leave one-out cross-validation on the ISLES2017 training dataset

## 5. predict_skf.py

Perform mRS prediction on features extracted from the lesion region with stratified 3-fold cross-validation on the ISLES2017 training dataset

## 6. oskar_features.py (run in python 2.7)

Extract the features descirbe in the ISLES2016 winning paper

## 7. oskar_predict_loo.py

Perform mR prediction on oskar's features with leave-one-out cross-validation on the ISLES2017 training dataset

## utils.py

Provide you different types of features and tools for processing the brain images

## analysis.py

Provide you the confusion matrix and the p-value

