# Prediction of Modified Rankin Scale (mRS) in Stroke Patients with Tractographic Features

## Dataset

[Ischemic Stroke Lesion Segmentation (ISLES) 2017](http://www.isles-challenge.org/ISLES2017/)

## Dependencies

Python 3.6

## Required Softwares

For image registration, you need to download [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki).

For fiber tracking and building connectivity matrix, you need to download [DSI Studio](http://dsi-studio.labsolver.org/).

## How to run the codes

### 1. Change the pathes inside paths.py

set `isles2017_dir` to the path you store the clinical parameters file (ISLES2017_Training.csv)

set `isles2017_training_dir` to the path you save ISLES2017 training data (ISLES2017/train)

set `mni152_1mm_path` to the path store the MNI152_T1_1mm_brain.nii.gz

set `dsi_studio_path` to the dsistudio directory

### 2. Run registerBrain.py

This script registers the MR-ADC image and the brain lesion from the subject space to MNI152 1mm space.

The outputs: 

ADC_MNI152_1mm.nii.gz, ADC_MNI152_1mm_invol2refvol.mat, and ADC_MNI152_1mm_refvol2invol.mat under ADC's directory

OT_MNI152_1mm.nii.gz and OT_prob_MNI152_T1_1mm.nii.gz under brain lesion's directory  

### 3. Run gt_fiber.py

This script generates the fiber tracts for the subject. 

We seed in the whole brain region and find the fiber tracts pass through the lesion region



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

