import paths
import os


def ISLES2017TrainingADCPaths(isles_train_dir=paths.isles2017_training_dir):
	adc_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(isles_train_dir) 
	for name in files if 'MR_ADC' in name and name.endswith('.nii')]
	adc_filepaths.sort()
	return adc_filepaths

adc_filepaths = ISLES2017TrainingADCPaths()

print(len(adc_filepaths))