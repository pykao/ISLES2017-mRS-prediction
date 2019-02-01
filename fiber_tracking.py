import os
import shutil
import subprocess
import csv
import paths
from utils import ReadImage
import numpy as np

def MoveLesionsMNIMask(isles_train_path, dst_dir):
    '''Move lesion in MNI251 space from isles_train_path to dst_dir'''
    lesion_mni_path = [os.path.join(root, name) for root, dirs, files in os.walk(isles_train_path) for name in files if 'MNI152_T1_1mm' in name and 'prob' not in name and name.endswith('nii.gz')]
    for src in lesion_mni_path:
        lesion = os.path.split(src)[1]
        print('Moving ', lesion)
        dst = os.path.join(dst_dir, lesion)
        shutil.copy(src, dst)

# Change the working directory to dsistudio 
work_dir = paths.dsi_studio_path

# create a folder called gt_stroke to store all stroke lesion in MNI152 space
dst_dir = os.path.join(work_dir, 'gt_stroke')
if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)
# move the strokes in MNI space to gt_stroke directory
if not os.listdir(dst_dir):
    MoveLesionsMNIMask(paths.isles2017_training_dir, dst_dir)

# create a folder to save network measures
network_measures_dir = os.path.join(work_dir, 'network_measures')

if not os.path.exists(network_measures_dir):
    os.mkdir(network_measures_dir)

if not os.path.exists(os.path.join(network_measures_dir, 'gt_stroke')):
    os.mkdir(os.path.join(network_measures_dir, 'gt_stroke'))


# create a folder to save connectogram
connectogram_dir = os.path.join(work_dir, 'connectogram')

if not os.path.exists(connectogram_dir):
    os.mkdir(connectogram_dir)

if not os.path.exists(os.path.join(connectogram_dir, 'gt_stroke')):
    os.mkdir(os.path.join(connectogram_dir, 'gt_stroke'))

# create a folder to save connectivity matrix
connectivity_dir = os.path.join(work_dir, 'connectivity')

if not os.path.exists(connectivity_dir):
    os.mkdir(connectivity_dir)

if not os.path.exists(os.path.join(connectivity_dir, 'gt_stroke')):
    os.mkdir(os.path.join(connectivity_dir, 'gt_stroke'))

# The average fiber tracts from 1021 HCP subjects
source ='HCP1021.1mm.fib.gz'

assert (os.path.exists(os.path.join(work_dir, source))), 'HCP1021 template is not in the dsi studio directory'

# tracking parameter
#parameter_id = '--parameter_id=3D69233E9A99193F32318D24ba3Fba3Fb404b0FA4340420Fca01cb01d'

# change the working directory
os.chdir(work_dir)

# find all stroke files and sort them
stroke_dir = dst_dir
stroke_files_dir = os.listdir(dst_dir)
stroke_files_dir.sort()
assert(len(stroke_files_dir)==43)


# region property
# seed in the whole brain region and find the tracts passing through stroke lesion
region_prop ='--roi='
# seed in the stroke lesion regions and find the tracts passing through stroke lesion
#region_prop = '--seed='

#region_prop ='--roa='

# pass type of connectivity matrices
for idx, stroke_file in enumerate(stroke_files_dir):

    pat_name = stroke_file[:stroke_file.find('_MNI152_T1_1mm.nii.gz')]
    print(idx, 'Working on creating pass-type of connectivity matrix of ', pat_name)
    #stroke_nda = ReadImage(os.path.join(stroke_dir, stroke_file))
    #number_of_seed = np.count_nonzero(stroke_nda)
    #number_of_seed = 964748
    number_of_seed = 2235858
    # connectivity matrix's parameters
    connectivity_type = '--connectivity_type=pass'
    connectivity_value = '--connectivity_value=count'
    connectivity_threshold = '--connectivity_threshold=0'
    #subprocess.call(['./dsi_studio', '--action=trk', '--source='+source, region_prop+os.path.join(stroke_dir, stroke_file), parameter_id,'--seed_count='+str(number_of_seed), '--output=no_file', '--connectivity=aal', connectivity_type, connectivity_value, connectivity_threshold])
    
    # fiber tracking and setting the tracking parameters
    subprocess.call(['./dsi_studio', '--action=trk', '--source='+source, region_prop+os.path.join(stroke_dir, stroke_file), '--seed_count='+str(number_of_seed), '--fa_threshold=0.15958', '--seed_plan=1', '--initial_dir=2', '--interpolation=0', '--turning_angle=90.0', '--step_size=.5', '--smoothing=.5', '--min_length=3', '--max_length=500', '--tip_iteration=0', '--thread_count=8', '--output=no_file', '--connectivity=aal', connectivity_type, connectivity_value, connectivity_threshold])
    # move the files
    network_measure_files = [os.path.join(root, name) for root, dirs, files in os.walk(work_dir) for name in files if 'network_measures' in name and name.endswith('.txt')]
    network_measure_file_dst = os.path.join(os.path.split(network_measure_files[0])[0], 'network_measures', 'gt_stroke', os.path.split(network_measure_files[0].replace(source, pat_name))[1])
    shutil.move(network_measure_files[0], network_measure_file_dst) 

    connectogram_files = [os.path.join(root, name) for root, dirs, files in os.walk(work_dir) for name in files if 'connectogram' in name and name.endswith('.txt')]
    connectogram_file_dst = os.path.join(os.path.split(connectogram_files[0])[0], 'connectogram', 'gt_stroke', os.path.split(connectogram_files[0].replace(source, pat_name))[1])
    shutil.move(connectogram_files[0], connectogram_file_dst)

    connectivity_files = [os.path.join(root, name) for root, dirs, files in os.walk(work_dir) for name in files if 'connectivity' in name and name.endswith('.mat')]
    connectivity_file_dst = os.path.join(os.path.split(connectivity_files[0])[0], 'connectivity', 'gt_stroke', os.path.split(connectivity_files[0].replace(source, pat_name))[1])
    shutil.move(connectivity_files[0], connectivity_file_dst)
    print('---'*30)


# end type of connectivity matrices
for idx, stroke_file in enumerate(stroke_files_dir):

    pat_name = stroke_file[:stroke_file.find('_MNI152_T1_1mm.nii.gz')]
    print(idx, 'Working on creating end-type of connectivity matrix of ', pat_name)
    #stroke_nda = ReadImage(os.path.join(stroke_dir, stroke_file))
    #number_of_seed = np.count_nonzero(stroke_nda)
    #number_of_seed = 964748
    number_of_seed = 2235858
    # connectivity matrix's parameters
    connectivity_type = '--connectivity_type=end'
    connectivity_value = '--connectivity_value=count'
    connectivity_threshold = '--connectivity_threshold=0'
    
    # fiber tracking and setting the tracking parameters
    #subprocess.call(['./dsi_studio', '--action=trk', '--source='+source, region_prop+os.path.join(stroke_dir, stroke_file), parameter_id, '--output=no_file', '--connectivity=aal', connectivity_type, connectivity_value, connectivity_threshold])
    subprocess.call(['./dsi_studio', '--action=trk', '--source='+source, region_prop+os.path.join(stroke_dir, stroke_file), '--seed_count='+str(number_of_seed), '--fa_threshold=0.15958', '--seed_plan=1', '--initial_dir=2', '--interpolation=0', '--turning_angle=90.0', '--step_size=.5', '--smoothing=.5', '--min_length=3', '--max_length=500', '--tip_iteration=0' '--thread_count=8', '--output=no_file', '--connectivity=aal', connectivity_type, connectivity_value, connectivity_threshold])

    network_measure_files = [os.path.join(root, name) for root, dirs, files in os.walk(work_dir) for name in files if 'network_measures' in name and name.endswith('.txt')]
    network_measure_file_dst = os.path.join(os.path.split(network_measure_files[0])[0], 'network_measures', 'gt_stroke', os.path.split(network_measure_files[0].replace(source, pat_name))[1])
    shutil.move(network_measure_files[0], network_measure_file_dst) 

    connectogram_files = [os.path.join(root, name) for root, dirs, files in os.walk(work_dir) for name in files if 'connectogram' in name and name.endswith('.txt')]
    connectogram_file_dst = os.path.join(os.path.split(connectogram_files[0])[0], 'connectogram', 'gt_stroke', os.path.split(connectogram_files[0].replace(source, pat_name))[1])
    shutil.move(connectogram_files[0], connectogram_file_dst)

    connectivity_files = [os.path.join(root, name) for root, dirs, files in os.walk(work_dir) for name in files if 'connectivity' in name and name.endswith('.mat')]
    connectivity_file_dst = os.path.join(os.path.split(connectivity_files[0])[0], 'connectivity', 'gt_stroke', os.path.split(connectivity_files[0].replace(source, pat_name))[1])
    shutil.move(connectivity_files[0], connectivity_file_dst)
    print('---'*30)
