import os
import shutil
import subprocess
import csv
import paths

def MoveLesionsMNIMask(isles_train_path, dst_dir):
	lesion_mni_path = [os.path.join(root, name) for root, dirs, files in os.walk(isles_train_path) for name in files if 'MNI152_T1_1mm' in name and 'prob' not in name and name.endswith('nii.gz')]
	for src in lesion_mni_path:
		lesion = os.path.split(src)[1]
		print(lesion)
		dst = os.path.join(dst_dir, lesion)
		shutil.copy(src, dst)

work_dir = paths.dsi_studio_path

dst_dir = os.path.join(work_dir, 'gt_stroke')

if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)

#MoveLesionsMNIMask(paths.isles2017_training_dir, dst_dir)

network_measures_dir = os.path.join(work_dir, 'network_measures')

if not os.path.exists(network_measures_dir):
    os.mkdir(network_measures_dir)

if not os.path.exists(os.path.join(network_measures_dir, 'gt_stroke')):
    os.mkdir(os.path.join(network_measures_dir, 'gt_stroke'))

connectogram_dir = os.path.join(work_dir, 'connectogram')

if not os.path.exists(connectogram_dir):
    os.mkdir(connectogram_dir)

if not os.path.exists(os.path.join(connectogram_dir, 'gt_stroke')):
    os.mkdir(os.path.join(connectogram_dir, 'gt_stroke'))

connectivity_dir = os.path.join(work_dir, 'connectivity')

if not os.path.exists(connectivity_dir):
    os.mkdir(connectivity_dir)

if not os.path.exists(os.path.join(connectivity_dir, 'gt_stroke')):
    os.mkdir(os.path.join(connectivity_dir, 'gt_stroke'))

source ='HCP1021.1mm.fib.gz'

assert (os.path.exists(os.path.join(work_dir, source))), 'HCP1021 template is not in the dsi studio directory'

# 2,000,000 tracts
#parameter_id = '--parameter_id=3D69233E9A99193F32318D24ba3Fba3Fb404b0FA4380841Eca01cb01d'

# 1,000,000 tracts
#parameter_id = '--parameter_id=3D69233E9A99193F32318D24ba3Fba3Fb404b0FA4340420Fca01dcba'

# 500,000 tracts
#parameter_id = '--parameter_id=3D69233E9A99193F32318D24ba3Fba3Fb404b0FA4320A107ca01cb01d'

# 250,000 tracts
#parameter_id = '--parameter_id=3D69233E9A99193F32318D24ba3Fba3Fb404b0FA4390Da3ca01cb01d'

# 125,000 tracts
#parameter_id = '--parameter_id=3D69233E9A99193F32318D24ba3Fba3Fb404b0FA4348E801ca01cb01d'

# 62,500 tracts
#parameter_id = '--parameter_id=3D69233E9A99193F32318D24ba3Fba3Fb404b0FA4324F4cb01cb01d'

# 31,250 tracts
#parameter_id = '--parameter_id=3D69233E9A99193F32318D24ba3Fba3Fb404b0FA43127Acb01cb01d'

# 15,626 tracts
parameter_id = '--parameter_id=3D69233E9A99193F32318D24ba3Fba3Fb404b0FA43093Dcb01cb01d'

os.chdir(work_dir)

stroke_dir = dst_dir
stroke_files_dir = os.listdir(dst_dir)
stroke_files_dir.sort()
assert(len(stroke_files_dir)==43)

region_prop ='--roi='

# pass type of connectivity matrices
for idx, stroke_file in enumerate(stroke_files_dir):

    pat_name = stroke_file[:stroke_file.find('_MNI152_T1_1mm.nii.gz')]
    print(idx, pat_name)
    #roi = '--roi='+os.path.join(stroke_dir, stroke_file)
    #seed = '--seed='+os.path.join(stroke_dir, stroke_file)
    connectivity_type = '--connectivity_type=pass'
    connectivity_value = '--connectivity_value=count'
    connectivity_threshold = '--connectivity_threshold=0'
    #subprocess.call(['./dsi_studio', '--action=trk', '--source='+source, seed, roi, parameter_id, '--output=no_file', '--connectivity=aal', connectivity_type, connectivity_value, connectivity_threshold])
    subprocess.call(['./dsi_studio', '--action=trk', '--source='+source, region_prop+os.path.join(stroke_dir, stroke_file), parameter_id, '--output=no_file', '--connectivity=aal', connectivity_type, connectivity_value, connectivity_threshold])

    network_measure_files = [os.path.join(root, name) for root, dirs, files in os.walk(work_dir) for name in files if 'network_measures' in name and name.endswith('.txt')]
    network_measure_file_dst = os.path.join(os.path.split(network_measure_files[0])[0], 'network_measures', 'gt_stroke', os.path.split(network_measure_files[0].replace(source, pat_name))[1])
    shutil.move(network_measure_files[0], network_measure_file_dst) 

    connectogram_files = [os.path.join(root, name) for root, dirs, files in os.walk(work_dir) for name in files if 'connectogram' in name and name.endswith('.txt')]
    connectogram_file_dst = os.path.join(os.path.split(connectogram_files[0])[0], 'connectogram', 'gt_stroke', os.path.split(connectogram_files[0].replace(source, pat_name))[1])
    shutil.move(connectogram_files[0], connectogram_file_dst)

    connectivity_files = [os.path.join(root, name) for root, dirs, files in os.walk(work_dir) for name in files if 'connectivity' in name and name.endswith('.mat')]
    connectivity_file_dst = os.path.join(os.path.split(connectivity_files[0])[0], 'connectivity', 'gt_stroke', os.path.split(connectivity_files[0].replace(source, pat_name))[1])
    shutil.move(connectivity_files[0], connectivity_file_dst)

# end type of connectivity matrices
for idx, stroke_file in enumerate(stroke_files_dir):

    pat_name = stroke_file[:stroke_file.find('_MNI152_T1_1mm.nii.gz')]
    print(idx, pat_name)
    #roi = '--roi='+os.path.join(stroke_dir, stroke_file)
    #seed = '--seed='+os.path.join(stroke_dir, stroke_file)
    connectivity_type = '--connectivity_type=end'
    connectivity_value = '--connectivity_value=count'
    connectivity_threshold = '--connectivity_threshold=0'
    #subprocess.call(['./dsi_studio', '--action=trk', '--source='+source, seed, roi, parameter_id, '--output=no_file', '--connectivity=aal', connectivity_type, connectivity_value, connectivity_threshold])
    subprocess.call(['./dsi_studio', '--action=trk', '--source='+source, region_prop+os.path.join(stroke_dir, stroke_file), parameter_id, '--output=no_file', '--connectivity=aal', connectivity_type, connectivity_value, connectivity_threshold])

    network_measure_files = [os.path.join(root, name) for root, dirs, files in os.walk(work_dir) for name in files if 'network_measures' in name and name.endswith('.txt')]
    network_measure_file_dst = os.path.join(os.path.split(network_measure_files[0])[0], 'network_measures', 'gt_stroke', os.path.split(network_measure_files[0].replace(source, pat_name))[1])
    shutil.move(network_measure_files[0], network_measure_file_dst) 

    connectogram_files = [os.path.join(root, name) for root, dirs, files in os.walk(work_dir) for name in files if 'connectogram' in name and name.endswith('.txt')]
    connectogram_file_dst = os.path.join(os.path.split(connectogram_files[0])[0], 'connectogram', 'gt_stroke', os.path.split(connectogram_files[0].replace(source, pat_name))[1])
    shutil.move(connectogram_files[0], connectogram_file_dst)

    connectivity_files = [os.path.join(root, name) for root, dirs, files in os.walk(work_dir) for name in files if 'connectivity' in name and name.endswith('.mat')]
    connectivity_file_dst = os.path.join(os.path.split(connectivity_files[0])[0], 'connectivity', 'gt_stroke', os.path.split(connectivity_files[0].replace(source, pat_name))[1])
    shutil.move(connectivity_files[0], connectivity_file_dst)
