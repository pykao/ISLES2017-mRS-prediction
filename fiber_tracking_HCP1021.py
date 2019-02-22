import os
import shutil
import subprocess
import csv
import paths
from utils import ReadImage
import numpy as np

# Change the working directory to dsistudio 
work_dir = paths.dsi_studio_path
os.chdir(work_dir)

source ='HCP1021.1mm.fib.gz'

connectivity_type = '--connectivity_type=end'
connectivity_value = '--connectivity_value=count'
connectivity_threshold = '--connectivity_threshold=0'
number_of_seed = 2235858
subprocess.call(['./dsi_studio', '--action=trk', '--source='+source, '--seed_count='+str(number_of_seed), '--fa_threshold=0.15958', '--seed_plan=1', '--initial_dir=2', '--interpolation=0', '--turning_angle=90.0', '--step_size=.5', '--smoothing=.5', '--min_length=3', '--max_length=500', '--tip_iteration=1', '--thread_count=8', '--output=no_file', '--connectivity=aal', connectivity_type, connectivity_value, connectivity_threshold])