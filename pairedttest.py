from scipy import stats
import numpy as np

volumetric_spatial_error = np.load('./volumetric_spatial_error.npy')
W_nrm_pass_roi_error = np.load('./W_nrm_pass_roi_error.npy')
#print(stats.ttest_rel(volumetric_spatial_error,W_nrm_pass_roi_error, axis=0))

a=stats.norm.rvs(loc=5,scale=10,size=500)
print(W_nrm_pass_roi_error.shape)
