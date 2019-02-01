import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, RFECV
from skimage.measure import regionprops, marching_cubes_classic, mesh_surface_area
from utils import extract_gt_mRS

y = np.load('./ISLES2017_gt.npy')

oskar_features = np.load('./oskar_ISLES2016_features.npy')

sel = VarianceThreshold(0.5*(1-0.5))
selected_oskar_features = sel.fit_transform(oskar_features)

scaler = StandardScaler()
normalized_selected_oskar_features = scaler.fit_transform(selected_oskar_features)

# Leave One Out Cross Validation
loo = LeaveOneOut()
#estimator = RandomForestRegressor(n_estimators=300, max_depth=3, random_state=1989, n_jobs=-1)
#rfecv = RFECV(estimator, step=1, cv=loo, scoring='neg_mean_absolute_error', n_jobs=-1)


#X = rfecv.fit_transform(normalized_selected_oskar_features, y)
X = normalized_selected_oskar_features
#X = all_features

y_pred_label = np.zeros((37,1), dtype=np.float32)
y_abs_error = np.zeros((37,1), dtype=np.float32)
idx=0
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(y_test)
    estimator = RandomForestRegressor(n_estimators=300, max_depth=3, random_state=1989, n_jobs=-1)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    y_pred_label[idx] = np.round(y_pred)
    y_abs_error[idx] = np.absolute(y_pred_label[idx]-y_test)
    idx += 1

accouracy = accuracy_score(y, y_pred_label)

np.save('./rfr_oskarISLES2016.npy', y_pred_label)

print("Best Scores of features  - Using RF Classifier - Accuracy: %0.4f , MAE: %0.4f (+/- %0.4f)" %(accouracy, np.mean(y_abs_error), np.std(y_abs_error)))
