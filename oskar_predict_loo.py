import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, RFECV
from skimage.measure import regionprops, marching_cubes_classic, mesh_surface_area

y = np.load('./gt/ISLES2017_gt.npy')

oskar_features = np.load('./features/oskar_features.npy')

scaler = StandardScaler()
normalized_oskar_features = scaler.fit_transform(oskar_features)

sel = VarianceThreshold(0.85*(1-0.85))
selected_normalized_oskar_features = sel.fit_transform(normalized_oskar_features)

# Leave One Out Cross Validation
loo = LeaveOneOut()
estimator = RandomForestRegressor(n_estimators=300, max_depth=3, random_state=1989, n_jobs=-1)
rfecv = RFECV(estimator, step=1, cv=loo, scoring='neg_mean_absolute_error', n_jobs=-1)


X = rfecv.fit_transform(selected_normalized_oskar_features, y)


y_pred_label = np.zeros((37,1), dtype=np.float32)
y_abs_error = np.zeros((37,1), dtype=np.float32)
idx=0
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    estimator = RandomForestRegressor(n_estimators=300, max_depth=3, random_state=1989, n_jobs=-1)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    y_pred_label[idx] = np.round(y_pred)
    y_abs_error[idx] = np.absolute(y_pred_label[idx]-y_test)
    idx += 1

accouracy = accuracy_score(y, y_pred_label)

np.save('./rfecv_oskar_pred_loo.npy', y_pred_label)

print("Best Scores of features  - Using RF Classifier - Accuracy: %0.4f , MAE: %0.4f (+/- %0.4f)" %(accouracy, np.mean(y_abs_error), np.std(y_abs_error)))
