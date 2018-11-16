import utils

features, features_list = utils.extract_morphological_features()

print(features.shape, len(features_list))
