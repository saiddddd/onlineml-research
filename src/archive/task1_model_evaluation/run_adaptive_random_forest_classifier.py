from river import ensemble
from river import metrics
from tools.DataPreparation import AirlineDataPreparation

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from tqdm import tqdm
import json

train_feature, test_feature, train_target, test_target = \
    AirlineDataPreparation().get_splitted_train_test_pd_df_data(random_seed=42, training_dataset_ratio=0.1)

print('Training Features Shape: ', train_feature.shape)
print('Training Target Shape: ', train_target.shape)
print('Testing Features Shape: ', test_feature.shape)
print('Testing Target Shape: ', test_target.shape)

model = ensemble.AdaptiveRandomForestClassifier(
    n_models=2,
    max_depth=10,
    seed=0,
)

for index, raw in tqdm(train_feature.iterrows(), total=train_feature.shape[0]):
    model.learn_one(raw, train_target[index])

pred_result_list = []
for index, raw in tqdm(test_feature.iterrows(), total=test_feature.shape[0]):
    pred_result = model.predict_one(raw)
    print(pred_result)
    pred_result_list.append(pred_result)



acc = accuracy_score(pred_result_list, test_target)
print("accuracy {}%".format(acc))


print("Tree basic structure measurements")
trees = model.models


print(json.dumps(trees[0].model.model_measurements, indent=2, default=str))
print("Tree structure details inspection")
print(json.dumps(trees[0].model.model_description(), default=str))
g = trees[0].model.draw()
g.render("AdaRF_tree1_Structure_do_parallel", format='png')

print(json.dumps(trees[1].model.model_measurements, indent=2, default=str))
print("Tree structure details inspection")
print(json.dumps(trees[1].model.model_description(), default=str))
g = trees[1].model.draw()
g.render("AdaRF_tree2_Structure_do_parallel", format='png')