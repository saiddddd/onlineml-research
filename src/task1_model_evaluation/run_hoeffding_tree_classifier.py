from tools.DataPreparation import AirlineDataPreparation

from river import tree
from river import metrics

import matplotlib.pyplot as plt
from graphviz import Graph
from tqdm import tqdm
import numpy as np

train_feature, test_feature, train_target, test_target = AirlineDataPreparation().get_splitted_train_test_pd_df_data(random_seed=42)

print('Training Features Shape: ', train_feature.shape)
print('Training Target Shape: ', train_target.shape)
print('Testing Features Shape: ', test_feature.shape)
print('Testing Target Shape: ', test_target.shape)


# train_feature = train_feature.head(n=300)
# train_target = train_target.head(n=300)

# test_feature = np.array(test_feature)
# test_x = test_feature[0,:]

model = tree.HoeffdingTreeClassifier()

trained_data = 0
for index, raw in tqdm(train_feature.iterrows(), total=train_feature.shape[0]):
    model.learn_one(raw, train_target[index])
    trained_data+=1

    if trained_data == 2000:

        pass

    # if trained_data in [50, 300, 500, 1000, 5000, 10000, 20000, 30000, 70000]:
    if trained_data %10 == 0:
        g = model.draw()
        outputfile = './model_structure_draw/HT_classifier_structure_accumulation_data_{}'.format(trained_data)
        g.render(outputfile, format='png')

#%%

test_feature = test_feature.head(n=20)
test_target = test_target.head(n=20)

preds = []
target = []
pred_prob = []
tested_data = 0
for index, x in tqdm(test_feature.iterrows(), total=test_feature.shape[0]):
    preds.append(model.predict_one(x))
    pred_prob.append(model.predict_proba_one(x))
    target.append(test_target[index])
    # print(model.predict_proba_one(x))
    print(model.debug_one(x), test_target[index])
    aaa = model.model_description()
    # print(aaa)

    tested_data+=1

print(preds)
print(target)
print(pred_prob)

metric = metrics.Accuracy()
for yt, yp in zip(target, preds):
    metric = metric.update(yt, yp)
print(metric)
