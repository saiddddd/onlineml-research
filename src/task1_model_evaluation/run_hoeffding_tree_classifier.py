import time
import json

from tools.DataPreparation import AirlineDataPreparation

from river import tree
from river import metrics

import matplotlib.pyplot as plt
from tqdm import tqdm

"""
Inspection of basic Hoeffding Tree structure and node split effect by memory limitation.
"""

train_feature, test_feature, train_target, test_target = \
    AirlineDataPreparation().get_splitted_train_test_pd_df_data(random_seed=42)

print('Training Features Shape: ', train_feature.shape)
print('Training Target Shape: ', train_target.shape)
print('Testing Features Shape: ', test_feature.shape)
print('Testing Target Shape: ', test_target.shape)


model = tree.HoeffdingTreeClassifier(
    # max_size=0.5,
    memory_estimate_period=200
)
# model = ensemble.AdaptiveRandomForestClassifier()


trained_data = 0
memory_usage_track = []
time_spend_track = []
data_acc_num = []

time_stamp = time.time()
for index, raw in tqdm(train_feature.iterrows(), total=train_feature.shape[0]):

    model.learn_one(raw, train_target[index])

    trained_data+=1

    if trained_data == 1:
        time_stamp = time.time()

    # if trained_data in [50, 300, 500, 1000, 5000, 10000, 20000, 30000, 70000]:
    if trained_data %200 == 0:
        time_spend_track.append(time.time() - time_stamp)
        # g = model.draw()
        # outputfile = './model_structure_draw/HT_classifier_structure_accumulation_data_{}'.format(trained_data)
        # g.render(outputfile, format='png')
        # print("Tree basic structure measurements")
        # print(json.dumps(model.model_measurements, indent=2, default=str))
        # print("Tree structure details inspection")
        # print(json.dumps(model.model_description(), default=str))
        raw_memory, unit = float(model._memory_usage[:-3]), model._memory_usage[-2:]
        memory_usage_track.append(raw_memory * 2 ** -10 if unit == 'KB' else raw_memory)
        data_acc_num.append(trained_data)
        time_stamp = time.time()

    # if trained_data > 1000:
    #     break

print("Tree basic structure measurements")
print(json.dumps(model.model_measurements, indent=2, default=str))
print("Tree structure details inspection")
print(json.dumps(model.model_description(), default=str))
g = model.draw()
g.render("./src/task1_model_evaluation/HT_structure", format='png')

print(memory_usage_track)
print(time_spend_track)

fig, ax = plt.subplots(figsize=(5, 5), nrows=2, dpi=300)
ax[0].grid(True)
ax[1].grid(True)

ax[0].plot(data_acc_num, time_spend_track)
ax[0].set_ylabel('Time (seconds)')
plt.setp(ax[0].get_xticklabels(), visible=False)

ax[1].plot(data_acc_num, memory_usage_track, label="Hoeffding Tree Classifier")
ax[1].set_ylabel('Memory (MB)')
ax[1].set_xlabel('# data')
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

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
