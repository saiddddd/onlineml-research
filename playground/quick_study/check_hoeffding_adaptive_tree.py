from river import tree
from river import ensemble

from sklearn.ensemble import RandomForestClassifier

import pandas as pd
from tqdm import tqdm

from sklearn.metrics import accuracy_score

def calculate_slicing_acc(pred_list, target_list, window_size, step=10):
    accumulating_acc = []

    if len(pred_list) != len(target_list):
        raise RuntimeWarning("length of prediction list and target list is not comparable")

    segment_start = 0
    segment_end = segment_start + window_size

    while segment_end < len(pred_list):

        acc = accuracy_score(pred_list[segment_start:segment_end], target_list[segment_start:segment_end])
        accumulating_acc.append(acc)

        segment_start += step
        segment_end = segment_start + window_size

    return accumulating_acc


# Initializing Model
# model_river = tree.HoeffdingAdaptiveTreeClassifier(
#             max_depth=3,
#             split_criterion='gini',
#             split_confidence=1e-2,
#             grace_period=10,
#             seed=0
#             # drift_window_threshold=300,
#             # adwin_confidence=0.002
#         )

model_sklearn = RandomForestClassifier(
            n_jobs=-1,
            n_estimators=10,
            max_depth=3,
            criterion='gini',
            random_state=42,
            verbose=1
        )

model_river = ensemble.AdaBoostClassifier(
            model=(
                tree.HoeffdingAdaptiveTreeClassifier(
                    max_depth=3,
                    split_criterion='gini',
                    split_confidence=1e-2,
                    grace_period=10,
                    seed=0
                )
            ),
            n_models=10,
            seed=42
        )

# input data
in_df = pd.read_csv("./dummy_toy/dummy_data.csv")
target = in_df.pop("Y").astype(int)

# sklearn model evaluation (batch training and prediction)
batch_training = in_df[:2000]
batch_target = target[:2000]

model_sklearn.fit(batch_training, batch_target)
sklearn_pred_proba = model_sklearn.predict_proba(in_df)[:, 1]
sklearn_pred_ans = [1 if i > 0.5 else 0 for i in sklearn_pred_proba]

# End of Batch Model evaluation

true_class_proba = []
false_class_proba = []
pred_ans = []
true_answer = []

for index, raw in tqdm(in_df.iterrows(), total=10000):

    # train model
    model_river.learn_one(raw, target[index])

    # prediction by features
    pred_proba = model_river.predict_proba_one(raw).get(1)

    if pred_proba is not None:

        true_answer.append(target[index])

        # casting probability to true/false answer
        # casting_proba = 1 if pred_proba > 0.5 else 0
        pred_ans.append(1 if pred_proba > 0.5 else 0)

        # accumulating prediction probability
        if target[index] == 1:
            true_class_proba.append(pred_proba)
        else:
            false_class_proba.append(pred_proba)


    # if index % 50 == 0:
    #     g = model_river.draw()
    #     g.attr(label=str(index))
    #     g.render("check_model_river/AdaHT_tree{}_Structure_after_online_learning".format(index), format='png')


sklearn_tracing_acc = calculate_slicing_acc(sklearn_pred_ans, target, window_size=300, step=10)
river_tracing_acc = calculate_slicing_acc(pred_ans, true_answer, window_size=300, step=10)

import matplotlib.pyplot as plt

x_list = [i*10 for i in range(len(river_tracing_acc))]
plt.plot(x_list, river_tracing_acc)
plt.plot(x_list[200:], sklearn_tracing_acc[200:])

plt.show()