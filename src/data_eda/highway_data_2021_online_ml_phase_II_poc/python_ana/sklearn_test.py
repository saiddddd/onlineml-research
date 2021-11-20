# %%
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('highway_traffic_analized_data_ready_for_ML.csv')

# %%

data.columns

# %%

feature_x = data

for col in feature_x:
    if feature_x[col].dtype == 'object':
        feature_x[col] = LabelEncoder().fit_transform(feature_x[col])
# aaa = feature_x.pop('國道別')
# aaa = feature_x.pop('TrafficJam')
# aaa = feature_x.pop('ICNUM')
# aaa = feature_x.pop('方向')
target_y = feature_x.pop('TrafficJam30MinLater')

# %%
feature_x = feature_x.fillna(0)
feature_x
#%%
target_y
# %%
traing_x, testing_x, traing_y, testing_y = train_test_split(feature_x, target_y, test_size=0.3)
# %%
model_sklearn = RandomForestClassifier(
            n_estimators=100,
            criterion="entropy",
            max_depth=70,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

model_sklearn.fit(traing_x, traing_y)


# %%
pred_result = model_sklearn.predict_proba(testing_x)

pred_result_true_subclass = pred_result[testing_y == 1][:, 1]
pred_result_false_subclass = pred_result[testing_y == 0][:, 1]

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 4))
plt.suptitle('highway_traffic_pred_proba_distribution'+"_sklearn")
plt.subplot(131)
plt.hist(pred_result_true_subclass, bins=50, alpha=0.5, label='Y True')
plt.hist(pred_result_false_subclass, bins=50, alpha=0.5, label='Y False')
plt.yscale('log')
plt.title('stacking prediction proba in both class')
plt.xlabel('pred proba')
plt.ylabel('statistics')
plt.grid()
plt.legend()
plt.subplot(132)
plt.hist(pred_result_true_subclass, bins=50)
plt.yscale('log')
plt.xlabel('pred proba')
plt.ylabel('statistics')
plt.grid()
plt.subplot(133)
plt.hist(pred_result_false_subclass, bins=50)
plt.yscale('log')
plt.grid()
plt.xlabel('pred proba')
plt.ylabel('statistics')
plt.savefig("highway_pred_proba_distribution" + "_.pdf")
# %%
