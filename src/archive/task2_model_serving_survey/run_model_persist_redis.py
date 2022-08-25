import time

from tqdm import tqdm
import multiprocessing as mp
from river import tree

from tools.DataPreparation import AirlineDataPreparation

def persist_model(model):
    picked_model = pickle.dumps(model)
    r.set('test_model', picked_model)

import redis
import pickle

if __name__ == '__main__':
    r = redis.Redis(host='localhost', port=6379, db=0)

    model_river_HTC = tree.HoeffdingTreeClassifier()

    training_features, test_features, training_target, test_target = \
        AirlineDataPreparation().get_splitted_train_test_pd_df_data(
            training_dataset_ratio=0.7,
            random_seed=42
        )

    for index, raw in tqdm(training_features.iterrows(), total=training_features.shape[0]):
        model_river_HTC.learn_one(raw, training_target[index])



        persist_model(model_river_HTC)

    # for i in range(1000):
    #     acq_model_from_redis = pickle.loads(r.get('test_model'))
    #     time.sleep(1)

    # print(type(acq_model_from_redis))


