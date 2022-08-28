from datetime import datetime
import pandas as pd

from river import ensemble
from river import tree

from sklearn.metrics import accuracy_score, mean_absolute_error

from tqdm import tqdm

from tools.data_loader import GeneralDataLoader
from tools.evaluator import MeanAbsoluteErrorEvaluator

class Model:

    def __init__(self, model_name=''):

        self._model = None

        self._model_name = model_name
        self._model_hyper_params = {}

        # ML model management
        self._model_version = 0
        self._model_model_timestamp = self.get_timestamp_now()

    @staticmethod
    def get_timestamp_now():
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def init_model(self, model_class, params: dict):

        self._model = model_class(params)

        print(self._model)

    def fit(self, x, y):
        self._model_version += 1
        self._model_model_timestamp = self.get_timestamp_now()

        raise NotImplementedError

    def get_tree_g(self, tree_index: int):

        trees = self._model.models
        return trees[tree_index].draw()

    def __str__(self):
        return self._model_name+'_'+str(self._model_version)+'_'+self._model_model_timestamp


class RiverHoeffdingTreeClassifier(Model):

    def __init__(self, model_name, **params):
        super().__init__(model_name=model_name)

        self._model = ensemble.AdaBoostClassifier(
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


    def fit(self, x, y):

        for index, row in tqdm(x.iterrows(), total=x.shape[0]):
            self._model.learn_one(row, y[index])

    def inference(self, x):

        tot_pred = []
        for index, row in tqdm(x.iterrow(), total=x.shape[0]):
            tot_pred.append(self._model.predict_one(row))

        return tot_pred

    def validate_acc(self, x, y):

        tot_pred = self.inference(x)
        acc = accuracy_score(y, tot_pred)

        return acc


class RiverHoeffdingTreeRegressor(Model):

    def __init__(self, model_name, **params):
        super().__init__(model_name=model_name)

        self._model = tree.HoeffdingAdaptiveTreeRegressor(
            **params
        )

    def fit(self, x, y):

        for index, row in tqdm(x.iterrows(), total=x.shape[0]):
            self._model.learn_one(row.to_dict(), y[index])

    def inference(self, x):

        tot_pred = []
        for index, row in tqdm(x.iterrows(), total=x.shape[0]):
            tot_pred.append(self._model.predict_one(row))

        return tot_pred

    def validate_mae(self, x, y):
        tot_pred = self.inference(x)
        mae = MeanAbsoluteErrorEvaluator(y, tot_pred).get_evaluation()

        return mae


if __name__ == "__main__":

    # model = RiverHoeffdingTreeClassifier(model_name="HoeffdingTreeClassifier", params=model_hyper_params)
    # dataloader = GeneralDataLoader("../../data/stock_index_predict/eda_TW50_top30_append_2010_2017.csv")
    # dataloader.drop_feature('Adj Close')
    # dataloader.drop_feature('DailyReturn')

    model_hyper_params = {
        'grace_period': 50,
        'leaf_prediction': 'adaptive',
        'model_selector_decay': 0.3,
        'seed': 0
    }

    model = RiverHoeffdingTreeRegressor(model_name="HoeffdingTreeRegressor", **model_hyper_params)
    dataloader = GeneralDataLoader("../../data/wind-farm/windfarm_data.csv")
    df = dataloader.get_full_df()

    # DataFrame Type Casting
    for column in df.columns:
        df[column] = df[column].astype(float)

    y = df.pop('power')

    model.fit(df, y)

    mae = model.validate_mae(df, y)


    print(mae)

# def kwargs_fun(a=0, b=0, c=0):
#
#     print(a, b, c)
#
#
# def run_func(*args, **kwargs):
#
#     kwargs_fun(**kwargs)
#
#
# if __name__ == "__main__":
#
#     param = {
#         'a': 1,
#         'b': 2,
#         'c': 3
#     }
#
#     run_func(**param)
#     # kwargs_fun(**param)



