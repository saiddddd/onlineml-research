from datetime import datetime
import pandas as pd

from river import ensemble
from river import tree

from sklearn.metrics import accuracy_score, mean_absolute_error

from tqdm import tqdm

from tools.data_loader import GeneralDataLoader
from tools.evaluator import MeanAbsoluteErrorEvaluator

import time

class Model:

    def __init__(self, model_class=None, model_hyper_params=None):

        self._model = model_class(
            **model_hyper_params
        )

        # ML model management
        self._model_version = 0
        self._model_model_timestamp = self.get_timestamp_now()

    @staticmethod
    def get_timestamp_now():
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def fit(self, x, y):
        # self._model_version += 1
        # self._model_model_timestamp = self.get_timestamp_now()

        raise NotImplementedError


    def get_model(self):
        return self._model


class RiverClassifier(Model):

    def __init__(self, model_class, **model_hyper_params):
        super().__init__(model_class=model_class, **model_hyper_params)

        # self._model = ensemble.AdaBoostClassifier(
        #     model=(
        #         tree.HoeffdingAdaptiveTreeClassifier(
        #             max_depth=3,
        #             split_criterion='gini',
        #             split_confidence=1e-2,
        #             grace_period=10,
        #             seed=0
        #         )
        #     ),
        #     n_models=10,
        #     seed=42
        # )

    def model_wrap(self, model_encapsulate, model_hyper_params):

        self._model = model_encapsulate(
            model=(self._model),
            **model_hyper_params
        )
        # self._model = ensemble.AdaBoostClassifier(
        #     model=(
        #         tree.HoeffdingAdaptiveTreeClassifier(
        #             max_depth=3,
        #             split_criterion='gini',
        #             split_confidence=1e-2,
        #             grace_period=10,
        #             seed=0
        #         )
        #     ),
        #     n_models=10,
        #     seed=42
        # )

        return self

    def fit(self, x, y):

        if isinstance(x, pd.DataFrame):
            for index, row in tqdm(x.iterrows(), total=x.shape[0]):
                self._model.learn_one(row, y[index])
        elif isinstance(x, pd.Series):
            self._model.learn_one(x, y)

    def fit_one(self, x, y):
        time_start = time.time()
        self._model.learn_one(x, y)
        print(hex(id(self._model)))
        time_end = time.time()

        print(
            '\r Events Trained, learn_one time spend:{} milliseconds'.format(
                (time_end - time_start) * 1000),
            end='',
            flush=True
        )


    def inference(self, x):

        tot_pred = []
        for index, row in tqdm(x.iterrow(), total=x.shape[0]):
            tot_pred.append(self._model.predict_one(row))

        return tot_pred

    def inference_proba(self, x, proba_cut_point=0.5):

        pred_target_proba = []
        pred_is_target = []

        for index, raw in x.iterrows():

            result = self._model.predict_proba_one(raw)

            try:

                pred_target_proba.append(result.get(1))

                if proba_cut_point is not None:
                    if result.get(1) >= proba_cut_point:
                        pred_is_target.append(1)
                    else:
                        pred_is_target.append(0)

            except Exception as e:
                pred_target_proba.append(None)
                pred_is_target.append(None)
                e.with_traceback()

        return pred_target_proba, pred_is_target



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

    model_hyper_params = {
        'max_depth': 3,
        'split_criterion': 'gini',
        'split_confidence': 1e-2,
        'grace_period': 10,
        'seed': 0
    }

    model = RiverClassifier(model_class=tree.HoeffdingAdaptiveTreeClassifier, model_hyper_params=model_hyper_params)

    model_wrap_params = {
        'n_models': 100
    }

    model.model_wrap(
        model_encapsulate=ensemble.AdaBoostClassifier,
        model_hyper_params=model_wrap_params
    )

    dataloader = GeneralDataLoader("../../data/stock_index_predict/eda_TW50_top30_append_2010_2017.csv")
    dataloader.drop_feature('Adj Close')
    dataloader.drop_feature('DailyReturn')

    # model_hyper_params = {
    #     'grace_period': 50,
    #     'leaf_prediction': 'adaptive',
    #     'model_selector_decay': 0.3,
    #     'seed': 0
    # }
    #
    # model = RiverHoeffdingTreeRegressor(model_name="HoeffdingTreeRegressor", **model_hyper_params)
    # dataloader = GeneralDataLoader("../../data/wind-farm/windfarm_data.csv")
    df = dataloader.get_full_df()

    # DataFrame Type Casting
    for column in df.columns:
        df[column] = df[column].astype(float)

    y = df.pop('LABEL')

    model.fit(df, y)

    # mae = model.validate_mae(df, y)

    # print(mae)

