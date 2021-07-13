
from __future__ import annotations
from abc import ABC, abstractmethod
import copy

import traceback

import numpy as np
import pandas

from tools.DataPreparation import CreditCardPreparation, AirlineDataPreparation
from tools.DataVisualization import TrendPlot

from graphviz import Graph
import statistics

from river import tree
from river import preprocessing
from river import linear_model
from river import ensemble

from sklearn.ensemble import RandomForestClassifier

from tqdm import tqdm

class ModelExperimentWorkflow(ABC):
    """
    An Abstraction Factory interface declares a set of methods that represent complete
    model experiment and evaluation working pipeline.
    Base on the specific purpose, in case, training and probing model performance workflow
    will be slightly different.
    Concrete class will be implement for specific purpose.
    """

    def __init__(self, input_data_preparation_class, random_seed=42):
        """
        Initializing Model Performacne testing workflow.
        """
        if isinstance(
                 input_data_preparation_class,
                (AirlineDataPreparation, CreditCardPreparation)):
            pass
        else:
           raise TypeError

        '''
            initialization of default dataset.
            prepare:
             1. full set(without split into training-testing dataset)
             2. split training-testing dataset, the dataset can be used in general model training process
        '''
        #=============================================#
        # Prepare the full dataset, without splitting #
        #=============================================#
        self._full_data_features, self._full_data_target = input_data_preparation_class.get_pd_df_data()
        #===========================#
        # Prepare the split dataset #
        #===========================#
        self._train_features, self._test_features, self._train_target, self._test_target = \
            input_data_preparation_class.get_splitted_train_test_pd_df_data(training_dataset_ratio=0.7, random_seed=random_seed)

        self._model_init = None
        self._model = None

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def pred_model_by_test_dataset(self):
        pass

    def evaluate_model_get_accuracy_recall(self, pred_result=[]):
        # Simply Prediction
        test_target = np.array(self._test_target)
        correct_cnt = 0
        target_cnt = 0
        pred_target_cnt = 0
        pickup_target_cnt = 0
        for i in range(len(test_target)):
            if test_target[i] == 1:
                target_cnt += 1
                if pred_result[i] == test_target[i]:
                    pickup_target_cnt += 1
            if pred_result[i] == 1:
                pred_target_cnt += 1
            if pred_result[i] == test_target[i]:
                correct_cnt += 1

        print("Total Testing Data: {}".format(len(test_target)))
        print("Total correct predict: {}".format(correct_cnt))
        print("Acc: {}".format(correct_cnt / len(test_target) * 100))
        print("Total High Risk events: ", str(target_cnt))
        print("Predict High Risk events: ", str(pred_target_cnt))
        print("Picked Up High Risk events: ", str(pickup_target_cnt))
        print("Target recall rate: {} %".format(pickup_target_cnt / target_cnt * 100))

        return correct_cnt / len(test_target), pickup_target_cnt / target_cnt

    def get_model_type(self):
        return type(self._model)

    def set_model(self, model):
        self._model_init = model
        self._model = copy.copy(self._model_init)

    def reset_model(self):
        self._model = copy.copy(self._model_init)

    def limited_rows_full_dataset(self, limit_num=100) -> pandas.DataFrame:
        """
        return limited rows of training dataset for specific use.
        Base on Full Dataset self._full_data_feature and self._full_data_target
        :param limit_num: default 100
        :return: limited_row_feature, limited_row_target
        """
        if type(limit_num) is not int:
            raise TypeError
        else:
            limited_row_features = self._full_data_features.head(n=limit_num)
            limited_row_target = self._full_data_target.head(n=limit_num)
            return limited_row_features, limited_row_target

    def limited_rows_split_train_dataset(self, limit_num=100) -> pandas.DataFrame:
        """
        return limited rows of training dataset for specific use.
        Base on Split Dataset self._train_features and self._train_target
        :param limit_num:
        :return:
        """
        if type(limit_num) is not int:
            raise TypeError
        else:
            limited_row_features = self._train_features.head(n=limit_num)
            limited_row_target = self._train_target.head(n=limit_num)
            return limited_row_features, limited_row_target

    def subset_rows_split_train_dataset(self, start_row=int, end_row=int) -> pandas.DataFrame:

        subset_row_features = self._train_features[start_row:end_row]
        subset_row_target = self._train_target[start_row:end_row]
        return subset_row_features, subset_row_target


class ModelSklearnWorkflow(ModelExperimentWorkflow):

    def train_model(self, train_num_limit=int):

        data_for_train, target_for_train = self.limited_rows_split_train_dataset(train_num_limit)
        print('Training Features Shape: ', data_for_train.shape)
        print('Training Target Shape: ', target_for_train.shape)

        self._model.fit(data_for_train, target_for_train)

    def pred_model_by_test_dataset(self):

        print('Testing Features Shape: ', self._test_features.shape)
        print('Testing Target Shape: ', self._test_target.shape)

        # self._pred_result = self._model.predict(self._test_features)
        pred_result = self._model.predict(self._test_features)
        return pred_result


class ModelRiverOnlineMLWorkflow(ModelExperimentWorkflow):

    def train_model(self, is_reset_mode=True, train_num_limit=int):
        """
        training model method on RiverML online learning implementation has slightly different from sklearn
        here have to check model is going to do incremental learning or reset model (re-train)
        :param is_reset_mode: will this training process reset the model(not incremental)? Default=True
        :param train_num_limit:
        :return:
        """
        data_for_train, target_for_train = self.limited_rows_split_train_dataset(train_num_limit)
        print('Training Features Shape: ', data_for_train.shape)
        print('Training Target Shape: ', target_for_train.shape)

        #-------------------------------------------------#
        # Using River ML model to do online training      #
        # Core section, the different from SK learn model #
        #-------------------------------------------------#
        if is_reset_mode:
            # reset model(not incremental from previous step)
            self.reset_model()

        for index, raw in tqdm(data_for_train.iterrows(), total=data_for_train.shape[0]):
            self._model.learn_one(raw, target_for_train[index])

    def pred_model_by_test_dataset(self):
        hf_pred = []
        for index, raw in tqdm(self._test_features.iterrows(), total=self._test_features.shape[0]):
            test_pred = self._model.predict_one(raw)
            hf_pred.append(test_pred)

        return hf_pred


if __name__ == '__main__':

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=0
    )

    # sklearn_batch_process = ModelSklearnWorkflow(AirlineDataPreparation())
    # sklearn_batch_process.set_model(model)
    # model_type = sklearn_batch_process.get_model_type()
    # print("sklearn check model type")
    # print(model_type)
    # sklearn_batch_process.train_model(train_num_limit=500)
    # pred_result = sklearn_batch_process.pred_model_by_test_dataset()
    # acc, recall = sklearn_batch_process.evaluate_model_get_accuracy_recall(pred_result=pred_result)


    model_river_online_ml = tree.HoeffdingTreeClassifier()
    river_online_ml_process = ModelRiverOnlineMLWorkflow(AirlineDataPreparation(), random_seed=43)
    river_online_ml_process.set_model(model_river_online_ml)

    river_online_ml_process.train_model(is_reset_mode=True, train_num_limit=-1)
    pred_result = river_online_ml_process.pred_model_by_test_dataset()
    acc, recall = river_online_ml_process.evaluate_model_get_accuracy_recall(pred_result=pred_result)
